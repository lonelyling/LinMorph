import torch, torchvision, imageio, os, json, pandas
import imageio.v3 as iio
from PIL import Image
import boto3
import io
from io import BytesIO
from PIL import Image
import csv
import json
import random
import time
from itertools import islice
from functools import partial, lru_cache

# webdataset
import webdataset as wds
import json
from botocore.config import Config
from botocore.exceptions import ClientError


_DATASET_LENGTH_MAP = {
    'webdatas/laion2b-multi-chinese-subset.json': 2_000_000_000,
    'webdatas/coyo-700m.json': 700_000_000
}


def _env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


def _throttle_cos_read():
    stagger_sec = max(0.0, _env_float("DIFFSYNTH_COS_RANK_STAGGER_SEC", "0"))
    jitter_sec = max(0.0, _env_float("DIFFSYNTH_COS_READ_JITTER_SEC", "0"))
    if stagger_sec == 0.0 and jitter_sec == 0.0:
        return

    rank = _env_int("RANK", _env_int("LOCAL_RANK", "0"))
    world_size = max(1, _env_int("WORLD_SIZE", "1"))
    delay = stagger_sec * (rank % world_size)
    if jitter_sec > 0:
        delay += random.random() * jitter_sec
    if delay > 0:
        time.sleep(delay)


class DataProcessingPipeline:
    def __init__(self, operators=None):
        self.operators: list[DataProcessingOperator] = [] if operators is None else operators
        
    def __call__(self, data):
        for operator in self.operators:
            data = operator(data)
        return data
    
    def __rshift__(self, pipe):
        if isinstance(pipe, DataProcessingOperator):
            pipe = DataProcessingPipeline([pipe])
        return DataProcessingPipeline(self.operators + pipe.operators)



class DataProcessingOperator:
    def __call__(self, data):
        raise NotImplementedError("DataProcessingOperator cannot be called directly.")
    
    def __rshift__(self, pipe):
        if isinstance(pipe, DataProcessingOperator):
            pipe = DataProcessingPipeline([pipe])
        return DataProcessingPipeline([self]).__rshift__(pipe)



class DataProcessingOperatorRaw(DataProcessingOperator):
    def __call__(self, data):
        return data



class ToInt(DataProcessingOperator):
    def __call__(self, data):
        return int(data)



class ToFloat(DataProcessingOperator):
    def __call__(self, data):
        return float(data)



class ToStr(DataProcessingOperator):
    def __init__(self, none_value=""):
        self.none_value = none_value
    
    def __call__(self, data):
        if data is None: data = self.none_value
        return str(data)



class LoadImage(DataProcessingOperator):
    def __init__(self, convert_RGB=True):
        self.convert_RGB = convert_RGB
    
    def __call__(self, data):
        with Image.open(data) as img:
            if self.convert_RGB: 
                image = img.convert("RGB")
            else:
                image = img.copy()
        return image



class ImageCropAndResize(DataProcessingOperator):
    def __init__(self, height, width, max_pixels, height_division_factor, width_division_factor):
        self.height = height
        self.width = width
        self.max_pixels = max_pixels
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor

    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image
    
    def get_height_width(self, image):
        if self.height is None or self.width is None:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width
    
    
    def __call__(self, data: Image.Image):
        image = self.crop_and_resize(data, *self.get_height_width(data))
        return image



class ToList(DataProcessingOperator):
    def __call__(self, data):
        return [data]
    


class LoadVideo(DataProcessingOperator):
    def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1, frame_processor=lambda x: x):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        # frame_processor is build in the video loader for high efficiency.
        self.frame_processor = frame_processor
        
    def get_num_frames(self, reader):
        num_frames = self.num_frames
        # if int(reader.count_frames()) < num_frames:
        if int(len(reader)) < num_frames:
            #num_frames = int(reader.count_frames())
            num_frames = int(len(reader))
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames
        
    def __call__(self, data: str):
        reader = imageio.get_reader(data)
        num_frames = self.get_num_frames(reader)
        frames = []
        for frame_id in range(num_frames):
            frame = reader.get_data(frame_id)
            frame = Image.fromarray(frame)
            frame = self.frame_processor(frame)
            frames.append(frame)
        reader.close()
        return frames

class AOSSLoadVideo(DataProcessingOperator):
    def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1, frame_processor=lambda x: x):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.frame_processor = frame_processor

    def get_num_frames(self, frames_list):
        num_frames = self.num_frames
        if len(frames_list) < num_frames:
            num_frames = len(frames_list)
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames

    def __call__(self, data_bytes: bytes):
        """
        data_bytes: 从AOSS读取的视频二进制流
        """
        with BytesIO(data_bytes) as f:
            reader = iio.imiter(f, plugin="pyav")  # 使用 pyav 读取视频帧
            frames = [Image.fromarray(frame) for frame in reader]

        num_frames = self.get_num_frames(frames)
        frames = frames[:num_frames]  # 取前 num_frames 帧

        # 处理每一帧
        frames = [self.frame_processor(f) for f in frames]

        return frames

class SequencialProcess(DataProcessingOperator):
    def __init__(self, operator=lambda x: x):
        self.operator = operator
        
    def __call__(self, data):
        return [self.operator(i) for i in data]



class LoadGIF(DataProcessingOperator):
    def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1, frame_processor=lambda x: x):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        # frame_processor is build in the video loader for high efficiency.
        self.frame_processor = frame_processor
        
    def get_num_frames(self, path):
        num_frames = self.num_frames
        images = iio.imread(path, mode="RGB")
        if len(images) < num_frames:
            num_frames = len(images)
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames
        
    def __call__(self, data: str):
        num_frames = self.get_num_frames(data)
        frames = []
        images = iio.imread(data, mode="RGB")
        for img in images:
            frame = Image.fromarray(img)
            frame = self.frame_processor(frame)
            frames.append(frame)
            if len(frames) >= num_frames:
                break
        return frames
    


class RouteByExtensionName(DataProcessingOperator):
    def __init__(self, operator_map):
        self.operator_map = operator_map
        
    def __call__(self, data: str):
        file_ext_name = data.split(".")[-1].lower()
        for ext_names, operator in self.operator_map:
            if ext_names is None or file_ext_name in ext_names:
                return operator(data)
        raise ValueError(f"Unsupported file: {data}")



class RouteByType(DataProcessingOperator):
    def __init__(self, operator_map):
        self.operator_map = operator_map
        
    def __call__(self, data):
        for dtype, operator in self.operator_map:
            if dtype is None or isinstance(data, dtype):
                return operator(data)
        raise ValueError(f"Unsupported data: {data}")



class LoadTorchPickle(DataProcessingOperator):
    def __init__(self, map_location="cpu"):
        self.map_location = map_location
        
    def __call__(self, data):
        return torch.load(data, map_location=self.map_location, weights_only=False)



class ToAbsolutePath(DataProcessingOperator):
    def __init__(self, base_path=""):
        self.base_path = base_path
        
    def __call__(self, data):
        return os.path.join(self.base_path, data)
    

def parse_s3_infos_from_string(string):
    # example: s3://ACCESS_KEY:SECRET_KEY@example.invalid/dataset
    if string.startswith('s3://'):
        string = string[5:]
    ak, sk = string.split('@')[0].split(':')
    bucket = string.split('@')[1].split('.')[0]
    endpoint = 'http://' + string.split('@')[1][len(bucket)+1:].split('/')[0]
    filepath = '/'.join(string.split('@')[1][len(bucket)+1:].split('/')[1:])
    return ak, sk, bucket, endpoint, filepath

# s3://ACCESS_KEY:SECRET_KEY@example.invalid/dataset
def parse_cos_infos_from_string(string):
    if string.startswith('cos://'):
        string = string[6:]
    creds, rest = string.split('@', 1)
    secret_id, secret_key = creds.split(':', 1)
    region, rest = rest.split('/', 1)
    if '/' in rest:
        bucket, filepath = rest.split('/', 1)
    else:
        bucket, filepath = rest, ""
    return secret_id, secret_key, region, bucket, filepath


def join_cos_key(base_path, filename):
    base_path = str(base_path).strip('/')
    filename = str(filename).strip()
    if filename.startswith('cos://'):
        _, _, _, _, filename = parse_cos_infos_from_string(filename)
    filename = filename.lstrip('/')

    if not base_path:
        return filename
    if not filename:
        return base_path

    # If CSV already stores a bucket-relative key like OpenVid/OpenVid_part1/xxx.mp4,
    # do not prefix base_path again.
    if filename == base_path or filename.startswith(base_path + '/'):
        return filename

    return f"{base_path}/{filename}"


def resolve_local_data_path(base_path, filename):
    base_path = str(base_path)
    if base_path.startswith("file://"):
        base_path = base_path[7:]
    filename = str(filename).strip()
    if filename.startswith("file://"):
        filename = filename[7:]
    filename = filename.lstrip("/")

    if os.path.isabs(filename):
        return filename

    filename = filename.replace("\\", "/")
    base_path = os.path.normpath(base_path)
    candidates = [os.path.join(base_path, filename)]

    first_part, _, rest = filename.partition("/")
    base_name = os.path.basename(base_path)
    if rest and first_part == base_name:
        candidates.insert(0, os.path.join(base_path, rest))
        candidates.append(os.path.join(os.path.dirname(base_path), filename))

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]

# @lru_cache
def get_s3_client(ak, sk, endpoint):
    config = Config(
        connect_timeout=max(1, int(float(os.environ.get("DIFFSYNTH_S3_CONNECT_TIMEOUT", "10")))),
        read_timeout=max(1, int(float(os.environ.get("DIFFSYNTH_S3_READ_TIMEOUT", "120")))),
        retries={
            "max_attempts": max(1, int(os.environ.get("DIFFSYNTH_S3_MAX_ATTEMPTS", "10"))),
            "mode": os.environ.get("DIFFSYNTH_S3_RETRY_MODE", "standard"),
        },
        max_pool_connections=max(10, int(os.environ.get("DIFFSYNTH_S3_MAX_POOL_CONNECTIONS", "64"))),
    )
    s3_client = boto3.client(
        's3',
        endpoint_url=endpoint,
        aws_access_key_id=ak,
        aws_secret_access_key=sk,
        config=config
    )
    return s3_client

def get_cos_client(secret_id, secret_key, region, token=None, scheme="https", domain=None):
    try:
        from qcloud_cos import CosConfig, CosS3Client
    except Exception as exc:
        raise RuntimeError("qcloud_cos is required for cos:// paths; install cos-python-sdk-v5") from exc
    config_kwargs = {
        "Region": region,
        "SecretId": secret_id,
        "SecretKey": secret_key,
        "Scheme": scheme,
    }
    if token:
        config_kwargs["Token"] = token
    if domain:
        config_kwargs["Domain"] = domain
    timeout = os.environ.get("DIFFSYNTH_COS_TIMEOUT")
    if timeout:
        config_kwargs["Timeout"] = max(1, int(float(timeout)))
    try:
        config = CosConfig(**config_kwargs)
    except TypeError:
        # Older cos-python-sdk-v5 builds may not expose Timeout in CosConfig.
        config_kwargs.pop("Timeout", None)
        config = CosConfig(**config_kwargs)
    return CosS3Client(config)

def _read_cos_response_bytes(response):
    body = response.get("Body")
    if body is None:
        return b""
    if hasattr(body, "get_raw_stream"):
        stream = body.get_raw_stream()
        try:
            return stream.read()
        finally:
            if hasattr(stream, "close"):
                stream.close()
    if hasattr(body, "read"):
        try:
            return body.read()
        finally:
            if hasattr(body, "close"):
                body.close()
    raise RuntimeError("Unsupported COS response body type")


class MultiWebDataset:
    def __init__(self):
        self.urls_list = []
        self.prefix_list = []
        self.datasets = []
        self.mix_probs = []
        self.lengths = []
        self.mix_dataset = None

    def append(self, urls, prefix, mix_prob=1, length=1e12):
        self.urls_list.append(urls)
        self.prefix_list.append(prefix)
        self.mix_probs.append(mix_prob)
        self.lengths.append(length)
    
    def score_filter(sample, min_score=5, score_key="aesthetic_score_laion_v2"):
        """基础分数过滤函数"""

        byte_string = sample['json']
        json_string = byte_string.decode('utf-8')

        sample = json.loads(json_string)
        if score_key in sample:
            try:
                score = float(sample[score_key])

                return score >= min_score
            except (ValueError, TypeError):
                return False
        return False

    def _load_s3_url(self, client, url, *_args, **_kwargs):
        bucket = url.split('s3://')[1].split('/')[0]
        key = url.split(bucket + '/')[1]

        try:
            response = client.get_object(Bucket=bucket, Key=key)
            # Read all bytes into memory
            body_bytes = response["Body"].read()
            # Explicitly close the streaming body and release the connection
            response["Body"].close()
            del response
            # Return a BytesIO object that behaves like a file
            return io.BytesIO(body_bytes)
            # return response["Body"]
        except ClientError as e:
            raise RuntimeError(f"Failed to load object from {url}: {e}")

    def _load_cos_url(self, client, url, *_args, **_kwargs):
        bucket = url.split('cos://')[1].split('/')[0]
        key = url.split(bucket + '/')[1]

        try:
            response = client.get_object(Bucket=bucket, Key=key)
            body_bytes = _read_cos_response_bytes(response)
            return io.BytesIO(body_bytes)
        except Exception as e:
            raise RuntimeError(f"Failed to load object from {url}: {e}")

    def prepare(self):
        if len(self.datasets) == len(self.urls_list):
            # already prepared
            return
        print(f'Mixing {len(self.urls_list)} datasets:')
        # init s3 client
        for idx, (urls, prefix) in enumerate(zip(self.urls_list, self.prefix_list)):
            if prefix.startswith('cos://'):
                secret_id, secret_key, region, bucket, path = parse_cos_infos_from_string(prefix)
                cos_client = get_cos_client(secret_id, secret_key, region)
                wds.gopen_schemes[f"cos"] = partial(self._load_cos_url, cos_client)
                scheme = "cos"
            else:
                ak, sk, bucket, endpoint, path = parse_s3_infos_from_string(prefix)
                s3_client = get_s3_client(ak, sk, endpoint)
                wds.gopen_schemes[f"s3"] = partial(self._load_s3_url, s3_client)
                scheme = "s3"

            # info = torch.utils.data.get_worker_info()
            # worker_id = info.id if info is not None else 0
            # num_workers = info.num_workers if info is not None else 1
            # rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            # world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
            # print(f'[Rank {rank}/{world_size}][dataloader worker {worker_id}/{num_workers}] initialize dataset...')

            dataset = wds.DataPipeline(
                wds.SimpleShardList([f"{scheme}://" + url for url in urls]),          # list of shards
                wds.split_by_node,
                wds.split_by_worker,
                wds.shuffle(1000),
                wds.tarfile_to_samples(),
                # wds.tarfile_to_samples(handler=wds.warn_and_continue),
                #wds.select(functools.partial(score_filter, min_score=5, score_key="aesthetic_score_laion_v2")),
                # wds.decode("pil"),                  # decode images into PIL.Image
                wds.to_tuple("jpg", "json"),         # return (image, caption)
            )
            self.datasets.append(dataset)
    
    def __iter__(self):
        mix_dataset = wds.RandomMix(
            self.datasets,
            self.mix_probs
        )
        data_iter = iter(mix_dataset)
        while True:
            try:
                image, label = next(data_iter)
                label = json.loads(label)
                if 'TEXT' in label:
                    caption = label['TEXT']
                elif 'caption' in label:
                    caption = label['caption']
                else:
                    continue
                yield image, caption
            # except wds.autodecode.DecodingError as e:
            #     print('error loading data, skip')
            except StopIteration:
                raise StopIteration

    def __len__(self):
        return int(sum(self.lengths))


class UnifiedDataset(torch.utils.data.Dataset):
    """
    使用示例
        base_path = "cos://<SecretId>:<SecretKey>@ap-beijing/<bucket-name-appid>/datasets/OpenVid"
        metadata_path = "cos://<SecretId>:<SecretKey>@ap-beijing/<bucket-name-appid>/datasets/OpenVid/meta.csv"

    """
    def __init__(
        self,
        base_path=None,
        metadata_path=None,
        repeat=1,
        data_file_keys=tuple(),
        main_data_operator=lambda x: x,
        special_operator_map=None,
    ):
        self.base_path = base_path
        self.metadata_path = metadata_path

        self.s3_client = None

        self.repeat = repeat
        self.data_file_keys = data_file_keys
        self.main_data_operator = main_data_operator
        self.cached_data_operator = LoadTorchPickle()
        self.special_operator_map = {} if special_operator_map is None else special_operator_map
        self.data = []
        self.cached_data = []
        self.load_from_cache = metadata_path is None
        self.data_index = []
        self.read_fns = []
        # NOTE: ugly code for webdataset
        self.is_webdata = False
        self.webdata = None
        self.webloader = None

        self.init_dataset(base_path, metadata_path, str(repeat))

    def init_dataset(self, base_path, metadata_path, repeat, num_lines=None):
        metadata_path_list = metadata_path.split(',')
        base_path_list = base_path.split(',')
        repeat_list = [float(r) for r in repeat.split(',')]
        assert len(metadata_path_list) == len(base_path_list)
        for base_path, metadata_path, repeat in zip(base_path_list, metadata_path_list, repeat_list):
            if metadata_path.startswith('s3://'):
                ak, sk, bucket, endpoint, filepath = parse_s3_infos_from_string(metadata_path)
                s3_client = get_s3_client(ak, sk, endpoint)
                # read annotation
                print(f'reading annotation from {filepath}')
                response = s3_client.get_object(Bucket=bucket, Key=filepath)
                data_bytes = response['Body'].read()
            elif metadata_path.startswith('cos://'):
                secret_id, secret_key, region, bucket, filepath = parse_cos_infos_from_string(metadata_path)
                cos_client = get_cos_client(secret_id, secret_key, region)
                print(f'reading annotation from {filepath}')
                response = cos_client.get_object(Bucket=bucket, Key=filepath)
                data_bytes = _read_cos_response_bytes(response)
            else:
                filepath = metadata_path[7:] if metadata_path.startswith("file://") else metadata_path
                print(f'reading annotation from local file: {filepath}')
                with open(filepath, 'rb') as f:
                    data_bytes = f.read()
            if filepath.endswith(".csv"):
                with BytesIO(data_bytes) as f:
                    reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"))
                    data = [{'video': row['video'], 'prompt': row['caption']} for row in islice(reader, num_lines)]
            elif filepath.endswith(".json"):
                data_temp = json.loads(data_bytes.decode("utf-8"))
                # NOTE: hardcode
                if 'VidGen' in metadata_path:
                    data = [
                        {'video': f'{item["vid"]}.mp4', 'prompt': item['caption']}
                        for item in islice(data_temp, num_lines)  # num_lines=None means read all lines
                    ]
                elif 'laion' in filepath.lower() or 'coyo' in filepath.lower():
                    urls = data_temp
                    self.is_webdata = True
                    assert len(self.data) == 0, "cannot mix normal dataset and webdataset"
                    if self.webdata is None:
                        self.webdata = MultiWebDataset()
                    self.webdata.append(urls, base_path, mix_prob=repeat, length=_DATASET_LENGTH_MAP.get(filepath, 1_000_000))
                    continue
                else:
                    data = [
                        {'video': f'{item["image"]}.jpeg', 'prompt': item['caption']}
                        for item in islice(data_temp, num_lines)  # num_lines=None means read all lines
                    ]
                if num_lines is not None:
                    num_lines -= len(data)
            else:
                raise RuntimeError('Not implemented')  # TODO: read from local disk
            
            assert self.webdata is None, "cannot mix normal dataset and webdataset"
            self.data_index.append(len(self.data))
            self.data.extend(data * int(repeat))

    def init_s3_client(self):
        base_path_list = self.base_path.split(',')
        for base_path in base_path_list:
            if base_path.startswith('s3://'):
                ak, sk, bucket, endpoint, path = parse_s3_infos_from_string(base_path)
                s3_client = get_s3_client(ak, sk, endpoint)
                self.read_fns.append(
                    partial(UnifiedDataset.read_bytes_from_s3, 
                        s3_client=s3_client,
                        bucket=bucket,
                        base_path=path
                    )
                )
            elif base_path.startswith('cos://'):
                secret_id, secret_key, region, bucket, path = parse_cos_infos_from_string(base_path)
                cos_client = get_cos_client(secret_id, secret_key, region)
                filename_index_state = {"ready": False, "map": {}}
                self.read_fns.append(
                    partial(UnifiedDataset.read_bytes_from_cos,
                        cos_client=cos_client,
                        bucket=bucket,
                        base_path=path,
                        filename_index_state=filename_index_state
                    )
                )
            else:
                local_base_path = base_path[7:] if base_path.startswith("file://") else base_path
                self.read_fns.append(
                    partial(UnifiedDataset.read_bytes_from_local, base_path=local_base_path)
                )

    @staticmethod
    def read_bytes_from_s3(
        filename,
        s3_client,
        bucket,
        base_path
    ):
        absolute_path = os.path.join(base_path, filename)
        retries = max(1, _env_int("DIFFSYNTH_S3_READ_RETRIES", "3"))
        backoff_sec = max(
            0.0, _env_float("DIFFSYNTH_S3_READ_RETRY_BACKOFF_SEC", "0.5")
        )
        max_backoff_sec = max(
            backoff_sec,
            _env_float("DIFFSYNTH_S3_READ_RETRY_MAX_BACKOFF_SEC", "4.0"),
        )
        last_error = None
        for attempt in range(1, retries + 1):
            body = None
            try:
                response = s3_client.get_object(Bucket=bucket, Key=absolute_path)
                body = response.get("Body")
                if body is None:
                    raise RuntimeError("S3 get_object response has no Body")
                return body.read()
            except Exception as e:
                last_error = e
                if attempt < retries and backoff_sec > 0:
                    delay = min(
                        backoff_sec * (2 ** (attempt - 1)), max_backoff_sec
                    )
                    time.sleep(delay * (0.75 + random.random() * 0.5))
            finally:
                if body is not None and hasattr(body, "close"):
                    body.close()

        if (
            os.environ.get("DIFFSYNTH_S3_DEBUG_ERRORS", "1") == "1"
            and last_error is not None
        ):
            print(
                f"S3 read failed after {retries} attempts: key={absolute_path} "
                f"error={type(last_error).__name__}: {last_error}"
            )
        return None

    @staticmethod
    def read_bytes_from_local(filename, base_path):
        path = resolve_local_data_path(base_path, filename)
        try:
            with open(path, "rb") as f:
                return f.read()
        except Exception as e:
            if os.environ.get("DIFFSYNTH_LOCAL_DEBUG_ERRORS", "1") == "1":
                print(f"Local read failed: path={path} error={type(e).__name__}: {e}")
            return None

    @staticmethod
    def read_bytes_from_cos(
        filename,
        cos_client,
        bucket,
        base_path,
        filename_index_state=None
    ):
        absolute_path = join_cos_key(base_path, filename)
        retries = max(1, int(os.environ.get("DIFFSYNTH_COS_READ_RETRIES", "3")))
        last_error = None
        for attempt in range(1, retries + 1):
            try:
                _throttle_cos_read()
                response = cos_client.get_object(Bucket=bucket, Key=absolute_path)
                data_bytes = _read_cos_response_bytes(response)
                return data_bytes
            except Exception as e:
                last_error = e
                if attempt < retries:
                    time.sleep(min(2 ** (attempt - 1), 8))
        if os.environ.get("DIFFSYNTH_COS_DEBUG_ERRORS", "1") == "1" and last_error is not None:
            print(f"COS read failed after {retries} attempts: key={absolute_path} error={type(last_error).__name__}: {last_error}")

        # CSV may only contain filename while objects are under subdirs like OpenVid/OpenVid_part{i}/xxx.mp4.
        if '/' in str(filename):
            return None

        if filename_index_state is None:
            return None

        if not filename_index_state["ready"]:
            filename_index_state["map"] = UnifiedDataset.build_cos_filename_index(cos_client, bucket, base_path)
            filename_index_state["ready"] = True

        indexed_path = filename_index_state["map"].get(os.path.basename(str(filename)))
        if indexed_path is None:
            return None

        last_error = None
        for attempt in range(1, retries + 1):
            try:
                _throttle_cos_read()
                response = cos_client.get_object(Bucket=bucket, Key=indexed_path)
                data_bytes = _read_cos_response_bytes(response)
                return data_bytes
            except Exception as e:
                last_error = e
                if attempt < retries:
                    time.sleep(min(2 ** (attempt - 1), 8))
        if os.environ.get("DIFFSYNTH_COS_DEBUG_ERRORS", "1") == "1" and last_error is not None:
            print(f"COS indexed read failed after {retries} attempts: key={indexed_path} error={type(last_error).__name__}: {last_error}")
        return None

    @staticmethod
    def build_cos_filename_index(cos_client, bucket, base_path):
        filename_to_key = {}
        prefix = str(base_path).strip('/')
        if prefix and not prefix.endswith('/'):
            prefix = prefix + '/'
        marker = ''
        while True:
            response = cos_client.list_objects(
                Bucket=bucket,
                Prefix=prefix,
                Marker=marker,
                MaxKeys=1000
            )
            contents = response.get('Contents', [])
            for content in contents:
                key = content.get('Key', '')
                filename = os.path.basename(key)
                if not filename:
                    continue
                ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
                if ext not in {"mp4", "avi", "mov", "wmv", "mkv", "flv", "webm", "gif", "jpg", "jpeg", "png", "webp"}:
                    continue
                if filename not in filename_to_key:
                    filename_to_key[filename] = key

            is_truncated = str(response.get('IsTruncated', '')).lower() == 'true'
            if not is_truncated:
                break
            marker = response.get('NextMarker')
            if not marker and contents:
                marker = contents[-1].get('Key', '')
            if not marker:
                break

        print(f"COS filename index built under '{prefix}': {len(filename_to_key)} files")
        return filename_to_key

    @staticmethod
    def default_image_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
    ):
        return RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor)),
            (list, SequencialProcess(ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor))),
        ])
    
    @staticmethod
    def default_video_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        num_frames=81, time_division_factor=4, time_division_remainder=1,
    ):
        return RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> RouteByExtensionName(operator_map=[
                (("jpg", "jpeg", "png", "webp"), LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor) >> ToList()),
                (("gif",), LoadGIF(num_frames, time_division_factor, time_division_remainder) >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor)),
                (("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"), LoadVideo(
                    num_frames, time_division_factor, time_division_remainder,
                    frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor),
                )),
            ])),
        ])
    
    @staticmethod
    def AOSS_video_operator(
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        num_frames=81, time_division_factor=4, time_division_remainder=1,
    ):
        return RouteByType(operator_map=[
            (bytes, AOSSLoadVideo(
                num_frames, time_division_factor, time_division_remainder,
                frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor),
            )),
            (BytesIO, LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor) >> ToList())
        ])
        
    def search_for_cached_data_files(self, path):
        for file_name in os.listdir(path):
            subpath = os.path.join(path, file_name)
            if os.path.isdir(subpath):
                self.search_for_cached_data_files(subpath)
            elif subpath.endswith(".pth"):
                self.cached_data.append(subpath)
    
    def load_metadata(self, metadata_path):
        if metadata_path is None:
            print("No metadata_path. Searching for cached data files.")
            self.search_for_cached_data_files(self.base_path)
            print(f"{len(self.cached_data)} cached data files found.")
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        elif metadata_path.endswith(".jsonl"):
            metadata = []
            with open(metadata_path, 'r') as f:
                for line in f:
                    metadata.append(json.loads(line.strip()))
            self.data = metadata
        else:
            metadata = pandas.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]

    def __getitem__(self, data_id):
        max_attempts = max(1, _env_int("DIFFSYNTH_DATA_MAX_RETRIES", "20"))
        reader_reset_interval = max(
            0, _env_int("DIFFSYNTH_DATA_READER_RESET_INTERVAL", "5")
        )
        retry_backoff_sec = max(
            0.0, _env_float("DIFFSYNTH_DATA_RETRY_BACKOFF_SEC", "0.1")
        )
        retry_max_backoff_sec = max(
            retry_backoff_sec,
            _env_float("DIFFSYNTH_DATA_RETRY_MAX_BACKOFF_SEC", "2.0"),
        )
        current_id = int(data_id)
        last_error = None
        last_failed_id = current_id
        recent_failed_ids = []
        for attempt in range(1, max_attempts + 1):
            try:
                if self.is_webdata:
                    return self._getitem_webdata()
                return self._getitem_normal(current_id)
            except Exception as e:
                last_error = e
                last_failed_id = current_id
                recent_failed_ids.append(current_id)
                recent_failed_ids = recent_failed_ids[-5:]

                # A long-lived DataLoader worker may hold a stale object-store
                # client. Recreate readers after repeated storage read failures.
                is_read_error = str(e).startswith("Error reading data index ")
                if (
                    is_read_error
                    and reader_reset_interval > 0
                    and attempt % reader_reset_interval == 0
                ):
                    self.read_fns.clear()

                if attempt < max_attempts:
                    if retry_backoff_sec > 0:
                        delay = min(
                            retry_backoff_sec * attempt, retry_max_backoff_sec
                        )
                        time.sleep(delay * (0.75 + random.random() * 0.5))
                    current_id = random.randint(0, len(self) - 1)
        raise RuntimeError(
            f"Failed to load data after {max_attempts} attempts; "
            f"last_failed_data_id={last_failed_id}; "
            f"recent_failed_data_ids={recent_failed_ids}; "
            f"last_error={type(last_error).__name__}: {last_error}"
        )
    
    def _getitem_webdata(self):
        self.webdata.prepare()
        if self.webloader is None:
            self.webloader = iter(self.webdata)
        try:
            image_bytes, caption = next(self.webloader)
        except StopIteration:
            self.webloader = iter(self.webdata)
            return self._getitem_webdata()
        
        data = {}
        data['video'] = self.main_data_operator(image_bytes)
        data['prompt'] = caption
        return data

    def _getitem_normal(self, data_id):
        if len(self.read_fns) == 0:
            self.init_s3_client()
        
        # find dataset index
        dataset_idx = 0
        for i in range(len(self.data_index)):
            idx = len(self.data_index) - i - 1
            if data_id >= self.data_index[idx]:
                dataset_idx = idx
                break

        data = self.data[data_id].copy()
        for key in self.data_file_keys:
            if key in data:
                if key in self.special_operator_map:
                    data[key] = self.special_operator_map[key]
                elif key in self.data_file_keys:
                    
                    # print("data[key] in dataloader:", data[key])
                    
                    filename = data[key]
                    temp = self.read_fns[dataset_idx](filename)
                    if temp is None:
                        raise RuntimeError(f"Error reading data index {data_id}, file={filename}")
                    if filename.split('.')[-1] in ("jpg", "jpeg", "png", "webp"):
                        with BytesIO(temp) as f:
                            data[key] = self.main_data_operator(f)
                    else:
                        data[key] = self.main_data_operator(temp)
                    
                    # print("data[key] in dataloader after main_data_operator:", data[key])
                    
        return data

    def __len__(self):
        if self.is_webdata:
            return len(self.webdata)
        if self.load_from_cache:
            return len(self.cached_data)
        else:
            return len(self.data)
        
    def check_data_equal(self, data1, data2):
        # Debug only
        if len(data1) != len(data2):
            return False
        for k in data1:
            if data1[k] != data2[k]:
                return False
        return True

def collate_extract_before_multi_sample_collate_fn(data):
    assert len(data) == 1
    res = collate_multi_sample_collate_fn(data[0])
    # print(res)
    return res

def collate_multi_sample_collate_fn(data):
    res = {}
    keys = list(data[0].keys())
    for key in keys:
        res[key] = []
    for item in data:
        for key in keys:
            res[key].append(item.get(key, None))
    return res

def multi_sample_collate_fn(data):
    res = {}
    keys = list(data[0].keys())
    for key in keys:
        res[key] = []
    for item in data:
        for key in keys:
            res[key].append(item.get(key, None))
    return res


