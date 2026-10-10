"""QWorld ``common`` dataset adapter for DiffSynth video training.

The source configuration remains authoritative for annotation paths, caption
selection, filters, and the video loading pipeline.  This adapter only narrows
its aspect-ratio buckets to one fixed DiffSynth training shape and unwraps the
QWorld batch-of-one result into the sample dictionary expected by DiffSynth.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys
import types

import numpy as np
import torch


DEFAULT_CONFIG_KEY = "video_acpect_ratio_bucket_datasets_common"
DEFAULT_DATASET_KEY = "video_kairos_common_short"
DEFAULT_PIPELINE_KEY = "video_pipeline_short"
DEFAULT_DURATION = "5s"
QWORLD_DATASET_RELATIVE_PATH = Path(
    "qworld/datasets/qworld_visual_caption_dtaset_v4.py"
)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "1" if default else "0").strip().lower()
    if raw not in {"0", "1", "false", "true", "no", "yes", "off", "on"}:
        raise ValueError(f"{name} must be a boolean flag, got {raw!r}")
    return raw in {"1", "true", "yes", "on"}


def _positive_int(value, name: str) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    raw = os.environ.get(name, str(default)).strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc
    if not minimum <= value <= maximum:
        raise ValueError(
            f"{name} must be in [{minimum}, {maximum}], got {value}"
        )
    return value


def _detail_filter_config_from_env() -> dict:
    ratio = _env_float("DIFFSYNTH_QWORLD_DETAIL_RATIO", 0.0, 0.0, 1.0)
    shot_sizes = tuple(
        value.strip()
        for value in os.environ.get(
            "DIFFSYNTH_QWORLD_DETAIL_SHOT_SIZES", "close_up,medium_shot"
        ).split(",")
        if value.strip()
    )
    max_persons = _positive_int(
        os.environ.get("DIFFSYNTH_QWORLD_DETAIL_MAX_PERSONS", "2"),
        "DIFFSYNTH_QWORLD_DETAIL_MAX_PERSONS",
    )
    min_clarity = _env_float(
        "DIFFSYNTH_QWORLD_DETAIL_MIN_CLARITY", 6.0, 0.0, 10.0
    )
    face_ratio = _env_float(
        "DIFFSYNTH_QWORLD_FACE_RATIO", 0.0, 0.0, 1.0
    )
    hand_ratio = _env_float(
        "DIFFSYNTH_QWORLD_HAND_RATIO", 0.0, 0.0, 1.0
    )
    if face_ratio + hand_ratio > 1.0:
        raise ValueError(
            "DIFFSYNTH_QWORLD_FACE_RATIO + DIFFSYNTH_QWORLD_HAND_RATIO must be <= 1"
        )
    if ratio > 0.0 and face_ratio + hand_ratio > 0.0:
        raise ValueError(
            "Legacy DIFFSYNTH_QWORLD_DETAIL_RATIO cannot be combined with "
            "ROI face/hand ratios"
        )
    dynamic_ratio = _env_float(
        "DIFFSYNTH_QWORLD_DETAIL_DYNAMIC_RATIO", 0.30, 0.0, 1.0
    )
    roi_cache_dir = os.environ.get(
        "DIFFSYNTH_QWORLD_ROI_CACHE_DIR", ""
    ).strip()
    if face_ratio + hand_ratio > 0.0 and not roi_cache_dir:
        raise ValueError(
            "ROI replay requires DIFFSYNTH_QWORLD_ROI_CACHE_DIR"
        )
    return {
        "ratio": ratio,
        "face_ratio": face_ratio,
        "hand_ratio": hand_ratio,
        "dynamic_ratio": dynamic_ratio,
        "roi_cache_dir": roi_cache_dir,
        "shot_sizes": shot_sizes,
        "max_persons": max_persons,
        "min_clarity": min_clarity,
    }


def qworld_roi_cache_key(item: dict, *, width: int, height: int, num_frames: int) -> str:
    identity = {
        "video_path": item.get("video_path"),
        "start": item.get("video_sub_start_frame_id"),
        "end": item.get("video_sub_end_frame_id"),
        "width": int(width),
        "height": int(height),
        "num_frames": int(num_frames),
    }
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()[:32]


def qworld_roi_cache_path(cache_dir: str, cache_key: str) -> Path:
    return Path(cache_dir) / "entries" / cache_key[:2] / f"{cache_key}.npz"


def _metadata_item_for_global_index(dataset, global_index: int) -> dict:
    file_id, local_index = dataset._locate(int(global_index))
    index_path = dataset.flat_ann_files[file_id][1]
    items = dataset._load_index_file(index_path)
    if not 0 <= local_index < len(items):
        raise IndexError(
            f"QWorld metadata local index {local_index} outside {index_path}"
        )
    item = items[local_index]
    if not isinstance(item, dict):
        raise TypeError(f"QWorld metadata item must be a dict: {index_path}")
    return item


def _is_human_detail_item(item: dict, config: dict) -> bool:
    tags = item.get("tags") or {}
    if tags.get("primary_category") != "human":
        return False
    sub_tags = tags.get("sub_tags") or {}
    if sub_tags.get("face_blur") is not False:
        return False
    camera = tags.get("camera_metadata") or {}
    if config["shot_sizes"] and camera.get("shot_size") not in config["shot_sizes"]:
        return False
    persons = item.get("score_maxPersonNum")
    if not isinstance(persons, (int, float)) or not 1 <= persons <= config["max_persons"]:
        return False
    clarity = ((item.get("scores") or {}).get("vlm_scores") or {}).get(
        "clarity_score"
    )
    return isinstance(clarity, (int, float)) and clarity >= config["min_clarity"]


def _build_or_load_detail_indices(dataset, cache_dir: str, config: dict) -> list[int]:
    """Build a compact global-index list from QWorld's cached metadata only."""

    flat_files = getattr(dataset, "flat_ann_files", None)
    offsets = getattr(dataset, "offsets", None)
    if not isinstance(flat_files, list) or not isinstance(offsets, list):
        raise TypeError(
            "QWorld detail sampling requires ImageAspectRaitioBucketDataset_V4 "
            "with flat_ann_files and offsets"
        )
    identity = {
        "files": [(str(row[1]), int(row[2])) for row in flat_files],
        "config": {key: value for key, value in config.items() if key != "ratio"},
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()[:20]
    index_dir = Path(cache_dir) / "detail_indices"
    index_dir.mkdir(parents=True, exist_ok=True)
    index_path = index_dir / f"human_face_hand_{digest}.json"
    lock_path = index_path.with_suffix(".lock")

    with open(lock_path, "a+") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        if index_path.is_file():
            with open(index_path, encoding="utf-8") as handle:
                detail_indices = json.load(handle)
        else:
            detail_indices = []
            for file_id, row in enumerate(flat_files):
                index_file = row[1]
                with open(index_file, encoding="utf-8") as handle:
                    items = json.load(handle)
                base_offset = int(offsets[file_id])
                detail_indices.extend(
                    base_offset + local_index
                    for local_index, item in enumerate(items)
                    if _is_human_detail_item(item, config)
                )
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            temporary_path = index_path.with_suffix(f".tmp.{os.getpid()}.{rank}")
            with open(temporary_path, "w", encoding="utf-8") as handle:
                json.dump(detail_indices, handle, separators=(",", ":"))
            os.replace(temporary_path, index_path)
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    if not detail_indices:
        raise RuntimeError(
            "QWorld human-detail filter selected zero samples; loosen shot size, "
            "clarity, or person-count constraints"
        )
    return [int(index) for index in detail_indices]


def _is_dynamic_detail_item(item: dict) -> bool:
    camera = ((item.get("tags") or {}).get("camera_metadata") or {})
    return str(camera.get("visual_dynamics", "")).strip().lower() in {
        "medium",
        "high",
    }


def _load_roi_manifest(config: dict) -> tuple[Path, dict]:
    manifest_path = Path(config["roi_cache_dir"]).expanduser().resolve() / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "ROI replay manifest is missing. Run "
            "scripts/build_qworld_pose_roi_cache_a800.sh first: "
            f"{manifest_path}"
        )
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != 1 or not isinstance(
        manifest.get("entries"), dict
    ):
        raise ValueError(f"Unsupported QWorld ROI manifest: {manifest_path}")
    return manifest_path, manifest["entries"]


def _build_or_load_roi_indices(
    dataset,
    cache_dir: str,
    config: dict,
    *,
    width: int,
    height: int,
    num_frames: int,
) -> dict[str, list[int]]:
    flat_files = getattr(dataset, "flat_ann_files", None)
    offsets = getattr(dataset, "offsets", None)
    if not isinstance(flat_files, list) or not isinstance(offsets, list):
        raise TypeError("QWorld ROI replay requires flat_ann_files and offsets")
    manifest_path, manifest_entries = _load_roi_manifest(config)
    manifest_stat = manifest_path.stat()
    identity = {
        "files": [(str(row[1]), int(row[2])) for row in flat_files],
        "filter": {
            "shot_sizes": config["shot_sizes"],
            "max_persons": config["max_persons"],
            "min_clarity": config["min_clarity"],
            "width": width,
            "height": height,
            "num_frames": num_frames,
        },
        "manifest": [
            str(manifest_path),
            manifest_stat.st_size,
            manifest_stat.st_mtime_ns,
        ],
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()[:20]
    index_dir = Path(cache_dir) / "detail_indices"
    index_dir.mkdir(parents=True, exist_ok=True)
    index_path = index_dir / f"pose_roi_{digest}.json"
    lock_path = index_path.with_suffix(".lock")

    with open(lock_path, "a+") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        if index_path.is_file():
            with index_path.open(encoding="utf-8") as handle:
                result = json.load(handle)
        else:
            result = {
                "face": [],
                "face_dynamic": [],
                "hand": [],
                "hand_dynamic": [],
            }
            for file_id, row in enumerate(flat_files):
                with open(row[1], encoding="utf-8") as handle:
                    items = json.load(handle)
                base_offset = int(offsets[file_id])
                for local_index, item in enumerate(items):
                    if not _is_human_detail_item(item, config):
                        continue
                    cache_key = qworld_roi_cache_key(
                        item, width=width, height=height, num_frames=num_frames
                    )
                    entry = manifest_entries.get(cache_key)
                    if not isinstance(entry, dict):
                        continue
                    global_index = base_offset + local_index
                    dynamic = _is_dynamic_detail_item(item)
                    if float(entry.get("face_coverage", 0.0)) > 0.0:
                        result["face"].append(global_index)
                        if dynamic:
                            result["face_dynamic"].append(global_index)
                    if float(entry.get("hand_coverage", 0.0)) > 0.0:
                        result["hand"].append(global_index)
                        if dynamic:
                            result["hand_dynamic"].append(global_index)
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            temporary_path = index_path.with_suffix(f".tmp.{os.getpid()}.{rank}")
            with temporary_path.open("w", encoding="utf-8") as handle:
                json.dump(result, handle, separators=(",", ":"))
            os.replace(temporary_path, index_path)
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    for kind in ("face", "hand"):
        if config[f"{kind}_ratio"] > 0.0 and not result[kind]:
            raise RuntimeError(
                f"ROI manifest contains no usable {kind} samples after metadata filtering"
            )
    return {
        key: [int(index) for index in values]
        for key, values in result.items()
    }


def _load_reference_namespace(config_path: str) -> dict:
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"QWorld reference config does not exist: {path}")
    namespace = runpy.run_path(str(path))
    for key in (DEFAULT_CONFIG_KEY, DEFAULT_DATASET_KEY, DEFAULT_PIPELINE_KEY):
        if key not in namespace:
            raise KeyError(f"QWorld reference config is missing {key!r}: {path}")
    return namespace


def derive_qworld_common_config(
    config_path: str,
    cache_dir: str,
    *,
    width: int,
    height: int,
    num_frames: int,
    seed: int,
    use_cache: bool,
    duration: str = DEFAULT_DURATION,
) -> dict:
    """Derive the fixed-resolution common wrapper without touching TOS."""

    width = _positive_int(width, "width")
    height = _positive_int(height, "height")
    num_frames = _positive_int(num_frames, "num_frames")
    if (num_frames - 1) % 4 != 0:
        raise ValueError(f"num_frames must satisfy 4n+1, got {num_frames}")

    cache_path = Path(cache_dir).expanduser().resolve()
    namespace = _load_reference_namespace(config_path)
    common_dataset = copy.deepcopy(namespace[DEFAULT_DATASET_KEY])
    common_wrapper = copy.deepcopy(namespace[DEFAULT_CONFIG_KEY])

    if common_dataset.get("type") != "QWorldVisualCaptionDataset_V4":
        raise ValueError(
            f"{DEFAULT_DATASET_KEY} must use QWorldVisualCaptionDataset_V4, "
            f"got {common_dataset.get('type')!r}"
        )
    if common_wrapper.get("type") != "ImageAspectRaitioBucketDataset_V4":
        raise ValueError(
            f"{DEFAULT_CONFIG_KEY} must use ImageAspectRaitioBucketDataset_V4, "
            f"got {common_wrapper.get('type')!r}"
        )

    duration = str(duration).strip().lower()
    if duration != DEFAULT_DURATION:
        raise ValueError(
            f"This training adapter is fixed to {DEFAULT_DURATION}, got {duration!r}"
        )
    ann_files = common_dataset.get("ann_file", [])
    data_prefixes = common_dataset.get("data_prefix", [])
    if not isinstance(ann_files, (list, tuple)) or not isinstance(
        data_prefixes, (list, tuple)
    ):
        raise TypeError(
            f"{DEFAULT_DATASET_KEY} ann_file and data_prefix must both be lists"
        )
    if len(ann_files) != len(data_prefixes):
        raise ValueError(
            f"{DEFAULT_DATASET_KEY} ann_file/data_prefix length mismatch: "
            f"{len(ann_files)} != {len(data_prefixes)}"
        )
    selected_sources = [
        (ann_file, data_prefix)
        for ann_file, data_prefix in zip(ann_files, data_prefixes)
        if str(ann_file).rstrip("/").endswith(f"/sub/{duration}/index.json")
    ]
    if len(selected_sources) != 1:
        raise RuntimeError(
            f"Expected exactly one {duration} common TOS index, found "
            f"{len(selected_sources)} in {ann_files}"
        )
    common_dataset["ann_file"] = [selected_sources[0][0]]
    common_dataset["data_prefix"] = [selected_sources[0][1]]

    source_bucket_key = f"visual_1280_704_{num_frames}"
    source_bucket_cfg = common_dataset.get("bucket_cfg", {}).get(source_bucket_key)
    if source_bucket_cfg is None:
        candidates = [
            (key, value)
            for key, value in common_dataset.get("bucket_cfg", {}).items()
            if key.rsplit("_", 1)[-1] == str(num_frames)
        ]
        if not candidates:
            raise KeyError(
                f"No {num_frames}-frame bucket exists in {DEFAULT_DATASET_KEY}"
            )
        source_bucket_key, source_bucket_cfg = candidates[0]

    target_bucket_key = f"visual_{width}_{height}_{num_frames}"
    common_dataset["bucket_cfg"] = {
        target_bucket_key: copy.deepcopy(source_bucket_cfg)
    }
    # A fixed DiffSynth tensor shape cannot consume QWorld's mixed-resolution
    # duplicate buckets. Keep the reference ratio/filter policy otherwise.
    common_dataset["multi_resolution_match_cfg"] = {"enable": False}
    common_dataset["cache_dir"] = str(cache_path)
    common_dataset["use_cache"] = bool(use_cache)
    common_dataset["seed"] = int(seed)
    common_dataset["pipeline"] = None

    common_wrapper["datasets"] = [common_dataset]
    common_wrapper["repeat_counts"] = [1]
    common_wrapper["batch_size"] = 1
    common_wrapper["pipeline"] = copy.deepcopy(namespace[DEFAULT_PIPELINE_KEY])
    common_wrapper["pipeline_for_each"] = True
    common_wrapper["do_transform"] = True
    common_wrapper["seed"] = int(seed)

    return {
        "dataset_config": common_wrapper,
        "config_path": str(Path(config_path).expanduser().resolve()),
        "cache_dir": str(cache_path),
        "source_bucket": source_bucket_key,
        "target_bucket": target_bucket_key,
        "duration": duration,
        "annotation_files": list(common_dataset.get("ann_file", [])),
        "dataset_name": common_dataset.get("dataset_name"),
        "task_group": common_dataset.get("task_group"),
        "caption_keys_probs": copy.deepcopy(
            common_dataset.get("caption_keys_probs", {})
        ),
        "use_cache": bool(use_cache),
    }


def _install_optional_decord_stub() -> None:
    """Let QWorld register unused datasets when decord is not installed.

    The selected ``LoadVideo_IIO3`` pipeline uses imageio/pyav. QWorld's package
    initializer also imports unrelated Cosmos datasets that import decord, so a
    fail-fast stub is sufficient and cannot silently decode training samples.
    """

    if "decord" in sys.modules or importlib.util.find_spec("decord") is not None:
        return

    module = types.ModuleType("decord")
    module.__spec__ = importlib.util.spec_from_loader("decord", loader=None)

    class _UnavailableVideoReader:
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "decord is unavailable; the QWorld common adapter must use "
                "LoadVideo_IIO3 (imageio/pyav)"
            )

    module.VideoReader = _UnavailableVideoReader
    module.cpu = lambda *args, **kwargs: None
    sys.modules["decord"] = module


def _import_qworld_builder(qworld_code_root: str):
    root = Path(qworld_code_root).expanduser().resolve()
    dataset_source = root / QWORLD_DATASET_RELATIVE_PATH
    if not dataset_source.is_file():
        raise FileNotFoundError(
            "QWorldVisualCaptionDataset_V4 source does not exist: "
            f"{dataset_source}"
        )
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    _install_optional_decord_stub()

    # Importing qworld.datasets performs the registry side effects required by
    # build_dataset, including all transforms in video_pipeline_short.
    import qworld.datasets  # noqa: F401
    import mmengine
    from qworld.commons.file_utils import QWorldFileClient
    from qworld.datasets.builder import build_dataset

    # QWorld's normal tools/train.py entrypoint performs these registrations.
    # DiffSynth enters through its own trainer, so mirror that required setup
    # here before QWorldVisualCaptionDataset_V4 calls mmengine.load(tos://...).
    mmengine.fileio.file_client.FileClient._backends["tos"] = QWorldFileClient
    mmengine.fileio.file_client.FileClient._prefix_to_backends["tos"] = QWorldFileClient
    mmengine.fileio.io.prefix_to_backends["tos"] = QWorldFileClient

    return build_dataset, dataset_source


@contextmanager
def _distributed_cache_build_lock(cache_dir: str):
    """Serialize the first cache build across independent training jobs.

    QWorld already synchronizes ranks inside one distributed job. This outer
    file lock prevents mapped and random jobs submitted together from writing
    the same shared cache files concurrently.
    """

    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    rank = torch.distributed.get_rank() if distributed else 0
    lock_handle = None
    if rank == 0:
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        lock_handle = open(cache_path / ".qworld_common_build.lock", "a+")
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
    if distributed:
        torch.distributed.barrier()
    try:
        yield
    finally:
        if distributed:
            torch.distributed.barrier()
        if lock_handle is not None:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            lock_handle.close()


class QWorldCommonDataset(torch.utils.data.Dataset):
    """Unwrap QWorld's one-item internal batch for DiffSynth's DataLoader."""

    load_from_cache = False

    def __init__(
        self,
        *,
        config_path: str,
        qworld_code_root: str,
        cache_dir: str,
        width: int,
        height: int,
        num_frames: int,
        repeat: int = 1,
        seed: int = 0,
        use_cache: bool = True,
        duration: str = DEFAULT_DURATION,
    ):
        self.repeat = _positive_int(repeat, "repeat")
        derived = derive_qworld_common_config(
            config_path,
            cache_dir,
            width=width,
            height=height,
            num_frames=num_frames,
            seed=seed,
            use_cache=use_cache,
            duration=duration,
        )
        # Cached manifests contain absolute shard paths from the machine that
        # built them. Repair migrated paths before QWorld reads/shuffles them.
        from diffsynth.core.data.qworld_cache_paths import repair_qworld_cache_paths

        repair_qworld_cache_paths(derived["cache_dir"])
        build_dataset, dataset_source = _import_qworld_builder(qworld_code_root)
        with _distributed_cache_build_lock(derived["cache_dir"]):
            self.dataset = build_dataset(derived["dataset_config"])
        self._base_length = len(self.dataset)
        if self._base_length <= 0:
            raise RuntimeError(
                "QWorld common dataset is empty after filtering; check cache and "
                f"bucket {derived['target_bucket']}"
            )
        self.width = int(width)
        self.height = int(height)
        self.num_frames = int(num_frames)
        self.seed = int(seed)
        self.detail_filter_config = _detail_filter_config_from_env()
        self.detail_indices = []
        self.roi_indices = {}
        if self.detail_filter_config["ratio"] > 0.0:
            self.detail_indices = _build_or_load_detail_indices(
                self.dataset, derived["cache_dir"], self.detail_filter_config
            )
        if (
            self.detail_filter_config["face_ratio"]
            + self.detail_filter_config["hand_ratio"]
            > 0.0
        ):
            self.roi_indices = _build_or_load_roi_indices(
                self.dataset,
                derived["cache_dir"],
                self.detail_filter_config,
                width=self.width,
                height=self.height,
                num_frames=self.num_frames,
            )
        self.description = {key: value for key, value in derived.items() if key != "dataset_config"}
        self.description["dataset_source"] = str(dataset_source)
        print(
            "QWorld common dataset ready: "
            f"samples={self._base_length} repeat={self.repeat} "
            f"duration={derived['duration']} "
            f"bucket={derived['target_bucket']} cache={derived['cache_dir']} "
            f"source={dataset_source}",
            flush=True,
        )
        if self.detail_indices:
            print(
                "QWorld human-detail replay ready: "
                f"detail_samples={len(self.detail_indices)} "
                f"detail_ratio={self.detail_filter_config['ratio']:.3f} "
                f"max_persons={self.detail_filter_config['max_persons']} "
                f"min_clarity={self.detail_filter_config['min_clarity']:.1f} "
                f"shot_sizes={','.join(self.detail_filter_config['shot_sizes'])}",
                flush=True,
            )
        if self.roi_indices:
            print(
                "QWorld pose-ROI replay ready: "
                f"face={len(self.roi_indices['face'])} "
                f"hand={len(self.roi_indices['hand'])} "
                f"dynamic_face={len(self.roi_indices['face_dynamic'])} "
                f"dynamic_hand={len(self.roi_indices['hand_dynamic'])} "
                f"ratios=face:{self.detail_filter_config['face_ratio']:.3f},"
                f"hand:{self.detail_filter_config['hand_ratio']:.3f} "
                f"dynamic_target={self.detail_filter_config['dynamic_ratio']:.3f}",
                flush=True,
            )

    def __len__(self) -> int:
        return self._base_length * self.repeat

    def __getitem__(self, index: int) -> dict:
        detail_sample = False
        detail_kind = 0
        selected_index = index % self._base_length
        if self.detail_indices:
            # Keep selection deterministic under distributed samplers and resume.
            selector = (int(index) * 1_103_515_245 + self.seed) & 0xFFFFFFFF
            threshold = int(self.detail_filter_config["ratio"] * 10_000)
            detail_sample = selector % 10_000 < threshold
            if detail_sample:
                detail_position = (
                    int(index) * 2_654_435_761 + self.seed
                ) % len(self.detail_indices)
                selected_index = self.detail_indices[detail_position]

        if self.roi_indices:
            selector = (int(index) * 1_103_515_245 + self.seed) & 0xFFFFFFFF
            selector_unit = (selector % 10_000) / 10_000.0
            face_ratio = self.detail_filter_config["face_ratio"]
            hand_ratio = self.detail_filter_config["hand_ratio"]
            if selector_unit < face_ratio:
                detail_kind = 1
                detail_sample = True
            elif selector_unit < face_ratio + hand_ratio:
                detail_kind = 2
                detail_sample = True

            if detail_kind:
                kind = "face" if detail_kind == 1 else "hand"
                dynamic_selector = (
                    int(index) * 2_246_822_519 + self.seed
                ) & 0xFFFFFFFF
                use_dynamic = (
                    self.roi_indices[f"{kind}_dynamic"]
                    and dynamic_selector % 10_000
                    < int(self.detail_filter_config["dynamic_ratio"] * 10_000)
                )
                pool = self.roi_indices[
                    f"{kind}_dynamic" if use_dynamic else kind
                ]
                selected_index = pool[
                    (int(index) * 3_266_489_917 + self.seed) % len(pool)
                ]

        selected_metadata = None
        if detail_kind:
            selected_metadata = _metadata_item_for_global_index(
                self.dataset, selected_index
            )

        batch = self.dataset[selected_index]
        if not isinstance(batch, list) or len(batch) != 1:
            raise RuntimeError(
                "QWorld common wrapper must return a one-item list, got "
                f"{type(batch).__name__} with length "
                f"{len(batch) if hasattr(batch, '__len__') else 'unknown'}"
            )
        sample = batch[0]
        if not isinstance(sample, dict) or "video" not in sample or "prompt" not in sample:
            raise RuntimeError(
                "QWorld common sample must contain video and prompt, got keys="
                f"{sorted(sample) if isinstance(sample, dict) else type(sample).__name__}"
            )
        video = sample["video"]
        if len(video) != self.num_frames:
            raise RuntimeError(
                f"QWorld returned {len(video)} frames, expected {self.num_frames}"
            )
        if video and video[0].size != (self.width, self.height):
            raise RuntimeError(
                f"QWorld returned frame size {video[0].size}, expected "
                f"{(self.width, self.height)}"
            )
        sample["detail_sample"] = detail_sample
        sample["detail_kind"] = detail_kind
        if self.roi_indices:
            sampled_indices = np.asarray([0], dtype=np.int64)
            max_persons = self.detail_filter_config["max_persons"]
            face_boxes = np.full((1, max_persons, 4), -1.0, dtype=np.float32)
            hand_boxes = np.full((1, max_persons * 2, 4), -1.0, dtype=np.float32)
            if detail_kind and selected_metadata is not None:
                # QWorld may resample inside a corrupt source file. Never attach
                # boxes belonging to a different decoded video in that case.
                if sample.get("filename") == selected_metadata.get("video_path"):
                    cache_key = qworld_roi_cache_key(
                        selected_metadata,
                        width=self.width,
                        height=self.height,
                        num_frames=self.num_frames,
                    )
                    cache_path = qworld_roi_cache_path(
                        self.detail_filter_config["roi_cache_dir"], cache_key
                    )
                    with np.load(cache_path, allow_pickle=False) as roi:
                        sampled_indices = np.asarray(
                            roi["sampled_frame_indices"], dtype=np.int64
                        )
                        face_boxes = np.asarray(roi["face_boxes"], dtype=np.float32)
                        hand_boxes = np.asarray(roi["hand_boxes"], dtype=np.float32)
                else:
                    sample["detail_sample"] = False
                    sample["detail_kind"] = 0
            sample["roi_sampled_frame_indices"] = torch.from_numpy(sampled_indices)
            sample["face_roi_boxes"] = torch.from_numpy(face_boxes)
            sample["hand_roi_boxes"] = torch.from_numpy(hand_boxes)
        return sample


def build_qworld_common_dataset_from_env(*, width, height, num_frames, repeat, seed):
    required = (
        "DIFFSYNTH_QWORLD_COMMON_CONFIG",
        "DIFFSYNTH_QWORLD_CODE_ROOT",
        "DIFFSYNTH_QWORLD_CACHE_DIR",
    )
    missing = [name for name in required if not os.environ.get(name, "").strip()]
    if missing:
        raise RuntimeError(
            "QWorld common dataset environment is incomplete: " + ", ".join(missing)
        )
    return QWorldCommonDataset(
        config_path=os.environ["DIFFSYNTH_QWORLD_COMMON_CONFIG"],
        qworld_code_root=os.environ["DIFFSYNTH_QWORLD_CODE_ROOT"],
        cache_dir=os.environ["DIFFSYNTH_QWORLD_CACHE_DIR"],
        width=width,
        height=height,
        num_frames=num_frames,
        repeat=repeat,
        seed=seed,
        use_cache=_env_flag("DIFFSYNTH_QWORLD_USE_CACHE", True),
        duration=os.environ.get(
            "DIFFSYNTH_QWORLD_COMMON_DURATION", DEFAULT_DURATION
        ),
    )


def _main() -> None:
    parser = argparse.ArgumentParser(description="Validate derived QWorld common config")
    parser.add_argument("--config", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=736)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--duration", default=DEFAULT_DURATION)
    parser.add_argument("--seed", type=int, default=20260731)
    args = parser.parse_args()
    derived = derive_qworld_common_config(
        args.config,
        args.cache_dir,
        width=args.width,
        height=args.height,
        num_frames=args.num_frames,
        seed=args.seed,
        use_cache=True,
        duration=args.duration,
    )
    print(
        f"OK: dataset={derived['dataset_name']} task_group={derived['task_group']} "
        f"duration={derived['duration']} "
        f"bucket={derived['target_bucket']} annotations={len(derived['annotation_files'])} "
        f"cache={derived['cache_dir']}"
    )


if __name__ == "__main__":
    _main()
