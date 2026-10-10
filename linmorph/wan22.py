"""Shared wan22 implementation; command-line entry points live in scripts/."""


import hashlib
import json
from pathlib import Path
import struct
import subprocess
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]


OUT = ROOT / 'output/wan22_5b_quick_eval_20260916'


RUN = ROOT / 'output/train/24gdn/train_24gdn_large_nonmemory_gdn_full_ffn_softmax_lora_r32_from_step5000_teacher_baseline_a800_16gpu_736x1280_f81'


CKPT = RUN / 'model-step-71000.safetensors'


BASE = ROOT / 'model_weight/Wan2.2-TI2V-5B'


BLOCKS = [0,1,2,3,5,6,7,8,10,11,12,13,15,16,17,18,20,21,22,23,25,26,27,28]


PROMPTS = [
 ('animal', 'A fluffy orange cat drinking water from a glass bowl, detailed fur, natural daylight, realistic smooth motion.'),
 ('human', 'A person in a long coat walking through a neon-lit rainy street at night, full body clearly visible, slow camera push-in.'),
 ('vehicle', 'A cinematic tracking shot of a vintage motorcycle riding along a coastal road at sunset.'),
 ('landscape', 'Morning mist flowing through a pine forest in a mountain valley, sunlight breaking through the trees, a slow forward camera move.'),
 ('hands', 'Close-up of a chef slicing a cucumber on a wooden cutting board, both hands visible, realistic controlled movements.'),
 ('interaction', 'A woman picks up a red cup from a table and takes a sip, medium shot, natural lighting.'),
 ('action', 'A man jogging along a beach, full body in frame, waves breaking on the shore, the camera follows smoothly.'),
 ('multiple', 'Two dogs playing and chasing each other across a grassy park, wide shot, sunny daylight.'),
 ('animation', 'A cheerful cartoon robot waves its hand while walking through a colorful futuristic city, smooth animation.'),
 ('composition', 'A red car parked to the left of a blue bicycle on a quiet street, the camera slowly pans from left to right.'),
 ('texture', 'A close-up of ocean waves breaking over dark rocks, detailed foam and water droplets, steady camera.'),
 ('camera', 'A slow orbiting camera shot around a stone statue in a sunlit garden, realistic perspective and stable geometry.'),
]


CONFIG = dict(height=736, width=1280, num_frames=81, num_inference_steps=50,
              cfg_scale=5.0, sigma_shift=5.0, tiled=True, fps=12, dtype='bf16',
              scheduler='DiffSynth FlowMatchScheduler(Wan), Euler update',
              negative_prompt='overexposed, static, blurry, low quality, jpeg artifacts, deformed, extra fingers, bad hands, bad face, messy background, jitter, frame skipping')


MODELS = ['step71000', 'wan22_base']


def header(path):
    with path.open('rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        if n > 64 * 1024 * 1024:
            raise ValueError('Invalid safetensors header')
        raw = f.read(n)
        h = json.loads(raw)
        data_bytes = path.stat().st_size - n - 8
    for k, v in h.items():
        if k != '__metadata__' and v['data_offsets'][1] > data_bytes:
            raise ValueError(f'Truncated tensor: {path}: {k}')
    return h, hashlib.sha256(raw).hexdigest()


def inputs():
    return [dict(id=f'p{i:02d}_seed{seed}', prompt_index=i, category=cat, prompt=prompt, seed=seed)
            for i, (cat, prompt) in enumerate(PROMPTS, 1) for seed in [43, 44]]


def idle_gpu(index):
    text = subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu',
                                    '--format=csv,noheader,nounits'], text=True)
    for line in text.splitlines():
        i,name,mem,util = [v.strip() for v in line.split(',')]
        if int(i) == index:
            if 'A800' not in name or int(mem)>256 or int(util)!=0:
                raise RuntimeError(f'GPU {index} occupied: {mem} MiB, utilization {util}%. No GPU work started.')
            return
    raise ValueError(f'GPU {index} does not exist')


def load_pipeline(args, model):
    from linmorph import inference as core
    core.import_runtime_deps()
    settings = SimpleNamespace(model_dir=str(BASE),device='cuda:0',dtype='bf16',
        gdn_block_indices=','.join(map(str,BLOCKS)),gdn_expand_v=2.,gdn_head_dim=256,
        gdn_hidden_size=0,gdn_num_heads=0)
    pipe = core.build_pipeline(settings, BLOCKS, args.memory_mode, replace_gdn=model=='step71000')
    if model == 'step71000':
        core.load_checkpoint_into_pipe(pipe,str(CKPT),BLOCKS,core.torch.bfloat16)
    core.maybe_force_fp32_time_mlp(pipe)
    pipe.eval()
    return core, pipe


def kwargs(row):
    return {k:v for k,v in CONFIG.items() if k not in ['fps','dtype','scheduler']} | dict(
        prompt=row['prompt'],seed=row['seed'],cfg_merge=False)

