#!/usr/bin/env python3
"""Generate one video with a configured LinMorph model or the original Wan model."""
import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/inference/wan21_14gdn.json')
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--base-model', action='store_true')
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--negative-prompt')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--memory-mode', choices=['auto', 'resident', 'cpu_offload'], default='auto')
    parser.add_argument('--scheduler', choices=['unipc', 'euler'], default='unipc')
    parser.add_argument('--unipc-source', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if bool(args.checkpoint) == args.base_model:
        parser.error('Choose exactly one of --checkpoint and --base-model')
    cfg = json.loads(args.config.read_text())
    plan = dict(cfg, prompt=args.prompt, seed=args.seed, scheduler=args.scheduler,
                base_model=args.base_model, checkpoint=str(args.checkpoint),
                negative_prompt=args.negative_prompt, model_dir=str(args.model_dir))
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if args.dry_run:
        return
    if args.output.exists():
        parser.error('Output already exists; choose a new filename')
    if args.checkpoint and not args.checkpoint.is_file():
        parser.error('--checkpoint must name an existing checkpoint file')
    os.environ['DIFFSYNTH_STANDARD_WAN_TEACHER'] = '1'
    os.environ['DIFFSYNTH_FORCE_FP32_TIME_MLP'] = '1'
    sys.path.insert(0, str(ROOT))
    from linmorph import inference as core
    from linmorph import sampling as standard
    core.import_runtime_deps()
    if args.device.startswith('cuda:'):
        core.torch.cuda.set_device(args.device)
    runtime = SimpleNamespace(**cfg['model'], model_dir=str(args.model_dir.resolve()),
                              device=args.device, dtype='bf16', memory_mode=args.memory_mode,
                              resident_min_free_gb=30.0)
    blocks = core.parse_block_indices(runtime.gdn_block_indices)
    mode, _ = core.select_memory_mode(runtime)
    pipe = core.build_pipeline(runtime, blocks, mode, replace_gdn=not args.base_model)
    core.maybe_force_fp32_time_mlp(pipe)
    if args.checkpoint:
        core.load_checkpoint_into_pipe(pipe, str(args.checkpoint.resolve()), blocks, core.torch.bfloat16)
    if args.scheduler == 'unipc':
        source = args.unipc_source or standard.DEFAULT_UNIPC_SOURCE
        cls = standard.load_unipc_class(source.resolve(), standard.DEFAULT_UNIPC_DEPS)
        pipe.scheduler = standard.DiffSynthUniPCAdapter(cls)
    sample = dict(cfg['sampling'])
    fps = sample.pop('fps')
    plan['negative_prompt'] = args.negative_prompt if args.negative_prompt is not None else standard.DEFAULT_NEGATIVE_PROMPT
    with core.torch.inference_mode():
        video = pipe(prompt=args.prompt, negative_prompt=plan['negative_prompt'], seed=args.seed, tiled=True, **sample)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    core.save_video_file(core.extract_video(video), str(args.output), fps=fps)
    args.output.with_suffix('.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
