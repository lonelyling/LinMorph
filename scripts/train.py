#!/usr/bin/env python3
"""Portable launcher for the preserved LinMorph training implementations."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['config', 'model-dir', 'data-root', 'metadata', 'output']:
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--init-checkpoint', type=Path)
    parser.add_argument('--num-processes', type=int, default=1)
    parser.add_argument('--accelerate-config', type=Path, default=ROOT / 'configs/accelerate.yaml')
    parser.add_argument('--dry-run', action='store_true')
    args, extra = parser.parse_known_args()
    if args.num_processes < 1:
        parser.error('--num-processes must be positive')
    cfg = json.loads(args.config.read_text())
    if cfg.get('requires_init') and args.init_checkpoint is None:
        parser.error('This stage requires --init-checkpoint; see docs/training.md')
    model = args.model_dir.resolve()
    shards = sorted(model.glob('diffusion_pytorch_model-*.safetensors'))
    dit = [str(p) for p in shards] if shards else str(model / 'diffusion_pytorch_model.safetensors')
    vae = model / cfg.get('vae', 'Wan2.1_VAE.pth')
    paths = [dit, str(model / 'models_t5_umt5-xxl-enc-bf16.pth'), str(vae)]
    flags = dict(cfg['arguments'])
    flags.update(model_paths=json.dumps(paths), tokenizer_path=str(model / 'google/umt5-xxl'),
                 dataset_base_path=str(args.data_root.resolve()),
                 dataset_metadata_path=str(args.metadata.resolve()), output_path=str(args.output.resolve()))
    if args.init_checkpoint:
        flags['student_init_ckpt'] = str(args.init_checkpoint.resolve())
    entry = ROOT / 'examples/wanvideo/model_training' / cfg['entrypoint']
    if not entry.is_file():
        parser.error(f'Missing entry point: {entry}')
    cmd = [sys.executable, '-m', 'accelerate.commands.launch', '--config_file',
           str(args.accelerate_config.resolve()), '--num_processes', str(args.num_processes), str(entry)]
    for key, value in flags.items():
        if isinstance(value, bool):
            if value:
                cmd.append('--' + key)
        elif value is not None:
            cmd.extend(['--' + key, str(value)])
    cmd.extend(extra[1:] if extra[:1] == ['--'] else extra)
    env = os.environ.copy()
    env.pop('DIFFSYNTH_QWORLD_COMMON_CONFIG', None)
    env.update({k: str(v) for k, v in cfg.get('environment', {}).items()})
    env['PYTHONPATH'] = os.pathsep.join([str(ROOT), str(entry.parent)])
    print(json.dumps({'environment': cfg.get('environment', {}), 'command': shlex.join(cmd)}, indent=2))
    if args.dry_run:
        return
    required = [args.metadata, args.data_root, vae, model / 'models_t5_umt5-xxl-enc-bf16.pth',
                model / 'google/umt5-xxl', *([Path(p) for p in dit] if isinstance(dit, list) else [Path(dit)])]
    if args.init_checkpoint:
        required.append(args.init_checkpoint)
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        parser.error('Missing inputs: ' + ', '.join(missing))
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Use a fresh output directory. For full-state resume use the original entry point with --resume.')
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


if __name__ == '__main__':
    main()
