#!/usr/bin/env python3
"""Generate all VBench1 prompts for Wan2.2 5B or the fixed 24GDN/LoRA 71k model."""

import sys
from pathlib import Path

# Support running this entry point directly from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import argparse
from dataclasses import asdict
import hashlib
import json
import os
import time
from types import SimpleNamespace
from linmorph.wan22 import ROOT, BASE, CKPT, CONFIG, BLOCKS, idle_gpu, header, load_pipeline
from linmorph.vbench import ALL_DIMENSIONS, load_prompt_records, build_tasks

OUT = ROOT/'output/vbench_standard/wan22_5b_736x1280_f81'
FULL_INFO = ROOT/'.deps/VBench/vbench/VBench_full_info.json'

def write_json(path, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp = path.with_suffix(path.suffix+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    temp.replace(path)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=['step71000','wan22_base'],default='step71000')
    p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--num-shards',type=int,default=1)
    p.add_argument('--shard-index',type=int,default=0)
    p.add_argument('--temporal-flicker-candidates',type=int,default=25)
    p.add_argument('--min-sample-index',type=int,default=0)
    p.add_argument('--max-tasks',type=int,default=0,help='Smoke test only; resume without this limit.')
    p.add_argument('--output-root',type=Path,default=OUT)
    p.add_argument('--memory-mode',choices=['resident','cpu_offload'],default='cpu_offload')
    p.add_argument('--dry-run',action='store_true')
    args = p.parse_args()
    if args.max_tasks<0:
        p.error('--max-tasks must be nonnegative')
    output = args.output_root.resolve()/args.model/'raw_videos'
    settings = SimpleNamespace(full_info=FULL_INFO,prompt_source='original',dimensions=list(ALL_DIMENSIONS),
        prompts_per_dimension=0,samples_per_prompt=5,temporal_flicker_candidates=args.temporal_flicker_candidates,
        base_seed=42,num_shards=1,shard_index=0,min_sample_index=0,max_tasks=0)
    records = load_prompt_records(settings)
    complete_tasks = build_tasks(settings,records,output)
    settings.num_shards = args.num_shards
    settings.shard_index = args.shard_index
    settings.min_sample_index = args.min_sample_index
    settings.max_tasks = args.max_tasks
    tasks = build_tasks(settings,records,output)
    protocol = dict(model=args.model,model_dir=str(BASE),checkpoint=str(CKPT) if args.model=='step71000' else None,
        checkpoint_header_sha256=header(CKPT)[1] if args.model=='step71000' else None,
        sampling=CONFIG,cfg_merge=False,prompt_source='official original prompt_en, no enhancement',
        full_info_sha256=hashlib.sha256(FULL_INFO.read_bytes()).hexdigest(),
        gdn_blocks=BLOCKS if args.model=='step71000' else [])
    protocol_path = output.parent/'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise RuntimeError('Existing videos have a different protocol; select a fresh --output-root.')
    write_json(protocol_path,protocol)
    manifest = dict(protocol='VBench1 complete official prompt suite with local Wan2.2 generation settings',
        dimensions=list(ALL_DIMENSIONS),unique_prompts=len(records),raw_candidates=len(complete_tasks),
        final_required_videos=len(records)*5,temporal_prompts=sum('temporal_flickering' in r.dimensions for r in records),
        temporal_candidates=args.temporal_flicker_candidates,shard_index=args.shard_index,num_shards=args.num_shards,
        max_tasks=args.max_tasks,static_filter_required=True,tasks=[asdict(t) for t in tasks])
    write_json(output.parent/f'generation_manifest_shard{args.shard_index:02d}.json',manifest)
    print(json.dumps({k:v for k,v in manifest.items() if k!='tasks'}|dict(model=args.model,
        tasks_this_invocation=len(tasks),output=str(output)),ensure_ascii=False,indent=2),flush=True)
    if args.dry_run:
        return
    idle_gpu(args.gpu)
    os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
    core,pipe=load_pipeline(args,args.model)
    output.mkdir(parents=True,exist_ok=True)
    journal=output.parent/f'generation_journal_shard{args.shard_index:02d}.jsonl'
    with core.torch.inference_mode():
        for i,task in enumerate(tasks,1):
            dest=Path(task.output_path)
            if dest.exists():
                continue
            print(f'[{i}/{len(tasks)}] {dest.name}',flush=True)
            started=time.monotonic()
            kw={k:v for k,v in CONFIG.items() if k not in ['fps','dtype','scheduler']}
            value=pipe(**kw,prompt=task.inference_prompt,seed=task.seed,cfg_merge=False)
            temp=dest.with_name(dest.stem+f'.partial.{os.getpid()}.mp4')
            core.save_video_file(core.extract_video(value),str(temp),CONFIG['fps'])
            temp.replace(dest)
            with journal.open('a') as f:
                f.write(json.dumps(asdict(task)|dict(elapsed_seconds=time.monotonic()-started),ensure_ascii=False)+'\n')
    write_json(output.parent/f'generation_complete_shard{args.shard_index:02d}.json',dict(
        model=args.model,shard_index=args.shard_index,num_shards=args.num_shards,max_tasks=args.max_tasks,
        complete_this_invocation=all(Path(t.output_path).exists() for t in tasks),tasks=len(tasks)))

if __name__=='__main__':
    main()
