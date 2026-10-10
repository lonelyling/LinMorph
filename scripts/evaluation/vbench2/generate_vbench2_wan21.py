"""Generate official VBench 2.0 videos with a configurable Wan2.1 GDN checkpoint."""
import sys
from pathlib import Path

# Support running this entry point directly from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import argparse, json, time
from datetime import datetime, timezone
from types import SimpleNamespace
from linmorph import vbench as standard
from linmorph.vbench2 import DIMENSIONS, CHECKPOINT, GDN_BLOCK_INDICES, RUN, plan

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=RUN/"videos")
    parser.add_argument("--checkpoint",type=Path,default=CHECKPOINT)
    parser.add_argument("--gdn-block-indices",default=GDN_BLOCK_INDICES)
    parser.add_argument("--dimensions",nargs="+",choices=DIMENSIONS,default=list(DIMENSIONS))
    parser.add_argument("--num-shards",type=int,default=1)
    parser.add_argument("--shard-index",type=int,default=0)
    parser.add_argument("--max-tasks",type=int,default=0)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--device",default="cuda")
    parser.add_argument("--memory-mode",choices=["auto","resident","cpu_offload"],default="cpu_offload")
    parser.add_argument("--dry-run",action="store_true")
    args=parser.parse_args();args.output=args.output.resolve();args.checkpoint=args.checkpoint.resolve()
    if not 0<=args.shard_index<args.num_shards:raise ValueError("Invalid shard index")
    if len(args.dimensions)!=len(set(args.dimensions)):raise ValueError("Duplicate dimension")
    if not args.checkpoint.is_file():raise FileNotFoundError(args.checkpoint)
    import re,struct
    with args.checkpoint.open("rb") as f:
        header=json.loads(f.read(struct.unpack("<Q",f.read(8))[0]))
    actual=sorted({int(m.group(1)) for key in header for m in [re.search(r"blocks\.(\d+)\.self_attn\..*q_proj",key)] if m})
    selected=sorted(int(x) for x in args.gdn_block_indices.split(","))
    if actual!=selected:raise ValueError(f"Checkpoint GDN blocks {actual} differ from requested {selected}")
    tasks=plan(args);assigned=[t for t in tasks if t["ordinal"]%args.num_shards==args.shard_index]
    if args.max_tasks>0:assigned=assigned[:args.max_tasks]
    manifest=dict(benchmark="VBench-2.0",checkpoint=str(args.checkpoint),checkpoint_bytes=args.checkpoint.stat().st_size,gdn_block_indices=args.gdn_block_indices,prompt_source="official Wanx augmented per dimension",sampling=dict(width=832,height=480,num_frames=81,fps=16,steps=50,cfg_scale=6.0,flow_shift=8.0,dtype="bf16",sampler="Wan official UniPC",negative_prompt=standard.DEFAULT_NEGATIVE_PROMPT),dimensions=args.dimensions,all_tasks=len(tasks),shard_index=args.shard_index,num_shards=args.num_shards,diagnostic_limit=args.max_tasks,tasks=assigned)
    path=args.output.parent/f"generation_manifest_shard{args.shard_index:02d}.json"
    if path.exists():
        previous=json.loads(path.read_text())
        if any(previous.get(k)!=manifest.get(k) for k in ["checkpoint","gdn_block_indices","sampling"]):
            raise ValueError("Existing run belongs to a different checkpoint or configuration; use a new RUN_ROOT")
    standard.write_json(path,manifest)
    print(f"[PLAN] {len(tasks)} total videos; {len(assigned)} assigned to shard {args.shard_index}",flush=True)
    if args.dry_run:return
    runtime=SimpleNamespace(model_dir=standard.DEFAULT_MODEL_DIR,device=args.device,dtype="bf16",memory_mode=args.memory_mode,resident_min_free_gb=30.0,gdn_block_indices=args.gdn_block_indices,gdn_expand_v=2.0,gdn_head_dim=256,gdn_hidden_size=3072,gdn_num_heads=24)
    from linmorph import inference as core
    core.import_runtime_deps()
    if args.device.startswith("cuda:"):core.torch.cuda.set_device(args.device)
    indices=core.parse_block_indices(runtime.gdn_block_indices)
    mode,memory=core.select_memory_mode(runtime)
    pipe=core.build_pipeline(runtime,indices,mode,replace_gdn=True)
    core.maybe_force_fp32_time_mlp(pipe)
    core.load_checkpoint_into_pipe(pipe,str(args.checkpoint),indices,core.torch_dtype_from_arg(runtime.dtype))
    cls=standard.load_unipc_class(standard.DEFAULT_UNIPC_SOURCE,standard.DEFAULT_UNIPC_DEPS)
    pipe.scheduler=standard.DiffSynthUniPCAdapter(cls);pipe.eval()
    journal=args.output.parent/f"generation_journal_shard{args.shard_index:02d}.jsonl"
    with core.torch.inference_mode():
        for t in assigned:
            target=Path(t["output_path"])
            if target.is_file() and target.stat().st_size>0:continue
            target.parent.mkdir(parents=True,exist_ok=True)
            print(f"[RUN] {t['ordinal']}: {t['dimension']} seed={t['seed']}",flush=True)
            start=time.monotonic()
            value=pipe(prompt=t["inference_prompt"],negative_prompt=standard.DEFAULT_NEGATIVE_PROMPT,seed=t["seed"],tiled=True,height=480,width=832,num_frames=81,num_inference_steps=50,cfg_scale=6.0,sigma_shift=8.0)
            temp=target.with_name(target.stem+".partial.mp4")
            core.save_video_file(core.extract_video(value),str(temp),fps=16);temp.replace(target)
            with journal.open("a") as f:f.write(json.dumps(dict(**t,elapsed_seconds=time.monotonic()-start,completed_at=datetime.now(timezone.utc).isoformat()),ensure_ascii=False)+"\n")
            core.torch.cuda.empty_cache()
    print("[DONE] shard generation completed",flush=True)


if __name__ == "__main__":
    main()
