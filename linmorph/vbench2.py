"""Shared vbench2 implementation; command-line entry points live in scripts/."""
import hashlib, json, os
from pathlib import Path
from linmorph import vbench as standard


ROOT=Path(__file__).resolve().parents[1]


SOURCE=Path(os.environ.get("VBENCH2_SOURCE", str(ROOT/".deps/VBench/VBench-2.0")))


DIMENSIONS=tuple(p.stem for p in sorted((SOURCE/"prompts/prompt").glob("*.txt")))


DEFAULT_CHECKPOINT=ROOT/"output/train/wan21_13B_warmup500_fullstate_8gpu_ga2/stage2_mapping_full_finetune/model-step-40000.safetensors"


CHECKPOINT=Path(os.environ.get("CHECKPOINT",str(DEFAULT_CHECKPOINT))).resolve()


GDN_BLOCK_INDICES=os.environ.get("GDN_BLOCK_INDICES",standard.DEFAULT_GDN_BLOCKS)


RUN=Path(os.environ.get("RUN_ROOT",str(ROOT/"output/vbench2/stage2_step40000"))).resolve()


RUN_NAME=os.environ.get("RUN_NAME",RUN.name)


WEIGHTS_RECORD=Path(os.environ.get("VBENCH2_WEIGHTS_RECORD", str(ROOT/"output/vbench2/weights")))


def plan(args):
    tasks=[];seen={};metadata=json.loads((SOURCE/"vbench2/VBench2_full_info.json").read_text())
    for dim in args.dimensions:
        originals=[x.strip() for x in (SOURCE/"prompts/prompt"/f"{dim}.txt").read_text().splitlines() if x.strip()]
        aug=[x.strip() for x in (SOURCE/"prompts/prompt_aug/wanx_aug_prompt"/f"{dim}.txt").read_text().splitlines() if x.strip()]
        if len(originals)!=len(aug):raise ValueError(f"Prompt alignment mismatch: {dim}")
        expected=[r["prompt_en"] for r in metadata if dim in r["dimension"]]
        if originals!=expected:raise ValueError(f"Prompt list differs from official metadata: {dim}")
        for original, enhanced in zip(originals,aug):
            for index in range(20 if dim=="Diversity" else 3):
                filename=f"{original[:180]}-{index}.mp4"
                if "/" in filename or "\x00" in filename or len(filename.encode())>255:raise ValueError("Invalid video filename")
                path=args.output/dim/filename
                if str(path) in seen:raise ValueError(f"Truncated filename collision: {dim}")
                seen[str(path)]=original
                seed=int.from_bytes(hashlib.sha256(f"vbench2\x00{args.seed}\x00{dim}\x00{original}\x00{index}".encode()).digest()[:4],"big")
                tasks.append(dict(ordinal=len(tasks),dimension=dim,original_prompt=original,inference_prompt=enhanced,sample_index=index,seed=seed,output_path=str(path)))
    return tasks

