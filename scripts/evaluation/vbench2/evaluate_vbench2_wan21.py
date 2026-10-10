#!/usr/bin/env python3
"""Validate inputs and evaluate a selected Wan2.1 GDN checkpoint with the official VBench2 implementation."""

import sys
from pathlib import Path

# Support running this entry point directly from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import argparse, importlib.util, json, os, subprocess
from types import SimpleNamespace
from linmorph import vbench2 as generation
ROOT=generation.ROOT;SOURCE=generation.SOURCE;RUN=generation.RUN

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimensions",nargs="+",choices=generation.DIMENSIONS,default=list(generation.DIMENSIONS))
    parser.add_argument("--device",default="cuda")
    parser.add_argument("--check-only",action="store_true")
    args=parser.parse_args()
    if len(args.dimensions)!=len(set(args.dimensions)):raise ValueError("Duplicate dimensions")
    tasks=generation.plan(SimpleNamespace(dimensions=args.dimensions,output=RUN/"videos",seed=42))
    missing=[t["output_path"] for t in tasks if not Path(t["output_path"]).is_file() or Path(t["output_path"]).stat().st_size==0]
    for path in [SOURCE,SOURCE/"vbench2/third_party/Instance_detector",SOURCE/"vbench2/third_party/YOLO-World",SOURCE/"vbench2/third_party/LLaVA_NeXT"]:sys.path.insert(0,str(path))
    required=["torch","torchvision","transformers","decord","mmengine","mmdet","mmyolo","lvis","retinaface","flash_attn"]
    modules={x:bool(importlib.util.find_spec(x)) for x in required}
    # Instance detector and YOLO World are vendored in the pinned official checkout.
    vendored=[SOURCE/"vbench2/third_party/Instance_detector",SOURCE/"vbench2/third_party/YOLO-World",SOURCE/"vbench2/third_party/LLaVA_NeXT"]
    core=json.loads((generation.WEIGHTS_RECORD/"download_status.json").read_text()) if (generation.WEIGHTS_RECORD/"download_status.json").exists() else {}
    cache=json.loads((generation.WEIGHTS_RECORD/"cache_status.json").read_text()) if (generation.WEIGHTS_RECORD/"cache_status.json").exists() else {}
    report=dict(dimensions=args.dimensions,expected_videos=len(tasks),missing_videos=len(missing),missing_examples=missing[:5],missing_dependencies=[x for x,v in modules.items() if not v],core_weights_status=core.get("status"),cache_status=cache.get("status"))
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    RUN.mkdir(parents=True,exist_ok=True)
    (RUN/"preflight.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    if args.check_only:return
    if missing or report["missing_dependencies"] or cache.get("status")!="complete":raise SystemExit("Preflight incomplete: see preflight.json")
    os.environ["VBENCH2_CACHE_DIR"]=str(ROOT/".cache/vbench2")
    os.environ["HF_HOME"]=str(ROOT/".cache/huggingface")
    os.environ["HF_HUB_CACHE"]=str(ROOT/".cache/huggingface/hub")
    os.environ["HF_HUB_OFFLINE"]="1";os.environ["TRANSFORMERS_OFFLINE"]="1";os.environ["USE_HF"]="1"
    os.environ["TORCH_HOME"]=str(ROOT/".cache/torch")
    for path in [SOURCE,*vendored]:sys.path.insert(0,str(path))
    os.chdir(SOURCE)
    from vbench2 import VBench2
    for dim in args.dimensions:
        output=RUN/"evaluation"/dim
        name=f"{generation.RUN_NAME}_{dim}"
        if (output/f"{name}_eval_results.json").exists():
            # The report validates coverage; a valid completed dimension is resumable.
            subprocess.run([sys.executable,str(ROOT/"scripts/evaluation/vbench2/report_vbench2_wan21.py")],check=True)
            summary=json.loads((RUN/"summary.json").read_text())
            row=next(r for r in summary["dimensions"] if r["dimension"]==dim)
            if row["status"]=="evaluated":continue
        evaluator=VBench2(args.device,str(SOURCE/"vbench2/VBench2_full_info.json"),str(output))
        evaluator.evaluate(videos_path=str(RUN/"videos"/dim),name=name,dimension_list=[dim],local=True)
        subprocess.run([sys.executable,str(ROOT/"scripts/evaluation/vbench2/report_vbench2_wan21.py")],check=True)
    print("[DONE] requested dimensions evaluated",flush=True)
if __name__=="__main__":main()
