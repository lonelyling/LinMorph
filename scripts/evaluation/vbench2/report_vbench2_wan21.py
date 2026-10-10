#!/usr/bin/env python3
"""Report complete per-video VBench2 results for a configurable experiment."""

import sys
from pathlib import Path

# Support running this entry point directly from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import argparse, csv, fcntl, json, math
from datetime import datetime, timezone
from types import SimpleNamespace
from linmorph import vbench2 as g

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-complete",action="store_true")
    args=parser.parse_args()
    run=g.RUN;run.mkdir(parents=True,exist_ok=True)
    lock=(run/".report.lock").open("a");fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
    tasks=g.plan(SimpleNamespace(dimensions=list(g.DIMENSIONS),output=run/"videos",seed=42))
    rows=[];scores={};errors={}
    for dim in g.DIMENSIONS:
        expected=[t for t in tasks if t["dimension"]==dim]
        available=sum(Path(t["output_path"]).is_file() and Path(t["output_path"]).stat().st_size>0 for t in expected)
        result=run/"evaluation"/dim/f"{g.RUN_NAME}_{dim}_eval_results.json"
        score=None
        if result.is_file():
            try:
                value,details=json.loads(result.read_text())[dim]
                value=float(value)
                if not math.isfinite(value) or available!=len(expected):raise ValueError("Incomplete inputs or nonfinite score")
                intended={str(Path(t["output_path"]).resolve()) for t in expected if dim!="Diversity" or t["sample_index"]==0}
                actual=[str(Path(r["video_path"]).resolve()) for r in details]
                if set(actual)!=intended or len(actual)!=len(intended):raise ValueError("Result video coverage mismatch")
                score=value;scores[dim]=value
            except Exception as e:errors[dim]=str(e)
        rows.append(dict(dimension=dim,expected_videos=len(expected),available_videos=available,raw_score=score,score_x100=None if score is None else score*100,status="evaluated" if score is not None else "pending"))
    aggregate=None
    if len(scores)==18:
        sys.path.insert(0,str(g.SOURCE/"scripts"))
        from cal_final_score import get_creativity_score,get_commonsense_score,get_controllability_score,get_human_fidelity_score,get_physics_score,get_final_score
        official={k.replace("_"," "):v for k,v in scores.items()}
        aggregate=dict(Creativity=get_creativity_score(official),Commonsense=get_commonsense_score(official),Controllability=get_controllability_score(official),Human_Fidelity=get_human_fidelity_score(official),Physics=get_physics_score(official))
        aggregate["Total"]=get_final_score(*aggregate.values())
    report=dict(updated_at=datetime.now(timezone.utc).isoformat(),benchmark="VBench-2.0",run_name=g.RUN_NAME,checkpoint=str(g.CHECKPOINT),gdn_block_indices=g.GDN_BLOCK_INDICES,dimensions=rows,raw_scores=scores,aggregate=aggregate,errors=errors)
    temp=run/"summary.json.tmp";temp.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n");temp.replace(run/"summary.json")
    with (run/"dimensions.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=[f"# {g.RUN_NAME} / VBench 2.0 实验报告","",f"更新时间：{report['updated_at']}","",f"Checkpoint：`{g.CHECKPOINT}`。",f"GDN 替换层：`{g.GDN_BLOCK_INDICES}`。","","生成协议：832×480，81 帧，16 fps，bf16，Wan UniPC 50 steps，CFG=6，shift=8；官方 Wanx 增强提示词；普通提示词 3 样本、Diversity 20 样本；base seed=42，每视频独立固定 seed。","",f"视频 {sum(r['available_videos'] for r in rows)}/3860；评测 {len(scores)}/18 维。","","| 维度 | 已有/预期视频 | 分数 ×100 |","|---|---:|---:|"]
    for r in rows:lines.append(f"| {r['dimension']} | {r['available_videos']}/{r['expected_videos']} | {'待评测' if r['score_x100'] is None else format(r['score_x100'],'.2f')} |")
    if aggregate:
        lines += ["","| 汇总 | 分数 ×100 |","|---|---:|"]+[f"| {k} | {v*100:.2f} |" for k,v in aggregate.items()]
    else:lines += ["","18 维全部完成且视频覆盖校验通过前不计算总分。"]
    if errors:lines += ["","结果校验错误：",json.dumps(errors,ensure_ascii=False,indent=2)]
    (run/"实验报告.md").write_text("\n".join(lines)+"\n")
    if aggregate:
        final=dict(run_name=g.RUN_NAME,checkpoint=str(g.CHECKPOINT),gdn_block_indices=g.GDN_BLOCK_INDICES,raw_aggregate=aggregate,percentage_scores={k:v*100 for k,v in aggregate.items()},completed_dimensions=18,updated_at=report["updated_at"])
        (run/"final_scores.json").write_text(json.dumps(final,ensure_ascii=False,indent=2)+"\n")
        (run/".evaluation_complete").touch()
        print("\nVBench 2.0 final scores (x100)",flush=True)
        for k,v in aggregate.items():print(f"{k}: {v*100:.4f}",flush=True)
        print(f"Report: {run / '实验报告.md'}",flush=True)
    else:
        (run/"final_scores.json").unlink(missing_ok=True)
        (run/".evaluation_complete").unlink(missing_ok=True)
        if args.require_complete:
            raise SystemExit(f"Incomplete VBench2 results: {len(scores)}/18; errors={errors}")
    print(json.dumps(dict(run_name=g.RUN_NAME,completed_dimensions=len(scores),videos=sum(r['available_videos'] for r in rows),errors=errors),ensure_ascii=False))
if __name__=="__main__":main()
