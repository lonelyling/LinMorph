#!/usr/bin/env python3
"""Matched single-A800 Wan2.2 5B latency with separate forward/denoise/end-to-end scopes."""

import sys
from pathlib import Path

# Support running this entry point directly from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import csv
import importlib
import json
import os
import statistics
import time
from linmorph.wan22 import ROOT, CONFIG, CKPT, inputs, idle_gpu, load_pipeline, header

OUT=ROOT/'output/benchmarks/wan22_5b_736x1280_f81'

def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(obj,indent=2)+'\n');temp.replace(path)

def summarize(values):
    ordered=sorted(values)
    return dict(median_ms=statistics.median(values),mean_ms=statistics.mean(values),
        p90_ms=ordered[round((len(ordered)-1)*.9)],min_ms=min(values),max_ms=max(values),samples_ms=values)

def report(output):
    rows=[]
    for model in ['wan22_base','step71000']:
        path=output/(model+'.json')
        if not path.exists():
            raise FileNotFoundError(f'Missing latency result: {path}')
        r=json.loads(path.read_text())
        if r['status']=='FAILED':
            raise RuntimeError(f'Failed benchmark: {path}')
        rows.append(dict(model=model,status=r['status'],forward_ms=r['forward']['wall']['median_ms'],
            denoising_seconds=r['denoising']['wall']['median_ms']/1000,
            end_to_end_seconds=r['end_to_end']['wall']['median_ms']/1000,
            peak_allocated_gib=r['end_to_end']['peak_allocated_bytes']/1024**3))
    ratios={k:rows[0][k]/rows[1][k] for k in ['forward_ms','denoising_seconds','end_to_end_seconds']}
    save(output/'summary.json',dict(rows=rows,baseline_divided_by_ours=ratios,
        valid_for_main_table=all(r['status']=='EXCLUSIVE_VALID' for r in rows)))
    with (output/'summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(dict(rows=rows,baseline_divided_by_ours=ratios),indent=2))

def run(args):
    idle_gpu(args.gpu)
    os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
    os.environ['DIFFSYNTH_DISABLE_FLASH_ATTN']='0'

    from linmorph.benchmark import Guard,smi,timing
    uuid=next(r['uuid'] for r in smi() if r['index']==args.gpu)
    guard=Guard(uuid)
    result=dict(model=args.model,status='FAILED',protocol=CONFIG,batch_size=1,memory_mode='resident',
        checkpoint=str(CKPT) if args.model=='step71000' else None,
        checkpoint_header_sha256=header(CKPT)[1] if args.model=='step71000' else None)
    try:
        args.memory_mode='resident'
        core,pipe=load_pipeline(args,args.model)
        torch=core.torch
        # Match all original Softmax and cross-attention to FA2 on A800.
        base_module=importlib.import_module('diffsynth.models.wan_video_dit')
        base_module.FLASH_ATTN_3_AVAILABLE=False
        base_module.SAGE_ATTN_AVAILABLE=False
        if not base_module.FLASH_ATTN_2_AVAILABLE:
            raise RuntimeError('FlashAttention-2 is required; no silent backend substitution.')
        torch.manual_seed(42)
        x=torch.randn(1,48,21,46,80,device='cuda',dtype=torch.bfloat16)
        context=torch.randn(1,512,4096,device='cuda',dtype=torch.bfloat16)
        t=torch.tensor([500.],device='cuda',dtype=torch.bfloat16)
        result['forward']=timing(lambda:pipe.model_fn(dit=pipe.dit,latents=x,timestep=t,context=context),
            guard,warmup=args.forward_warmup,repeats=args.forward_repeats)
        del x,context,t
        records=[];state={}
        original_model_fn=pipe.model_fn
        original_step=pipe.scheduler.step
        original_set=pipe.scheduler.set_timesteps
        def reset(*a,**kw):
            state.clear();state['steps']=0
            return original_set(*a,**kw)
        def model_fn(*a,**kw):
            if 'start' not in state:
                torch.cuda.synchronize();state['start']=time.perf_counter()
                state['start_event']=torch.cuda.Event(enable_timing=True);state['start_event'].record()
            return original_model_fn(*a,**kw)
        def step(*a,**kw):
            value=original_step(*a,**kw);state['steps']+=1
            if state['steps']==CONFIG['num_inference_steps']:
                end=torch.cuda.Event(enable_timing=True);end.record();torch.cuda.synchronize()
                records.append(dict(wall_ms=(time.perf_counter()-state['start'])*1000,
                                    cuda_ms=state['start_event'].elapsed_time(end)))
            return value
        pipe.scheduler.set_timesteps=reset;pipe.scheduler.step=step;pipe.model_fn=model_fn
        kw={k:v for k,v in CONFIG.items() if k not in ['fps','dtype','scheduler']}
        row=inputs()[0]
        kw.update(prompt=row['prompt'],seed=row['seed'],cfg_merge=False,progress_bar_cmd=lambda x:x)
        result['end_to_end']=timing(lambda:pipe(**kw),guard,warmup=args.generation_warmup,repeats=args.generation_repeats)
        expected=1+args.generation_warmup+args.generation_repeats
        if len(records)!=expected:
            raise RuntimeError(f'Denoising instrument captured {len(records)}/{expected} generations')
        measured=records[-args.generation_repeats:]
        result['denoising']=dict(wall=summarize([r['wall_ms'] for r in measured]),
            cuda=summarize([r['cuda_ms'] for r in measured]),first_call=records[0],
            warmup=args.generation_warmup,repeats=args.generation_repeats)
        result['scopes']=dict(forward='Synthetic latents/context; single positive DiT call, batch=1.',
            denoising='First positive DiT call through final scheduler update; all 50 steps, separate positive/negative CFG calls.',
            end_to_end='T5 + denoising + tiled VAE + PIL conversion; excludes model/checkpoint loading and mp4 encoding.')
        result['environment']=dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(0),flash_attn=importlib.import_module('flash_attn').__version__,
            fla=importlib.import_module('fla').__version__,peft=importlib.import_module('peft').__version__,
            original_attention='FlashAttention-2',gdn_attention='FLA GatedDeltaNet chunk',lora='unmerged; scale=1',
            cfg_merge=False,time_mlp='fp32',resident=True,tiled=True)
        result['provenance']=guard.close()
        result['status']='EXCLUSIVE_VALID' if result['provenance']['valid_exclusive'] else 'DIAGNOSTIC_ONLY'
        save(args.output/(args.model+'.json'),result)
        print('SAVED',args.output/(args.model+'.json'),result['status'])
    except Exception as exc:
        result['error']=repr(exc)
        if not guard.stop.is_set():
            result['provenance']=guard.close()
        save(args.output/(args.model+'.json'),result)
        raise
    finally:
        if not guard.stop.is_set():guard.close()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=['wan22_base','step71000'],default='step71000')
    p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--output',type=Path,default=OUT)
    p.add_argument('--forward-warmup',type=int,default=3)
    p.add_argument('--forward-repeats',type=int,default=10)
    p.add_argument('--generation-warmup',type=int,default=1)
    p.add_argument('--generation-repeats',type=int,default=3)
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--report',action='store_true')
    args=p.parse_args()
    if min(args.forward_warmup,args.generation_warmup)<0 or min(args.forward_repeats,args.generation_repeats)<1:
        p.error('Warmups must be nonnegative, repeats must be positive.')
    if args.report:report(args.output)
    elif args.dry_run:
        print(json.dumps(dict(model=args.model,gpu=args.gpu,output=str(args.output),protocol=CONFIG,
            scopes=['forward','denoising','end_to_end'],forward=dict(warmup=args.forward_warmup,repeats=args.forward_repeats),
            generation=dict(first_call=1,warmup=args.generation_warmup,repeats=args.generation_repeats),
            memory_mode='resident',lora='unmerged'),indent=2))
    else:run(args)

if __name__=='__main__':main()
