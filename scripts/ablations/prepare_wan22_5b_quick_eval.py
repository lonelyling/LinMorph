"""Prepare and run paired Wan2.2 diagnostics; GPU stages require an idle A800."""
import sys
from pathlib import Path

# Support running this entry point directly from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
import csv
import json
import math
import os
from linmorph.wan22 import ROOT, OUT, CKPT, BASE, BLOCKS, CONFIG, MODELS, header, inputs, idle_gpu, load_pipeline, kwargs

def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    h, digest = header(CKPT)
    keys = [k for k in h if k != '__metadata__']
    blocks = sorted({int(k.split('.')[1]) for k in keys if '.self_attn.attn.q_proj.' in k})
    assert blocks == BLOCKS, blocks
    lora_keys = [k for k in keys if '.lora_' in k]
    assert len(lora_keys) == 168
    assert h['blocks.0.self_attn.attn.q_proj.weight']['shape'] == [6144, 3072]
    assert h['blocks.0.self_attn.attn.v_proj.weight']['shape'] == [12288, 3072]
    assert all(h[k]['shape'][0] == 32 for k in lora_keys if '.lora_A.' in k)
    shards = sorted(BASE.glob('diffusion_pytorch_model-*.safetensors'))
    for p in [BASE/'config.json', BASE/'models_t5_umt5-xxl-enc-bf16.pth', BASE/'Wan2.2_VAE.pth', BASE/'google/umt5-xxl', *shards]:
        if not p.exists():
            raise FileNotFoundError(p)
    assert shards, 'No base DiT shards'
    receipt = dict(checkpoint=str(CKPT), bytes=CKPT.stat().st_size, mtime=CKPT.stat().st_mtime,
                   tensor_count=len(keys), embedded_lora_tensors=len(lora_keys),
                   safetensors_header_sha256=digest, note='Header/size validation only; not a full tensor checksum.',
                   gdn_blocks=BLOCKS, gdn_expand_v=2., gdn_head_dim=256, lora_rank=32,
                   selection='First-test candidate, not a validated globally best checkpoint.')
    write_json(OUT/'checkpoint_receipt.json', receipt)
    write_json(OUT/'manifest.json', dict(protocol=CONFIG, models=MODELS, inputs=inputs(),
        video_count=48, status='PREPARED_NO_SCORES', checkpoint=receipt))
    write_json(OUT/'prompt_map.json', {str(OUT/'videos'/model/(row['id']+'.mp4')): row['prompt']
        for model in MODELS for row in inputs()})
    # Major operations from the actual 5B config. GDN recurrence is an equivalent
    # operation count, not an executed fused/chunk kernel profiler count.
    cfg = json.loads((BASE/'config.json').read_text())
    d, m, layers = cfg['dim'], cfg['ffn_dim'], cfg['num_layers']
    n = ((CONFIG['num_frames']-1)//4+1) * (CONFIG['height']//32) * (CONFIG['width']//32)
    text, nh, kd, vd = 512, 24, 6144, 12288
    soft = 8*n*d*d + 4*n*n*d
    cross = 4*n*d*d + 4*text*d*d + 4*n*text*d
    ffn = 4*n*d*m
    gdn = 2*n*d*(2*kd+3*vd+2*nh) + 2*n*4*(2*kd+vd) + 8*n*nh*256*512
    lora = 0
    for k in lora_keys:
        shape = h[k]['shape']
        module = k.split('.lora_')[0]
        tokens = text if '.cross_attn.k' in module or '.cross_attn.v' in module else n
        lora += 2*tokens*math.prod(shape)
    flop_rows = []
    for model, count in [('wan22_base',0), ('step71000',24)]:
        total = layers*(cross+ffn) + (layers-count)*soft + count*gdn + (lora if count else 0)
        flop_rows.append(dict(model=model, latent_grid=[21,46,80], token_count=n,
            blocks_major_flops=total, blocks_major_tflops=total/1e12,
            generation_blocks_major_tflops=total/1e12*CONFIG['num_inference_steps']*2,
            unmerged_lora_flops=lora if count else 0))
    write_json(OUT/'analytical_flops.json', dict(status='ANALYTICAL_BLOCKS_ONLY', rows=flop_rows,
        scope='Batch=1 single positive/negative DiT call; FMA=2. Generation assumes two calls/step, CFG merge disabled.',
        excludes='Patch/head/text/time embeddings, elementwise, norms, RoPE, softmax, T5/VAE. GDN is recurrent-equivalent, not executed chunk FLOPs.',
        warning='Do not label these block-major estimates as measured full-model FLOPs.'))
    if not (OUT/'manual_review.csv').exists():
        with (OUT/'manual_review.csv').open('w') as f:
            w = csv.DictWriter(f, fieldnames=['model','id','category','prompt_match_0_2','motion_0_2','artifacts_0_2','notes'])
            w.writeheader()
            for model in MODELS:
                for row in inputs():
                    w.writerow(dict(model=model,id=row['id'],category=row['category']))
    print(json.dumps(dict(status='PREPARED', checkpoint=str(CKPT), videos=48,
                         output=str(OUT), flops=flop_rows), ensure_ascii=False, indent=2))


def generate(args):
    model = args.model
    core, pipe = load_pipeline(args,model)
    rows = inputs()[:args.limit] if args.limit else inputs()
    folder = OUT/'videos'/model
    folder.mkdir(parents=True,exist_ok=True)
    meta_path = folder/'inference_meta.json'
    expected_meta = dict(model=model,checkpoint=str(CKPT) if model=='step71000' else None,
                        protocol=CONFIG,memory_mode=args.memory_mode,inputs=inputs(),
                        header_sha256=header(CKPT)[1] if model=='step71000' else None)
    if meta_path.exists() and json.loads(meta_path.read_text()) != expected_meta:
        raise RuntimeError('Existing output uses a different protocol; use a fresh output directory.')
    write_json(meta_path,expected_meta)
    with core.torch.inference_mode():
        for row in rows:
            path = folder/(row['id']+'.mp4')
            if path.exists():
                continue
            video = core.extract_video(pipe(**kwargs(row)))
            temp = path.with_name(path.stem+'.partial.mp4')
            core.save_video_file(video,str(temp),CONFIG['fps'])
            temp.replace(path)
            print('SAVED',path,flush=True)
    del pipe
    core.torch.cuda.empty_cache()


def compare():
    from linmorph import inference as core
    count = 0
    for row in inputs():
        paths = [OUT/'videos'/m/(row['id']+'.mp4') for m in MODELS]
        if not all(p.exists() for p in paths):
            continue
        dest = OUT/'side_by_side'/(row['id']+'_ours_left_base_right.mp4')
        dest.parent.mkdir(exist_ok=True)
        if not core.make_side_by_side(str(paths[0]),str(paths[1]),str(dest)):
            raise RuntimeError('ffmpeg comparison failed')
        count += 1
    print('PAIRS',count)
    if count != 24:
        raise RuntimeError(f'Incomplete generation: {count}/24 pairs. Complete both models before evaluation.')


def benchmark(args):
    # Forward and end-to-end timings are kept separate from quality generation.

    from linmorph.benchmark import Guard, smi, timing
    import torch
    uuid = next(r['uuid'] for r in smi() if r['index']==args.gpu)
    guard = Guard(uuid)
    try:
        core, pipe = load_pipeline(args,args.model)
        x = torch.randn(1,48,21,46,80,device='cuda',dtype=torch.bfloat16)
        context = torch.randn(1,512,4096,device='cuda',dtype=torch.bfloat16)
        t = torch.tensor([500.],device='cuda',dtype=torch.bfloat16)
        forward = timing(lambda:pipe.model_fn(dit=pipe.dit,latents=x,timestep=t,context=context),
                         guard,warmup=3,repeats=10)
        del x, context, t
        end_to_end = timing(lambda:pipe(**kwargs(inputs()[0])),guard,warmup=1,repeats=3)
        result = dict(model=args.model,protocol=CONFIG,memory_mode=args.memory_mode,
                      forward=forward,end_to_end=end_to_end,
                      environment=dict(torch=core.torch.__version__,cuda=core.torch.version.cuda,
                          flash_attn=__import__('flash_attn').__version__,peft=__import__('peft').__version__,
                          fla=__import__('fla').__version__,python=sys.version),
                      end_to_end_scope='Includes T5, denoising, VAE and PIL conversion; excludes checkpoint/model load and video encoding.',
                      forward_scope='Single synthetic-latent/context positive DiT forward; batch 1, unmerged LoRA.',
                      provenance=guard.close())
        write_json(OUT/'latency'/f'{args.model}.json',result)
        print('LATENCY_SAVED; valid_exclusive=', result['provenance']['valid_exclusive'])
    finally:
        if not guard.stop.is_set():
            guard.close()


def score_preflight():
    import importlib
    import decord
    import torch
    import numpy
    modules = ['subject_consistency','background_consistency','motion_smoothness',
               'dynamic_degree','aesthetic_quality','imaging_quality']
    for name in modules:
        importlib.import_module('vbench.'+name)
        print('SCORE_IMPORT_OK',name,flush=True)
    cache = ROOT/'.cache/vbench'
    for name in ['dino_model/dino_vitbase16_pretrain.pth',
                 'dino_model/facebookresearch_dino_main/hubconf.py',
                 'clip_model/ViT-B-32.pt','clip_model/ViT-L-14.pt',
                 'aesthetic_model/emb_reader/sa_0_4_vit_l_14_linear.pth',
                 'amt_model/amt-s.pth','raft_model/models/raft-things.pth',
                 'pyiqa_model/musiq_spaq_ckpt-358bb6af.pth']:
        if not (cache/name).is_file():
            raise FileNotFoundError(cache/name)
    sample = ROOT/'output/validation/24gdn/nonmemory_gdn_full_ffn_softmax_lora_r32_step71000/step71000/sample_01_seed43.mp4'
    video = decord.VideoReader(str(sample),ctx=decord.cpu(0))
    info = dict(status='CPU_SCORE_PREFLIGHT_OK',torch=torch.__version__,numpy=numpy.__version__,
                pyiqa=importlib.import_module('pyiqa').__version__,frames=len(video),
                first_frame_shape=list(video[0].shape),six_dimensions=modules)
    write_json(OUT/'score_preflight.json',info)
    print(json.dumps(info,indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['prepare','generate','compare','benchmark','preflight','score-preflight'],default='prepare')
    p.add_argument('--model',choices=MODELS,default='step71000')
    p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--memory-mode',choices=['resident','cpu_offload'],default='cpu_offload')
    p.add_argument('--limit',type=int,default=0,help='Debug only: first N samples, full protocol is 24/model.')
    p.add_argument('--dry-run',action='store_true')
    args = p.parse_args()
    if args.limit < 0 or args.limit > 24:
        p.error('--limit must be between 0 and 24')
    if args.stage == 'prepare':
        prepare()
    elif args.stage == 'score-preflight':
        score_preflight()
    elif args.stage == 'preflight':
        from linmorph import inference as core
        core.import_runtime_deps()
        import peft, flash_attn
        from peft.tuners.lora.layer import LoraLayer
        print('CPU runtime imports OK',core.torch.__version__,peft.__version__,flash_attn.__version__)
    elif args.dry_run:
        print(json.dumps(dict(stage=args.stage,model=args.model,protocol=CONFIG,
                              samples=args.limit or 24,output=str(OUT)),indent=2))
    elif args.stage == 'compare':
        compare()
    else:
        if not (OUT/'manifest.json').exists():
            raise RuntimeError('Run --stage prepare first')
        if args.stage == 'benchmark' and args.memory_mode != 'resident':
            raise ValueError('Benchmark requires --memory-mode resident')
        idle_gpu(args.gpu)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        os.environ['PYTHONNOUSERSITE'] = '1'
        {'generate':generate,'benchmark':benchmark}[args.stage](args)


if __name__ == "__main__":
    main()
