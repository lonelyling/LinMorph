"""Run matched text-only inference for base Softmax or 24-GDN checkpoints."""
import sys
from pathlib import Path

# Support running this entry point directly from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
import json
import os
from datetime import datetime
from linmorph.inference import PROJECT_ROOT, DEFAULT_GDN_BLOCKS, project_path, parse_block_indices, resolve_checkpoint, import_runtime_deps, torch_dtype_from_arg, select_memory_mode, extract_video, save_video_file, make_side_by_side, build_pipeline, load_checkpoint_into_pipe, maybe_force_fp32_time_mlp
from linmorph import inference as core

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate matched text-only samples for the base Softmax Wan2.2-TI2V-5B "
            "model or 24-GDN checkpoints."
        )
    )
    parser.add_argument(
        "--softmax_only",
        action="store_true",
        help=(
            "Run the original base DiT with all Softmax self-attention layers. "
            "No GDN replacement or training checkpoint is loaded."
        ),
    )
    parser.add_argument(
        "--random_work_dir",
        type=str,
        default="output/train/24gdn/train_24gdn_ib_random_init_control_lr5e-6_warmup10x",
        help="Random-init run directory. The latest model-step-*.safetensors is used by default.",
    )
    parser.add_argument(
        "--mapped_work_dir",
        type=str,
        default="output/train/24gdn/train_24gdn_ib_mapped_block15_lowrank8_step7000_recover",
        help="Mapped-init run directory. The latest model-step-*.safetensors is used by default.",
    )
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help=(
            "Optional repeatable run spec. PATH can be a run directory or a .safetensors file. "
            "If any --run is supplied, --random_work_dir/--mapped_work_dir are ignored."
        ),
    )
    parser.add_argument("--matched_step", type=int, default=None, help="Use model-step-N from every run directory.")
    parser.add_argument("--random_ckpt_step", type=int, default=None, help="Use model-step-N for --random_work_dir.")
    parser.add_argument("--mapped_ckpt_step", type=int, default=None, help="Use model-step-N for --mapped_work_dir.")
    parser.add_argument("--model_dir", type=str, default="model_weight/Wan2.2-TI2V-5B")
    parser.add_argument("--gdn_block_indices", type=str, default=DEFAULT_GDN_BLOCKS)
    parser.add_argument("--gdn_expand_v", type=float, default=1.0)
    parser.add_argument("--gdn_head_dim", type=int, default=0)
    parser.add_argument(
        "--gdn_hidden_size",
        type=int,
        default=int(os.environ.get("GDN_HIDDEN_SIZE", 0)),
        help=(
            "Internal projected GDN hidden size; non-positive keeps the Wan "
            "block width. Exact-large Wan2.1 mappings use 3072."
        ),
    )
    parser.add_argument(
        "--gdn_num_heads",
        type=int,
        default=int(os.environ.get("GDN_NUM_HEADS", 0)),
        help=(
            "Internal projected GDN head count; non-positive keeps the Wan "
            "block head count. Exact-large Wan2.1 mappings use 24."
        ),
    )
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=672)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--prompt", action="append", default=[], help="Prompt to generate. Can be repeated.")
    parser.add_argument("--prompt_file", type=str, default="", help="Text file with one prompt per non-empty line.")
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=(
            "overexposed, static, blurry, low quality, jpeg artifacts, deformed, "
            "extra fingers, bad hands, bad face, messy background, jitter, frame skipping"
        ),
    )
    parser.add_argument("--out_dir", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument(
        "--memory_mode",
        choices=("auto", "resident", "cpu_offload"),
        default="auto",
        help=(
            "Model placement mode. auto keeps all models on GPU when enough VRAM is free, "
            "otherwise it moves T5, DiT, and VAE between CPU and GPU by inference phase."
        ),
    )
    parser.add_argument(
        "--resident_min_free_gb",
        type=float,
        default=30.0,
        help="In auto mode, use resident mode only when at least this much GPU memory is free.",
    )
    parser.add_argument("--disable_tiled", action="store_true", help="Do not pass tiled=True to the pipeline.")
    parser.add_argument("--make_side_by_side", action="store_true", help="Use ffmpeg to hstack matched videos.")
    parser.add_argument("--dry_run", action="store_true", help="Resolve checkpoints and print the run plan, then exit.")
    return parser.parse_args()


def parse_run_specs(args):
    specs = []
    if args.run:
        for spec in args.run:
            if "=" not in spec:
                raise ValueError(f"--run must be LABEL=PATH, got: {spec}")
            label, path = spec.split("=", 1)
            label = label.strip()
            if not label:
                raise ValueError(f"--run label is empty: {spec}")
            specs.append((label, path, args.matched_step))
        return specs

    random_step = args.matched_step if args.matched_step is not None else args.random_ckpt_step
    mapped_step = args.matched_step if args.matched_step is not None else args.mapped_ckpt_step
    return [
        ("random", args.random_work_dir, random_step),
        ("mapped", args.mapped_work_dir, mapped_step),
    ]


def default_prompts():
    return [
        "A cinematic tracking shot of a vintage motorcycle riding along a coastal road at sunset.",
        "A person in a long coat walking through a neon-lit rainy street at night, slow camera push-in.",
        "Morning mist flowing through a pine forest in a mountain valley, sunlight breaking through the trees.",
    ]


def load_prompts(args):
    prompts = list(args.prompt)
    if args.prompt_file:
        prompt_path = Path(project_path(args.prompt_file))
        with prompt_path.open("r", encoding="utf-8") as f:
            prompts.extend(line.strip() for line in f if line.strip())
    if not prompts:
        prompts = default_prompts()
    return prompts[: args.num_samples]


def main():
    args = parse_args()
    prompts = load_prompts(args)
    if args.softmax_only:
        resolved_runs = [
            {
                "label": "softmax_base",
                "input": project_path(args.model_dir),
                "ckpt_path": None,
                "ckpt_step_arg": None,
                "attention": "softmax",
            }
        ]
    else:
        run_specs = parse_run_specs(args)
        resolved_runs = [
            {
                "label": label,
                "input": path,
                "ckpt_path": resolve_checkpoint(path, step),
                "ckpt_step_arg": step,
                "attention": "gdn",
            }
            for label, path, step in run_specs
        ]
    duplicate_ckpts = {}
    for run in resolved_runs:
        if run["ckpt_path"] is not None:
            duplicate_ckpts.setdefault(run["ckpt_path"], []).append(run["label"])
    for ckpt_path, labels in duplicate_ckpts.items():
        if len(labels) > 1:
            print(f"[WARN] Multiple labels resolve to the same checkpoint {ckpt_path}: {labels}")

    default_output_prefix = "infer_softmax_base_text_only_" if args.softmax_only else "infer_24gdn_compare_text_only_"
    default_output_category = "base" if args.softmax_only else "24gdn"
    out_dir = Path(project_path(args.out_dir)) if args.out_dir else (
        PROJECT_ROOT
        / "output"
        / "validation"
        / default_output_category
        / (default_output_prefix + datetime.now().strftime("%Y%m%d_%H%M%S"))
    )

    print("[PLAN] runs:")
    for run in resolved_runs:
        source = run["ckpt_path"] if run["ckpt_path"] is not None else run["input"]
        print(f"[PLAN]   {run['label']} ({run['attention']}): {source}")
    print("[PLAN] prompts:")
    for idx, prompt in enumerate(prompts, start=1):
        print(f"[PLAN]   {idx}: seed={args.seed + idx - 1} prompt={prompt[:100]}")
    print(
        f"[PLAN] generation: {args.width}x{args.height}, "
        f"{args.num_frames} frames, {args.steps} steps, cfg={args.cfg_scale}"
    )
    if not args.softmax_only:
        print(
            f"[PLAN] GDN: blocks={args.gdn_block_indices} "
            f"expand_v={args.gdn_expand_v} head_dim={args.gdn_head_dim} "
            f"hidden_size={args.gdn_hidden_size} num_heads={args.gdn_num_heads}"
        )
    print(
        f"[PLAN] memory_mode: {args.memory_mode}"
        + (
            f" (resident when free VRAM >= {args.resident_min_free_gb:.2f} GiB)"
            if args.memory_mode == "auto"
            else ""
        )
    )
    print(f"[PLAN] output: {out_dir}")
    if args.dry_run:
        return

    import_runtime_deps()
    dtype = torch_dtype_from_arg(args.dtype)
    block_indices = parse_block_indices(args.gdn_block_indices)
    memory_mode, initial_cuda_memory = select_memory_mode(args)
    pipe = build_pipeline(args, block_indices, memory_mode, replace_gdn=not args.softmax_only)
    maybe_force_fp32_time_mlp(pipe)

    out_dir.mkdir(parents=True, exist_ok=True)
    generated = {}
    with core.torch.inference_mode():
        for run in resolved_runs:
            label = run["label"]
            run_out_dir = out_dir / label
            run_out_dir.mkdir(parents=True, exist_ok=True)
            if run["ckpt_path"] is not None:
                load_checkpoint_into_pipe(pipe, run["ckpt_path"], block_indices, dtype)
            else:
                print("[CKPT] Using original base DiT weights; no training checkpoint loaded.")
            generated[label] = []
            for idx, prompt in enumerate(prompts, start=1):
                seed = args.seed + idx - 1
                print(f"[{label}] [{idx}/{len(prompts)}] seed={seed} prompt={prompt[:100]}")
                output = pipe(
                    prompt=prompt,
                    negative_prompt=args.negative_prompt,
                    seed=seed,
                    tiled=not args.disable_tiled,
                    height=args.height,
                    width=args.width,
                    num_frames=args.num_frames,
                    num_inference_steps=args.steps,
                    cfg_scale=args.cfg_scale,
                )
                video = extract_video(output)
                save_path = run_out_dir / f"sample_{idx:02d}_seed{seed}.mp4"
                save_video_file(video, str(save_path), fps=args.fps)
                generated[label].append(str(save_path))
                print(f"[OK] saved: {save_path}")
                if core.torch.cuda.is_available():
                    core.torch.cuda.empty_cache()

    side_by_side = []
    if args.make_side_by_side and len(resolved_runs) == 2:
        left_label = resolved_runs[0]["label"]
        right_label = resolved_runs[1]["label"]
        compare_dir = out_dir / "side_by_side"
        compare_dir.mkdir(exist_ok=True)
        for idx in range(len(prompts)):
            out_path = compare_dir / f"sample_{idx + 1:02d}_{left_label}_vs_{right_label}.mp4"
            ok = make_side_by_side(generated[left_label][idx], generated[right_label][idx], str(out_path))
            if ok:
                side_by_side.append(str(out_path))
                print(f"[OK] side-by-side: {out_path}")

    meta = {
        "mode": "text_only_softmax_base" if args.softmax_only else "text_only_24gdn_compare",
        "runs": resolved_runs,
        "gdn_block_indices": None if args.softmax_only else block_indices,
        "gdn_expand_v": args.gdn_expand_v,
        "gdn_head_dim": args.gdn_head_dim,
        "gdn_hidden_size": args.gdn_hidden_size,
        "gdn_num_heads": args.gdn_num_heads,
        "memory_mode_requested": args.memory_mode,
        "memory_mode_selected": memory_mode,
        "initial_cuda_memory": initial_cuda_memory,
        "model_dir": project_path(args.model_dir),
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "num_inference_steps": args.steps,
        "cfg_scale": args.cfg_scale,
        "seed": args.seed,
        "fps": args.fps,
        "prompts": prompts,
        "negative_prompt": args.negative_prompt,
        "generated": generated,
        "side_by_side": side_by_side,
    }
    with (out_dir / "inference_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[DONE] output dir: {out_dir}")


if __name__ == "__main__":
    main()
