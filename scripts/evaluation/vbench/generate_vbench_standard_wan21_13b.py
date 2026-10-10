"""Generate the official VBench prompt suite with a Wan2.1-1.3B GDN checkpoint.

The saved filename always contains the original VBench prompt, while inference
uses VBench's published Wan2.1 enhanced prompt at the matching metadata index.
That distinction is required for VBench's ``vbench_standard`` matcher."""
from __future__ import annotations
import sys
from pathlib import Path

# Support running this entry point directly from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import argparse
import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from linmorph.vbench import DEFAULT_MODEL_DIR, DEFAULT_RUN_DIR, DEFAULT_OUTPUT, DEFAULT_FULL_INFO, DEFAULT_AUGMENTED_PROMPTS, DEFAULT_UNIPC_SOURCE, DEFAULT_UNIPC_DEPS, ALL_DIMENSIONS, DEFAULT_GDN_BLOCKS, DEFAULT_NEGATIVE_PROMPT, resolve, write_json, load_prompt_records, select_diagnostic_records, write_subset_full_info, build_tasks, load_unipc_class, DiffSynthUniPCAdapter, validate_official_sampling_args

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Generate correctly named videos for official VBench standard evaluation.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help="A checkpoint file or directory; a directory resolves to its latest checkpoint.",
    )
    parser.add_argument(
        "--parent-checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional full parent checkpoint loaded before --checkpoint. Use this when "
            "the child checkpoint only stores parameters trainable in a later stage, "
            "such as a mixed full-GDN/LoRA checkpoint initialized from Stage 2."
        ),
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--full-info", type=Path, default=DEFAULT_FULL_INFO)
    parser.add_argument("--augmented-prompts", type=Path, default=DEFAULT_AUGMENTED_PROMPTS)
    parser.add_argument("--unipc-source", type=Path, default=DEFAULT_UNIPC_SOURCE)
    parser.add_argument("--unipc-deps", type=Path, default=DEFAULT_UNIPC_DEPS)
    parser.add_argument(
        "--prompt-source",
        choices=("wan-enhanced", "original"),
        default="wan-enhanced",
    )
    parser.add_argument("--dimensions", nargs="+", choices=ALL_DIMENSIONS, default=list(ALL_DIMENSIONS))
    parser.add_argument(
        "--prompts-per-dimension",
        type=int,
        default=0,
        help=(
            "Enable a deterministic official-prompt diagnostic subset. A positive value "
            "greedily selects that many official prompts per requested dimension; zero uses "
            "the complete official suite."
        ),
    )
    parser.add_argument(
        "--diagnostic-prompt-count",
        type=int,
        default=0,
        help=(
            "For a diagnostic subset, pad the dimension-covering selection with "
            "additional official prompts until it contains exactly this many unique "
            "prompts. Zero keeps the minimum covering selection."
        ),
    )
    parser.add_argument("--samples-per-prompt", type=int, default=5)
    parser.add_argument(
        "--temporal-flicker-candidates",
        type=int,
        default=25,
        help="Generate this many candidates for temporal-flickering prompts before VBench static filtering.",
    )
    parser.add_argument(
        "--only-prompts-file",
        type=Path,
        default=None,
        help="Generate only the official prompt_en names listed one per line; keeps the original enhanced prompts.",
    )
    parser.add_argument(
        "--min-sample-index",
        type=int,
        default=0,
        help="Generate only candidates at or above this original sample index.",
    )
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=0,
        help="Limit this shard for a smoke test. Any positive value makes the output incomplete.",
    )
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=6.0)
    parser.add_argument("--flow-shift", type=float, default=8.0)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--gdn-block-indices", default=DEFAULT_GDN_BLOCKS)
    parser.add_argument("--gdn-expand-v", type=float, default=2.0)
    parser.add_argument("--gdn-head-dim", type=int, default=256)
    parser.add_argument("--gdn-hidden-size", type=int, default=3072)
    parser.add_argument("--gdn-num-heads", type=int, default=24)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument(
        "--memory-mode", choices=("auto", "resident", "cpu_offload"), default="auto"
    )
    parser.add_argument("--resident-min-free-gb", type=float, default=30.0)
    parser.add_argument("--disable-tiled", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_official_sampling_args(args)
    output = resolve(args.output)
    records = load_prompt_records(args)
    records = select_diagnostic_records(
        records,
        args.dimensions,
        args.prompts_per_dimension,
        args.diagnostic_prompt_count,
    )
    if args.only_prompts_file is not None:
        prompt_file = resolve(args.only_prompts_file)
        requested_prompts = [
            line for line in prompt_file.read_text(encoding="utf-8").splitlines() if line
        ]
        if not requested_prompts or len(requested_prompts) != len(set(requested_prompts)):
            raise ValueError("--only-prompts-file must contain distinct nonempty prompt names")
        available = {record.original_prompt for record in records}
        unknown = set(requested_prompts) - available
        if unknown:
            raise ValueError(f"Prompts are not in the selected official metadata: {sorted(unknown)}")
        requested = set(requested_prompts)
        records = [record for record in records if record.original_prompt in requested]
    tasks = build_tasks(args, records, output)
    diagnostic_subset = args.prompts_per_dimension > 0
    targeted_recovery = args.only_prompts_file is not None or args.min_sample_index > 0

    checkpoint_input = resolve(args.checkpoint)
    parent_checkpoint_input = (
        resolve(args.parent_checkpoint) if args.parent_checkpoint is not None else None
    )
    protocol_label = (
        "targeted official-prompt candidate recovery" if targeted_recovery
        else "official-prompt diagnostic subset" if diagnostic_subset
        else "complete official suite"
    )
    print(f"[PLAN] protocol: {protocol_label}")
    print(f"[PLAN] selected official prompts: {len(records)} unique")
    print(f"[PLAN] dimensions: {', '.join(args.dimensions)}")
    print(
        f"[PLAN] shard: {args.shard_index}/{args.num_shards}; tasks in this invocation: {len(tasks)}"
    )
    print(
        f"[PLAN] sampling: {args.width}x{args.height}, frames={args.num_frames}, fps={args.fps}, "
        f"steps={args.steps}, cfg={args.cfg_scale}, UniPC flow_shift={args.flow_shift}"
    )
    print(f"[PLAN] checkpoint input: {checkpoint_input}")
    if parent_checkpoint_input is not None:
        print(f"[PLAN] parent checkpoint input: {parent_checkpoint_input}")
    print(f"[PLAN] output: {output}")
    if tasks:
        print(f"[PLAN] first task: sample={tasks[0].sample_index} seed={tasks[0].seed} prompt={tasks[0].original_prompt}")

    manifest = {
        "protocol": (
            f"VBench targeted candidate recovery with {args.prompt_source} prompts"
            if targeted_recovery
            else f"VBench official-prompt subset diagnostic with {args.prompt_source} prompts"
            if diagnostic_subset
            else f"VBench standard v1 with {args.prompt_source} prompts"
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "full_protocol_plan": not diagnostic_subset and not targeted_recovery and args.max_tasks == 0,
        "targeted_recovery": targeted_recovery,
        "only_prompts_file": str(resolve(args.only_prompts_file)) if args.only_prompts_file else None,
        "min_sample_index": args.min_sample_index,
        "diagnostic_subset": diagnostic_subset,
        "prompts_per_dimension": args.prompts_per_dimension,
        "diagnostic_prompt_count": args.diagnostic_prompt_count,
        "output_verified_complete": False,
        "prompt_source": args.prompt_source,
        "record_count": len(records),
        "task_count_this_shard": len(tasks),
        "dimensions": args.dimensions,
        "samples_per_prompt": args.samples_per_prompt,
        "temporal_flicker_candidates": args.temporal_flicker_candidates,
        "static_filter_required_for_official_score": "temporal_flickering" in args.dimensions,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "max_tasks": args.max_tasks,
        "checkpoint_input": str(checkpoint_input),
        "parent_checkpoint_input": (
            str(parent_checkpoint_input) if parent_checkpoint_input is not None else None
        ),
        "model_dir": str(resolve(args.model_dir)),
        "sampling": {
            "scheduler": "Wan official FlowUniPCMultistepScheduler",
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "fps": args.fps,
            "steps": args.steps,
            "cfg_scale": args.cfg_scale,
            "flow_shift": args.flow_shift,
            "base_seed": args.base_seed,
            "negative_prompt": args.negative_prompt,
        },
        "tasks": [asdict(task) for task in tasks],
    }
    output.mkdir(parents=True, exist_ok=True)
    if diagnostic_subset:
        subset_full_info = output.parent / "official_subset_full_info.json"
        write_subset_full_info(resolve(args.full_info), records, subset_full_info)
        manifest["subset_full_info"] = str(subset_full_info)
        manifest["selected_prompts"] = [
            {
                "prompt": record.original_prompt,
                "dimensions": [
                    dimension
                    for dimension in args.dimensions
                    if dimension in record.dimensions
                ],
                "metadata_indices": list(record.metadata_indices),
            }
            for record in records
        ]
        print(f"[PLAN] subset metadata: {subset_full_info}")
    manifest_path = output.parent / f"generation_manifest_shard{args.shard_index:02d}.json"
    write_json(manifest_path, manifest)
    print(f"[PLAN] manifest: {manifest_path}")
    if args.dry_run or not tasks:
        return

    if str(args.device).startswith("cuda:"):
        import torch
        torch.cuda.set_device(torch.device(args.device))
        print(f"[GPU] selected device={args.device}", flush=True)
    from linmorph import inference as core

    scheduler_class = load_unipc_class(resolve(args.unipc_source), resolve(args.unipc_deps))
    core.import_runtime_deps()
    checkpoint = core.resolve_checkpoint(str(checkpoint_input))
    parent_checkpoint = (
        core.resolve_checkpoint(str(parent_checkpoint_input))
        if parent_checkpoint_input is not None
        else None
    )
    block_indices = core.parse_block_indices(args.gdn_block_indices)
    memory_mode, memory_info = core.select_memory_mode(args)
    pipe = core.build_pipeline(args, block_indices, memory_mode, replace_gdn=True)
    core.maybe_force_fp32_time_mlp(pipe)
    if parent_checkpoint is not None:
        print(f"[CKPT] Loading parent checkpoint before child: {parent_checkpoint}")
        core.load_checkpoint_into_pipe(
            pipe,
            parent_checkpoint,
            block_indices,
            core.torch_dtype_from_arg(args.dtype),
        )
    core.load_checkpoint_into_pipe(pipe, checkpoint, block_indices, core.torch_dtype_from_arg(args.dtype))
    pipe.scheduler = DiffSynthUniPCAdapter(scheduler_class)
    pipe.eval()

    manifest["resolved_checkpoint"] = checkpoint
    manifest["resolved_parent_checkpoint"] = parent_checkpoint
    manifest["memory_mode_selected"] = memory_mode
    manifest["initial_cuda_memory"] = memory_info
    write_json(manifest_path, manifest)

    journal_path = output.parent / f"generation_journal_shard{args.shard_index:02d}.jsonl"
    generated = 0
    skipped = 0
    started = time.monotonic()
    with core.torch.inference_mode():
        for index, task in enumerate(tasks, start=1):
            save_path = Path(task.output_path)
            if save_path.exists() and not args.overwrite:
                print(f"[SKIP {index}/{len(tasks)}] {save_path.name}")
                skipped += 1
                continue
            print(
                f"[RUN {index}/{len(tasks)}] sample={task.sample_index} seed={task.seed} "
                f"prompt={task.original_prompt[:100]}"
            )
            task_started = time.monotonic()
            output_value = pipe(
                prompt=task.inference_prompt,
                negative_prompt=args.negative_prompt,
                seed=task.seed,
                tiled=not args.disable_tiled,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                num_inference_steps=args.steps,
                cfg_scale=args.cfg_scale,
                sigma_shift=args.flow_shift,
            )
            video = core.extract_video(output_value)
            core.save_video_file(video, str(save_path), fps=args.fps)
            elapsed = time.monotonic() - task_started
            event = {
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": elapsed,
                **asdict(task),
            }
            with journal_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
            generated += 1
            print(f"[OK {index}/{len(tasks)}] {save_path} ({elapsed:.1f}s)")
            if core.torch.cuda.is_available():
                core.torch.cuda.empty_cache()

    total_elapsed = time.monotonic() - started
    print(
        f"[DONE] generated={generated}, skipped={skipped}, elapsed={total_elapsed / 60:.1f} min, "
        f"output={output}"
    )


if __name__ == "__main__":
    main()
