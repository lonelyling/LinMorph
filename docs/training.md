# Training

Run from the checkout root after installation. The launcher uses the preserved research implementations and supplies explicit paths. Add `--dry-run` to inspect commands without loading GPU dependencies; extra original arguments may follow `--`.

## Data

Use local videos and a UTF-8 CSV with `video,caption` columns; video paths must be relative to the dataset root. See `examples/data/metadata.csv`. The example references illustrative files, not bundled videos. The private QWorld corpus is not included. Substituting local data is an adaptation, not a claim of identical paper reproduction.

```bash
export MODEL_DIR="$PWD/model_weight/Wan2.1-T2V-1.3B"
export DATA_ROOT="/path/to/your/videos"
export METADATA="/path/to/your/metadata.csv"
```

## Mapping and materialization

```bash
python scripts/train.py --config configs/training/mapping_base.json --model-dir "$MODEL_DIR" --data-root "$DATA_ROOT" --metadata "$METADATA" --output output/mapping/base
python scripts/train.py --config configs/training/mapping_residual.json --model-dir "$MODEL_DIR" --data-root "$DATA_ROOT" --metadata "$METADATA" --init-checkpoint output/mapping/base/model-final.safetensors --output output/mapping/residual
python scripts/initialization/materialize_wan21_13b_fullmix_gdn_init.py --teacher-checkpoint "$MODEL_DIR/diffusion_pytorch_model.safetensors" --mapping-checkpoint output/mapping/residual/model-final.safetensors --output-path output/init/gdn_bank.safetensors
```

Base FullMix fits a shared mapper on all 30 blocks using 17 frames at 256×448, up to 8,192 examples, learning rate 1e-4, two epochs and accumulation 4. The residual stage freezes the base and trains native PostFullResidual at 1e-5. The 3072-hidden/24-head internal GDN and boundary adapters are intentional for the 1536-hidden Wan2.1 backbone. Materialization produces the 21-block initialization bank; it is not a complete inference model.

## Progressive distillation

```bash
B7=0,4,8,12,16,20,24
B14=0,1,4,5,8,9,12,13,16,17,20,21,24,25
B21=0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22,24,25,26
python scripts/initialization/merge_staged_gdn_init.py --base-init output/init/gdn_bank.safetensors --target-blocks "$B7" --output-path output/init/7gdn.safetensors
python scripts/train.py --config configs/training/distill_7gdn.json --model-dir "$MODEL_DIR" --data-root "$DATA_ROOT" --metadata "$METADATA" --init-checkpoint output/init/7gdn.safetensors --num-processes 8 --output output/kd7
python scripts/initialization/merge_staged_gdn_init.py --base-init output/init/gdn_bank.safetensors --target-blocks "$B14" --overlay-checkpoint output/kd7/model-final.safetensors --overlay-blocks "$B7" --output-path output/init/14gdn.safetensors
python scripts/train.py --config configs/training/distill_14gdn.json --model-dir "$MODEL_DIR" --data-root "$DATA_ROOT" --metadata "$METADATA" --init-checkpoint output/init/14gdn.safetensors --num-processes 8 --output output/kd14
python scripts/initialization/merge_staged_gdn_init.py --base-init output/init/gdn_bank.safetensors --target-blocks "$B21" --overlay-checkpoint output/kd14/model-final.safetensors --overlay-blocks "$B14" --output-path output/init/21gdn.safetensors
python scripts/train.py --config configs/training/distill_21gdn.json --model-dir "$MODEL_DIR" --data-root "$DATA_ROOT" --metadata "$METADATA" --init-checkpoint output/init/21gdn.safetensors --num-processes 8 --output output/kd21
```

Each recipe uses L1 prediction KD at 5e-6, 1,500 optimizer updates, 500 warmup updates, 81 frames at 480×832 and accumulation 2. Eight processes give effective batch 16. Previously recovered blocks are retained when expanding the converted set. Configure Accelerate/DeepSpeed explicitly if full DDP replicas do not fit.

## Teacher-free recovery

```bash
python scripts/train.py --config configs/training/finetune_14gdn.json --model-dir "$MODEL_DIR" --data-root "$DATA_ROOT" --metadata "$METADATA" --init-checkpoint output/kd14/model-final.safetensors --num-processes 8 --output output/finetune14
python scripts/train.py --config configs/training/finetune_21gdn.json --model-dir "$MODEL_DIR" --data-root "$DATA_ROOT" --metadata "$METADATA" --init-checkpoint output/kd21/model-final.safetensors --num-processes 8 --output output/finetune21
```

Full-DiT flow matching uses 1e-5 and up to 4,000/20,000 optimizer updates, corresponding to microsteps 8,000/40,000 at accumulation 2. Matching update counts alone does not reproduce the original checkpoint: data, ordering, preceding stages and optimizer history matter.

Use fresh output directories. Initialization with `--init-checkpoint` is not full-state resume. For interrupted training, use the original entry point with `--resume` and restore optimizer, scheduler, RNG and data position using the original Accelerate configuration.
