# 脚本导航 / Script guide

从仓库根目录运行下列命令。训练和单视频生成从顶层两个入口开始；模型加载、采样器及其他共用实现集中在 `linmorph/`，无需直接执行。

```text
scripts/
├── train.py
├── inference.py
├── initialization/
├── evaluation/
│   ├── vbench/
│   └── vbench2/
├── analysis/
├── ablations/
└── benchmarks/
tools/
└── check_release.py
```

## 常用入口

| 文件 | 用途 |
|---|---|
| [train.py](train.py) | 读取训练配置，启动映射拟合、渐进蒸馏或微调 |
| [inference.py](inference.py) | 生成单个视频，支持 LinMorph 和原始 Wan 对照 |

参数用 `python scripts/train.py --help`、`python scripts/inference.py --help` 查看。完整流程见 [训练说明](../docs/training.md) 和 [安装说明](../docs/installation.md)。

## 初始化权重

| 文件 | 用途 |
|---|---|
| [materialize_wan21_13b_fullmix_gdn_init.py](initialization/materialize_wan21_13b_fullmix_gdn_init.py) | 从学到的 FullMix 映射生成 GDN 初始化权重 |
| [merge_staged_gdn_init.py](initialization/merge_staged_gdn_init.py) | 组合新增层初始化与已有训练层，衔接 7→14→21 层恢复 |

## 评测

| 文件 | 用途 |
|---|---|
| [setup_vbench_standard.sh](evaluation/vbench/setup_vbench_standard.sh) | 准备并校验 Wan UniPC 和增强提示词 |
| [generate_vbench_standard_wan21_13b.py](evaluation/vbench/generate_vbench_standard_wan21_13b.py) | 生成 Wan2.1 的 VBench 视频 |
| [generate_vbench_standard_wan22_5b.py](evaluation/vbench/generate_vbench_standard_wan22_5b.py) | 生成 Wan2.2 的 VBench 视频 |
| [evaluate_vbench_standard.py](evaluation/vbench/evaluate_vbench_standard.py) | 校验覆盖率并运行 VBench 评分 |
| [generate_vbench2_wan21.py](evaluation/vbench2/generate_vbench2_wan21.py) | 生成 VBench-2.0 视频 |
| [evaluate_vbench2_wan21.py](evaluation/vbench2/evaluate_vbench2_wan21.py) | 检查环境并运行 VBench-2.0 评分 |
| [report_vbench2_wan21.py](evaluation/vbench2/report_vbench2_wan21.py) | 汇总 VBench-2.0 评分和完成情况 |

这些入口仍需外部评测模型和数据，详见 [评测说明](../docs/evaluation.md)。

## 分析、消融与测速

| 文件 | 用途 |
|---|---|
| [analyze_linemorph_learned_mapping.py](analysis/analyze_linemorph_learned_mapping.py) | 论文中的映射学习分析 |
| [summarize_wan21_1_3b_weight_mapping_eval.py](analysis/summarize_wan21_1_3b_weight_mapping_eval.py) | 汇总不同映射方案的评测结果 |
| [materialize_wan21_13b_manual_gdn_init.py](ablations/materialize_wan21_13b_manual_gdn_init.py) | Wan2.1 手工复制初始化基线 |
| [materialize_wan22_5b_manual_gdn_init.py](ablations/materialize_wan22_5b_manual_gdn_init.py) | Wan2.2 手工复制初始化基线 |
| [infer_24gdn_compare_text_only.py](ablations/infer_24gdn_compare_text_only.py) | 多模型/初始化方案的匹配提示词对比；共用实现已抽离 |
| [prepare_wan22_5b_quick_eval.py](ablations/prepare_wan22_5b_quick_eval.py) | Wan2.2 小规模诊断；共用加载代码已抽离 |
| [benchmark_wan22_5b.py](benchmarks/benchmark_wan22_5b.py) | A800 上的前向、去噪、端到端延迟测试 |

发布前运行 `python tools/check_release.py`。从旧路径迁移的对应关系见 [路径映射表](../docs/script_migration.json)。
