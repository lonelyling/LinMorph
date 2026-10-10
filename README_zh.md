# LinMorph

**Learning to Morph Weights for Efficient Video DiT Linearization**

[English](README.md) · [安装](docs/installation.md) · [训练](docs/training.md) · [评测](docs/evaluation.md) · [权重发布](docs/models.md)

本目录是为 GitHub 整理的独立代码副本。原有 `DiffSynth-Studio`、`from_congyue`、实验日志、数据和权重均保留在原处，没有移动或删除。

方法包含：共享权重映射 → GDN 参数物化 → 7/14/21 层渐进蒸馏 → 全部 hybrid DiT 的无教师微调。为了兼容已有权重，保留 `diffsynth` 模块名及训练文件中的参数命名。

## 已整理内容

- 主线：Wan2.1-T2V-1.3B 的 14-GDN 和 21-GDN；另含 Wan2.2 扩展工具。
- `scripts/train.py`：读取配置，以本地 CSV 数据启动训练，支持 `--dry-run`。
- `scripts/inference.py`：单视频生成，包含主表模型层配置及原始 Wan 对照。
- 权重映射代码已从外部目录收进本仓库，修正物化脚本的默认路径。
- 依赖说明、英文 README、配置、来源校验清单、评测说明、模型发布说明和 `.gitignore`。
- 副本移除了数据加载文件内带私有存储凭据的测试示例，不带数据、权重、缓存或环境目录。

## 脚本入口

`scripts/` 顶层仅保留 `train.py` 和 `inference.py`。其余按初始化、评测、分析、消融和测速分目录；发布检查位于 `tools/`，共用实现位于 `linmorph/`。完整文件说明见 [脚本导航](scripts/README.md)。

## 权重和开源状态

主表对应候选权重是 **14-GDN microstep 8000** 和 **21-GDN microstep 40000**。先核对最终论文，再把这两个推理权重分别放到 Hugging Face；GitHub 放代码、配置和下载说明。映射器权重可单独提供用于复现方法，无需上传每个训练中间 checkpoint 或优化器状态。

目前没有执行 GitHub/Hugging Face 上传，没有虚构远程地址。作者信息、论文链接、映射器代码授权与公开模型地址仍需发布者确认；详见 [发布清单](docs/release_checklist.md)。验证范围记录在 [整理验证报告](docs/validation.md)，不等同于重新完成训练或全量 GPU 评测。
