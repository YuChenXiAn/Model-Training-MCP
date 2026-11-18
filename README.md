# Model-Training-MCP

从训练数据到 EV-SDK 业务算法的可复用工程：**模型训练 → 导出 ONNX → 业务提示词改造 → EV-SDK 集成 → 自动化测试**。

> 状态（2026-11-18）：已根据上传的训练模板、老鼠入侵模板、人员徘徊技术文档及模型骨架包组织方案。**尚未完成真实训练/EV-SDK 设备实测**：当前执行环境没有 `/home/data/` 数据集、NVIDIA GPU 和 EV-SDK 工程；不得把本地静态测试当作训练通过或 SDK 验收。

## 目录

- `skills/model-training-mcp/SKILL.md`：统一 Skill，区分检测训练、分割训练、ONNX 导出及业务应用开发。
- `prompts/loitering-evsdk.md`：按老鼠入侵提示词结构重构的人员徘徊 EV-SDK 开发任务规格。
- `prompts/prompt-rewrite.md`：输入模型信息、业务算法技术文档及工程限制时的提示词改造任务。
- `training/train.py`：YOLO26x/YOLO26x-seg 单任务训练与模型验证；独立于 EV-SDK 开发。
- `training/prepare.py`：LabelMe 多边形 → YOLO Seg / Pascal VOC XML → YOLO Detect；数据校验、确定性划分。
- `training/export_onnx.py`：从 best.pt 导出 best.onnx；不假定输出算子、解码规则与 YOLOv8 完全一致。
- `training/preflight.py`：训练前校验：数据集、权重、依赖、CUDA 与输出路径。
- `tests/test_prepare.py`：无需 GPU 的数据准备单元测试。
- `references/`：用户提供的原始 Markdown 来源；原始方案的冲突和缺陷不代表新代码约束。
- `assets/README.md`：模型骨架包与权重清单及大文件管理说明。

## 最短运行流程

```bash
python -m pip install 'ultralytics==8.4.78' pyyaml pillow pytest
python training/prepare.py --task segment --source /home/data --output /home/data/yolo_seg_dataset --classes person
python training/preflight.py --task segment --data /home/data/yolo_seg_dataset/dataset.yaml --weights /project/train/src_repo/yolo26x-seg.pt
python training/train.py --task segment --data /home/data/yolo_seg_dataset/dataset.yaml --weights /project/train/src_repo/yolo26x-seg.pt --epochs 150 --device 0
python training/export_onnx.py --weights runs/segment/experiment/weights/best.pt
python -m pytest -q tests/
```

检测任务把 `--task segment` 改为 `detect`，并提供检测权重 `yolo26x.pt`。实际训练使用与标注类别严格匹配的 YAML；**不得把 26 类骨架配置直接用于单类 person 模型**。

## 重要边界

1. `best.onnx` 仅用于 EV-SDK 推理，不是训练输入；训练输入是 `.pt` 权重与标注数据。
2. 只修改 `ev_sdk/config` 和 `ev_sdk/src` 的限制仅作用于 **EV-SDK 业务工程**；训练项目可创建 `training/`、`tests/`、`runs/` 等目录。
3. `YOLO26` 权重的 ONNX 输出可能不同于传统 YOLOv8。业务侧若要走“YOLOv8 前向推理”，必须验证输入预处理、输出形状、坐标解码、NMS 与目标类别映射；不一致时禁止冒充适配成功。
4. 来源骨架包包含大于 GitHub 常规 100 MB 单文件限制的权重，未作为普通 Git 对象上传。通过 Git LFS 或部署机器上的本地路径提供。
5. 不把没有数据的 dry-run、合成单测视为真实训练指标。

## 复用示例

输入：`best.onnx`、标签 `0: person`、YOLOv8 前向推理、修改范围仅 `ev_sdk/config` 与 `ev_sdk/src`、测试图位于 `/home/data/`、算法原文 `references/person-loitering-original.md`。

使用 `prompts/prompt-rewrite.md` 改写后，使用 `prompts/loitering-evsdk.md` 交给有真实 EV-SDK 骨架的代码代理执行；另在有 CUDA 与数据的机器运行训练 Skill。

## 验收

- 可检查：提示词与原始技术文档字段一致性；数据预处理格式；训练命令与版本；单元测试。
- 须在用户环境执行：GPU 训练日志、best.pt、mAP/IoU、ONNX 运行对齐、EV-SDK 的构建与逐案例结果。

