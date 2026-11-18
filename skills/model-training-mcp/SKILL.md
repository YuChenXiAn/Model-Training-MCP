# Model Training MCP Skill

## 目标与适用系统
将业务文档和已有模型骨架转换为可复现的训练项目，并把 `best.onnx` 应用到 EV-SDK 业务算法。按两个独立工作流执行：**A. 训练模型** 和 **B. 开发 EV-SDK 应用**。不能混淆检测/分割任务，也不能用 `best.onnx` 作为训练初始权重。

## 输入
- 用户业务文档（例如 `references/person-loitering-original.md`）
- `YOLO26x-Seg实例分割训练skills(2).md`（设计参考，默认 150 epoch，ultralytics 8.4.78）
- EV-SDK 老鼠入侵完整提示词（业务格式参考）
- 骨架包：`prepare_and_train.py`、`run_train.py`、`dataset.yaml`、`yolo26x.pt`、`yolo26x-seg.pt`、`ultralytics-8.4.78.zip`
- /home/data 下图像及标注；模型权重、类别和 GPU 环境
- 业务部署条件：`best.onnx`、label 0: person、YOLOv8 推理适配接口、EV-SDK 工程代码

## 工作流 A：训练
1. 理解任务：bbox 目标检测用 `detect` + `yolo26x.pt`；多边形实例分割用 `segment` + `yolo26x-seg.pt`。先确认有实例分割标注才执行 segment。
2. 检查骨架，保留已有可用工程结构。**不得**把骨架的 26 类列表直接当单类 person 数据集使用。
3. 执行 `python training/prepare.py --task segment --source /home/data --output /home/data/yolo_seg_dataset --classes person`；单类不等于原始 26 类训练任务。
4. 执行 `python training/preflight.py --task segment --data /home/data/yolo_seg_dataset/dataset.yaml --weights /project/train/src_repo/yolo26x-seg.pt`；失败则停止，不允许进入训练。
5. 执行 `python training/train.py --task segment --data ... --weights ... --epochs 150 --device 0`；保留日志和失败原因。
6. 根据 `best.pt` 执行 `training/export_onnx.py`，验证 ONNX 存在、非空及输入输出约定；运行测试图像真实推理并报告结果。
7. 输出训练指标（mAP50-95、精确率、召回率、验证集类别分布等）以及模型路径；没有真实运行不得虚构。

## 工作流 B：提示词改写 + EV-SDK 应用
1. 读取原业务算法技术文档；使用 `prompts/prompt-rewrite.md` 作为通用改写指令。
2. 参考 `prompts/loitering-evsdk.md`：保留原参数及 JSON 协议，明确 ROI、跟踪 ID、计时、人数条件及多帧投票语义。
3. 仅修改目标 EV-SDK 工程的 `ev_sdk/config` 与 `ev_sdk/src`。训练仓库脚本不受此约束。
4. 不凭“YOLOv8 前推理”文字判断 YOLO26 ONNX 已兼容。检查输入张量、预处理、输出张量、是否端到端/NMS 和输出解码。
5. 对真实 /home/data 图像做推理测试；对时序逻辑使用受控帧+时间戳及多 CID 交错测试。区分单元测试/集成测试/实机测试。
6. 若工程、图片、onnx、GPU 或运行环境缺失，标注阻塞，不得报“测试成功”。

## 质量门槛
- 训练开始前：图像标注一一匹配，类别映射整数范围有效，训练/验证文件不重叠；无有效样本必须失败。
- 实例分割标签为 `class x1 y1 x2 y2 x3 y3 ...`，目标检测为 `class cx cy w h`；不允许格式互换。
- 使用固定 random seed；严格校验 labelme 坐标合法性、面积及未知类别，避免静默丢标。
- 仅在现有架构确认支持情况下修改网络结构或注意力模块；不允许臆造模型 YAML。
- 配置和输出 JSON 必须能被标准 JSON 解析；不允许重复 key。
- 结论分级：静态检查通过 / 单测通过 / 真实训练通过 / EV-SDK 集成通过 / 实机验收通过。

## 参考与命令
详见根目录 README、training 下脚本与 prompts 下 Markdown。