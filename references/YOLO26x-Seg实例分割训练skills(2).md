# YOLO26x-Seg 实例分割训练全流程方案

> 框架版本：ultralytics==8.4.78（锁定）  
> 基础权重：yolo26x-seg.pt（预训练）  
> 任务类型：实例分割（task=segment）  
> 训练轮数：150 epochs  
> 本文档包含：工程目录结构、数据预处理脚本、数据集配置、启动脚本、训练主程序、模块选型评估、注意力模块注入样例、可直接复制的完整提示词。

---

## 一、项目基础信息


| 项目     | 内容                                                           |
| ------ | ------------------------------------------------------------ |
| 框架版本   | ultralytics==8.4.78（严格锁定）                                    |
| 基础权重   | yolo26x-seg.pt                                               |
| 数据集目录  | /home/data/                                                  |
| 任务     | 实例分割 Seg 训练，epoch=150                                        |
| 输入尺寸   | imgsz=640（小目标场景可调 800/960）                                   |
| 原始数据   | jpg 图像 + png 掩码 + json 标注（labelme）+ xml 标注（VOC）              |
| 数据对齐规则 | 按同名前缀匹配 jpg ↔ png ↔ json ↔ xml；bbox 以 json 轮廓导出值为基准，xml 交叉校验 |
| 类别     | 1 类：target_obj（按需修改，见注意事项）                                   |


---



## 二、完整工程目录结构

```text
/home/data/
├── 0001.jpg  0001.png  0001.json  0001.xml     # 原始数据四件套（同名匹配）
├── 0002.jpg  0002.png  0002.json  0002.xml
├── ...（共 N 组样本）
│
├── yolo_seg_dataset/                           # data_prepare.py 输出（规范数据集）
│   ├── images/
│   │   ├── train/    *.jpg                     # 训练图像（默认 90%）
│   │   └── val/      *.jpg                     # 验证图像（默认 10%）
│   ├── labels/
│   │   ├── train/    *.txt                     # YOLO Seg 多边形标签（训练真正读取的数据）
│   │   └── val/      *.txt
│   └── masks/
│       ├── train/    *.png                     # 原始掩码（保留作参照/后处理，训练不直接读取）
│       └── val/      *.png
│
├── seg_dataset.yaml                            # ① 数据集配置（train/val 指向）
├── data_prepare.py                             # ② 数据对齐与转换脚本
├── train_start.sh                              # ③ 训练启动脚本
├── train_seg.py                                # ④ 训练主程序
├── models/
│   └── yolo26x-seg-with-attn.yaml              # ⑤ 注意力增强模型配置（可选，默认不启用）
└── yolo26x-seg.pt                              # 基础权重
```

---



## 三、数据集预处理脚本 data_prepare.py



### 3.1 功能与处理流程

1. 遍历 `/home/data/`，按同名前缀匹配 **jpg ↔ png ↔ json ↔ xml** 四件套，缺失配套文件的样本跳过并打印 WARN；
2. 解析 labelme json → 提取**归一化多边形点**（YOLO Seg 标签本体）与轮廓导出 bbox；
3. 解析 VOC xml → 提取归一化 bbox，仅用于**交叉校验**；
4. bbox 对齐校验：图像尺寸、目标数量不一致时打印 WARN；业务规则固定为**以 json 轮廓为基准**，保证 图像 / 掩码 / json 标注 / bbox 四者边界一致；
5. 类别名称 → 整数 ID 映射（YOLO 标签要求整数类别，不能写字符串）；
6. 输出 YOLO Seg 标准标签：每行 `class_id x1 y1 x2 y2 ... xn yn`（归一化多边形）；
7. 复制 jpg → `images/`、png → `masks/`、txt → `labels/`，自动划分 train/val（默认 9:1，固定随机种子可复现）。

> **相对原始脚本的三处修正**（均影响能否正确训练，已在代码中落实）：
>
> 1. **标签格式**：原始脚本把 bbox（cx,cy,w,h）写入 txt，这是检测格式；`task=segment` 必须写入**多边形点**，否则掩码退化为矩形或训练报错；
> 2. **类别 ID**：原始脚本把 json 的类别字符串直接写入 txt，YOLO 要求整数 ID，已增加 CLASS_TO_ID 映射；
> 3. **train/val 划分**：需求要求自动划分，原始脚本未实现且 yaml 中 train/val 指向同一目录，已补充分割并同步修改 yaml。



### 3.2 完整代码

```python
# -*- coding: utf-8 -*-
"""
data_prepare.py — 数据集对齐与转换脚本（YOLO Seg 格式）
功能：
  1. 遍历 /home/data/，按同名前缀匹配 jpg / png / json / xml 四件套
  2. 解析 labelme json 与 VOC xml，做 bbox 对齐校验（以 json 轮廓为基准）
  3. 输出 YOLO Seg 标准标签（txt 归一化多边形），自动划分 train / val
用法：python data_prepare.py
"""
import os
import json
import shutil
import random
from xml.etree import ElementTree as ET

DATA_ROOT = "/home/data"
OUTPUT_ROOT = "/home/data/yolo_seg_dataset"
VAL_RATIO = 0.1                 # 验证集比例（按需调整，如 0.2）
RANDOM_SEED = 42                # 固定随机种子，保证划分可复现

# 类别映射：必须与 seg_dataset.yaml 的 names 一一对应，按需扩展（如 ["defect_a", "defect_b"]）
CLASS_NAMES = ["target_obj"]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}

# 输出目录（train / val 两套）
SPLITS = {
    "train": {
        "images": os.path.join(OUTPUT_ROOT, "images", "train"),
        "labels": os.path.join(OUTPUT_ROOT, "labels", "train"),
        "masks":  os.path.join(OUTPUT_ROOT, "masks", "train"),
    },
    "val": {
        "images": os.path.join(OUTPUT_ROOT, "images", "val"),
        "labels": os.path.join(OUTPUT_ROOT, "labels", "val"),
        "masks":  os.path.join(OUTPUT_ROOT, "masks", "val"),
    },
}
for split in SPLITS.values():
    for d in split.values():
        os.makedirs(d, exist_ok=True)


def parse_xml_get_bbox(xml_path):
    """读取 VOC xml，返回 (w, h, [(cls_name, cx, cy, bw, bh)])，仅用于 bbox 交叉校验"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    w = int(root.find("size/width").text)
    h = int(root.find("size/height").text)
    bboxes = []
    for obj in root.findall("object"):
        cls_name = obj.find("name").text
        bnd = obj.find("bndbox")
        x1 = float(bnd.find("xmin").text)
        y1 = float(bnd.find("ymin").text)
        x2 = float(bnd.find("xmax").text)
        y2 = float(bnd.find("ymax").text)
        cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
        bw, bh = (x2 - x1) / w, (y2 - y1) / h
        bboxes.append([cls_name, cx, cy, bw, bh])
    return w, h, bboxes


def parse_json_get_annotation(json_path):
    """读取 labelme json，返回 (w, h, [{'label','poly','bbox'}])，poly/bbox 均已归一化"""
    with open(json_path, "r", encoding="utf-8") as f:
        anno = json.load(f)
    img_w = anno["imageWidth"]
    img_h = anno["imageHeight"]
    items = []
    for shape in anno["shapes"]:
        pts = shape["points"]
        if len(pts) == 2:  # labelme rectangle 模式 -> 展开为四点多边形
            (x1, y1), (x2, y2) = pts
            pts = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        # 归一化多边形点（YOLO Seg 每行：cls x1 y1 x2 y2 ... xn yn）
        poly = []
        for p in pts:
            poly += [p[0] / img_w, p[1] / img_h]
        poly = [max(0.0, min(1.0, v)) for v in poly]  # 越界坐标裁剪到 [0,1]
        # 由轮廓导出 bbox（业务规则：以 json 为基准）
        bbox = [(min(xs) + max(xs)) / 2 / img_w,
                (min(ys) + max(ys)) / 2 / img_h,
                (max(xs) - min(xs)) / img_w,
                (max(ys) - min(ys)) / img_h]
        items.append({"label": shape["label"], "poly": poly, "bbox": bbox})
    return img_w, img_h, items


def main():
    random.seed(RANDOM_SEED)
    all_jpgs = [f for f in os.listdir(DATA_ROOT) if f.lower().endswith(".jpg")]
    valid = []

    for jpg_name in sorted(all_jpgs):
        base = os.path.splitext(jpg_name)[0]
        jpg_path = os.path.join(DATA_ROOT, jpg_name)
        png_path = os.path.join(DATA_ROOT, f"{base}.png")
        json_path = os.path.join(DATA_ROOT, f"{base}.json")
        xml_path = os.path.join(DATA_ROOT, f"{base}.xml")

        # 1) 四件套完整性校验
        if not all(os.path.exists(p) for p in [png_path, json_path, xml_path]):
            print(f"WARN: {base} 配套文件缺失（需 png/json/xml），跳过")
            continue

        # 2) bbox 对齐校验：json 轮廓为基准，xml 交叉核对
        jw, jh, json_items = parse_json_get_annotation(json_path)
        xw, xh, xml_bboxes = parse_xml_get_bbox(xml_path)
        if (jw, jh) != (xw, xh):
            print(f"WARN: {base} 图像尺寸不一致 json=({jw},{jh}) xml=({xw},{xh})")
        if len(json_items) != len(xml_bboxes):
            print(f"WARN: {base} 目标数不一致 json={len(json_items)} xml={len(xml_bboxes)}，以 json 为准")

        # 业务规则：最终标签以 json 轮廓为准，保证 图像/掩码/标注/bbox 四者边界一致
        valid.append((jpg_name, base, jpg_path, png_path, json_items))

    if not valid:
        print("错误：未找到任何完整的 jpg/png/json/xml 四件套样本，请检查 DATA_ROOT 目录")
        return

    # 3) 划分 train / val
    random.shuffle(valid)
    n_val = max(1, int(len(valid) * VAL_RATIO))
    val_names = {base for _, base, *_ in valid[:n_val]}

    # 4) 输出规范数据集
    for jpg_name, base, jpg_path, png_path, items in valid:
        split = "val" if base in val_names else "train"
        shutil.copy(jpg_path, os.path.join(SPLITS[split]["images"], jpg_name))
        shutil.copy(png_path, os.path.join(SPLITS[split]["masks"], f"{base}.png"))

        txt_out = os.path.join(SPLITS[split]["labels"], f"{base}.txt")
        with open(txt_out, "w", encoding="utf-8") as ft:
            for item in items:
                cls_name = item["label"]
                if cls_name not in CLASS_TO_ID:
                    print(f"WARN: {base} 未注册类别 '{cls_name}'，已跳过该目标（请在 CLASS_NAMES 中补充）")
                    continue
                line = f"{CLASS_TO_ID[cls_name]} " + " ".join(f"{v:.6f}" for v in item["poly"])
                ft.write(line + "\n")

    n_train = len(valid) - n_val
    print(f"数据集对齐&转换完成：共 {len(valid)} 张（train {n_train} / val {n_val}）")
    print(f"输出路径：{OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
```

---



## 四、数据集配置 seg_dataset.yaml

> 与 data_prepare.py 输出的目录结构对应；类别必须与脚本中 CLASS_NAMES 保持一致。

```yaml
# seg_dataset.yaml — 数据集配置（ultralytics 8.4.78）
path: /home/data/yolo_seg_dataset
train: images/train
val: images/val

# 类别：按需修改，且必须与 data_prepare.py 中的 CLASS_NAMES 一一对应
names:
  0: target_obj
```

---



## 五、训练启动脚本 train_start.sh

```bash
#!/bin/bash
# train_start.sh — YOLO26x-seg 分割训练启动脚本（epoch=150）
# 用法：bash train_start.sh
# 说明：负责环境准备 + 启动训练，训练逻辑在 train_seg.py

set -e  # 任一环节失败即终止

# 1. 激活 conda 环境（环境名按实际环境修改）
source /opt/conda/bin/activate yolov8_env

# 2. 锁定 ultralytics 版本（保证与工程一致）
pip install ultralytics==8.4.78

# 3. 启动训练
python train_seg.py
```

---



## 六、训练主程序 train_seg.py

```python
# train_seg.py — YOLO26x-seg 实例分割训练主程序（ultralytics 8.4.78）
# 用法：python train_seg.py（或由 train_start.sh 调用）
from ultralytics import YOLO


def main():
    # 基础方案：直接加载预训练权重训练
    model = YOLO("yolo26x-seg.pt")

    # 可选方案：启用注意力模块时，改为加载自定义模型 yaml 并载入预训练权重
    # model = YOLO("models/yolo26x-seg-with-attn.yaml").load("yolo26x-seg.pt")

    model.train(
        data="/home/data/seg_dataset.yaml",
        epochs=150,             # 固定 150 epochs
        task="segment",         # 实例分割任务
        imgsz=640,              # 小目标场景可提至 800/960（多尺度训练，注意显存）
        batch=8,                # 按显存调整（40G 可尝试 16）
        device=0,               # GPU 编号；多卡可填 "0,1,2,3"
        workers=4,
        amp=True,               # 混合精度；显存不足或数值不稳时关闭
        mosaic=1.0,
        mixup=0.1,
        # ---- 数据增强层：光照增强（推荐开启，零推理成本）----
        hsv_h=0.015,            # 色调扰动
        hsv_s=0.7,              # 饱和度扰动
        hsv_v=0.4,              # 明度扰动（应对明暗不均 / 逆光 / 阴影）
        # 说明：CLAHE / Gamma 需在 ultralytics 增强流程中自定义或经 albumentations 接入，
        #       8.4.78 未内置对应训练参数，需要时见「七、模块选型评估 - 光照模块」。
        # ---- 小目标增强（按目标尺寸决定是否开启，见 7.2）----
        # 多尺度训练：imgsz=800 或 960；copy-paste 增强需自定义 dataloader 接入
        # ---- 网络结构层（需改模型 yaml + 注册模块，见「八」）----
        cfg=None,               # 自定义模块注入入口：加载自定义 yaml 时此处自动生效
        name="yolo26x_seg_run", # 输出目录 runs/segment/yolo26x_seg_run
        pretrained=True,        # 使用预训练权重
        verbose=True,
    )


if __name__ == "__main__":
    main()
```

---



## 七、模块选型评估（光照 / 小目标 / 注意力）



### 7.0 总览对比表


| 模块    | 适用场景                  | 推荐度            | 实现层                                     | 显存 / 推理代价           |
| ----- | --------------------- | -------------- | --------------------------------------- | ------------------- |
| 光照模块  | 明暗不均、逆光、反光、阴影（工业现场常见） | ✅ 推荐开启（数据增强方案） | 数据增强（推荐）／网络端校正（极端场景）                    | 训练期略增计算；推理零成本       |
| 小目标增强 | 目标像素 < 32×32、小目标漏检严重  | ✅ 按目标尺寸决定      | 数据增强 + 训练策略（多尺度 / copy-paste / loss 加权） | imgsz 提升显著增加显存与训练时长 |
| 注意力机制 | 目标特征弱、背景杂乱、工业缺陷       | ⚠️ 按需选择        | 网络结构改造（改模型 yaml + 注册模块）                 | 参数量↑、推理延迟↑、过拟合风险↑   |


> 关键原则：**能靠数据增强解决的不动网络**。增强层零推理成本、可随时开关；网络结构改造不可逆性高、维护成本大，仅在增强手段失效后引入。



### 7.1 光照模块

- **适用场景**：现场图像明暗不均、逆光、反光、阴影，是工业视觉最常见的光照退化形式。
- **推荐度**：✅ 推荐开启。

**方案 A：在线数据增强（推荐，默认）**

- 手段：CLAHE 自适应直方图均衡、随机亮度 / 对比度扰动、gamma 校正；ultralytics 自带 HSV 扰动（`hsv_h / hsv_s / hsv_v`），CLAHE/Gamma 需自定义增强或经 albumentations 接入 dataloader。
- 优点：不改网络结构、**不增加任何推理算力**、实现简单、可随时开关、几乎不过拟合。
- 缺点：对极端光照抖动（如光源故障、强烈反光频闪）的鲁棒性有限。

**方案 B：网络端光照校正模块（仅极端场景）**

- 手段：在预处理 / backbone 头部增加光照归一化分支（如自适应光照估计 + 校正层）。
- 适用：极端光照抖动、光照分布随批次漂移、数据增强已无法覆盖的场景。
- 缺点：必须修改 ultralytics 源码、增加推理开销、工程复杂度与回归风险高。

**何时必须加**：光照波动极大、且数据增强（含 CLAHE/Gamma）无法稳定指标时，才升级到方案 B。
**何时不建议加**：光照整体均匀稳定——仅开启基础 HSV 扰动即可，避免过拟合与工程复杂度。

### 7.2 小目标识别增强

- **适用判定**：目标像素 **< 32×32** 建议开启；目标尺寸偏大（> 64px）或小目标占比很低，则不建议。
- **推荐度**：✅ 按目标尺寸决定。

**实现手段（均为增强 / 训练策略层，不动网络）**

1. 多尺度训练：`imgsz=800 / 960`，提升小目标在特征图上的分辨率（最直接有效）；
2. 复制粘贴增强（copy-paste）：把小目标裁剪后复制到其他图像，增加小目标样本量；ultralytics 8.4.x 未内置，需自定义 dataloader；
3. 小目标 loss 加权：对面积小的 mask 增加权重（分割任务中小掩码更易丢失），需自定义 loss 或使用支持权重配置的分支；
4. （网络结构层，代价最高）降低下采样倍率 / 增加 P2 高分辨率检测头——仅当上述手段均不足时考虑。

**代价**：`imgsz` 提升后显存占用与训练时长**显著增加**，batch 可能需要同步下调；小目标权重 / copy-paste 增加训练期计算量，推理零成本。

**何时必须加**：目标普遍 < 32px 且小目标漏检是主要指标瓶颈。
**何时不建议加**：目标偏大（> 64px）或小目标占比极低——开启只会白增显存与训练时长。

### 7.3 注意力机制

- **适用场景**：工业缺陷、目标特征弱、背景杂乱，常规增强无法拉开前景 / 背景区分度。
- **推荐度**：⚠️ 按需选择（默认不开启，先跑基线）。

**实现手段（网络结构改造层）**

- 模块选型：CA / CoordAtt（坐标注意力，兼顾通道 + 位置，轻量）或 CBAM（通道 + 空间串联）；
- 插入位置：backbone 或 neck 的 C2f 层之后（通常 P3/P4 尺度收益最明显）；
- 接入方式：改模型 yaml + 注册自定义模块（详见第八章），ultralytics 8.4.78 的 parse_model 支持自定义模块。

**代价**：参数量 ↑、推理延迟 ↑、训练显存略增；数据量小时**容易过拟合**；自定义模块带来源码维护与版本升级成本。

**何时必须加**：目标特征弱 + 背景杂乱，且基线模型 Recall 明确不足、增强手段已用尽。
**何时不建议加**：目标简单、背景干净，或数据集规模小——过拟合风险大于收益，先跑无注意力基线对比。

---



## 八、自定义模块注入：带注意力模型的 YAML 样例

> ultralytics 8.4.78 支持自定义模块，但需要 **三处配合**：① 编写模块类；② 在 `ultralytics/nn/tasks.py` 的 parse_model 中注册；③ 在模型 yaml 中插入调用行。



### 8.1 模型 yaml（插入位置示意）

```yaml
# models/yolo26x-seg-with-attn.yaml（示意）
# 注意：请先通过 YOLO("yolo26x-seg.pt").model.yaml 导出完整原始结构，
#       再在指定 C2f 层之后插入注意力行；不要手写全量 backbone/head，避免与原结构不一致。
nc: 1
names: ['target_obj']

backbone:
  [[-1, 1, Conv, [64, 3, 2]],
   [-1, 1, C2f, [64, True]],
   [-1, 1, CoordAtt, [64]],          # 新增注意力：通道数保持不变，插在 C2f 之后
   [-1, 1, Conv, [128, 3, 2]],
   [-1, 1, C2f, [128, True]],
   [-1, 1, CoordAtt, [128]],         # 深层同样插入（按需，一般 P3/P4 收益最明显）
   # ... 其余 backbone 层保持原样（从导出的 model.yaml 原样补齐）
   ]

head:
  # ... 保持原样，不修改
```



### 8.2 CoordAtt / CBAM 模块实现（ultralytics 适配版）

```python
# ultralytics/nn/extra_modules.py（或新建 coordatt.py，再在 tasks.py 中 import）
import torch
import torch.nn as nn
import torch.nn.functional as F


class CoordAtt(nn.Module):
    """Coordinate Attention（坐标注意力）：通道 + 位置感知，轻量，适合弱特征小目标"""

    def __init__(self, inp, oup=None, reduction=32):
        super().__init__()
        oup = inp if oup is None else oup
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()
        self.conv_h = nn.Conv2d(mip, oup, 1, bias=False)
        self.conv_w = nn.Conv2d(mip, oup, 1, bias=False)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)                                  # (n,c,h,1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)              # (n,c,1,w)->(n,c,w,1)
        y = torch.cat([x_h, x_w], dim=2)                      # (n,c,h+w,1)
        y = self.act(self.bn1(self.conv1(y)))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)                         # (n,c,1,w)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        return identity * a_w * a_h


class CBAM(nn.Module):
    """CBAM（通道 + 空间注意力）：两分支串联，实现简单、适配性强"""

    def __init__(self, c1, reduction=16, kernel_size=7):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(c1, c1 // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1 // reduction, c1, 1, bias=False),
        )
        self.spatial = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.mlp(torch.mean(x, dim=(2, 3), keepdim=True))
        mx = self.mlp(torch.amax(x, dim=(2, 3), keepdim=True))
        x = x * self.sigmoid(avg + mx)                        # 通道注意力
        avg_s = torch.mean(x, dim=1, keepdim=True)
        mx_s = torch.amax(x, dim=1, keepdim=True)
        x = x * self.sigmoid(self.spatial(torch.cat([avg_s, mx_s], dim=1)))  # 空间注意力
        return x
```



### 8.3测试（ultralytics/nn/tasks.py）

```python
# 文件：ultralytics/nn/tasks.py —— 修改示例（两处）
# ① 文件顶部 import 区追加：
from ultralytics.nn.extra_modules import CoordAtt, CBAM

# ② parse_model 函数内部、现有 elif 通道计算分支之后追加
#    （实际插入时缩进需与函数体一致，注意力模块保持输入/输出通道不变）：
if m in {CoordAtt, CBAM}:
    c2 = ch[f]
```
