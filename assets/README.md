# 原始模型骨架文件包清单与处理

来源：用户提供的 `模型骨架文件包.zip`（压缩包约 234 MiB，未原样提交 GitHub）。

已读取归档目录，核心文件如下：

| 归档路径 | 说明 | 本仓库处理 |
|---|---|---|
| `train 2/prepare_and_train.py` | 原始 26 类 VOC 检测预处理 + 启动训练 | 参考；新实现 `training/prepare.py` |
| `train 2/run_train.py` | 26 类训练 200 epochs | 参考；新实现 `training/train.py` |
| `train 2/dataset.yaml` | 26 类检测配置 | 不直接适用单类 person |
| `train 2/export_onnx.py` | 输出 ONNX 的原始脚本 | 参考；新实现 `training/export_onnx.py` |
| `train 2/run.sh` / `start_train.sh` | 调用原始准备脚本 | 新训练使用 README 命令 |
| `train 2/export_onnx.sh` | 拷贝预期的 ONNX 文件 | 新训练使用明确参数导出 |
| `train 2/ultralytics-8.4.78.zip` | 本地框架压缩包 | 安装时按其来源验证；未提交 |
| `train 2/yolo26x.pt` | 目标检测预训练权重，约 119 MB | 本地部署或 Git LFS，不进入普通 Git |
| `train 2/yolo26x-seg.pt` | 实例分割预训练权重，约 142 MB | 本地部署或 Git LFS，不进入普通 Git |

## 安装说明

1. 在训练服务器上解压用户原始骨架包，保留完整模型权重。
2. 将输入图像及标注放入 `/home/data/`；检测需要同名 VOC XML，分割需要同名 LabelMe JSON。
3. 新 Skill 的脚本位于本仓库 `training/`，不要用原始 26 类 `dataset.yaml` 覆盖新生成的单类 YAML。
4. 若需要保留所有二进制文件，请单独配置 Git LFS（或受控对象存储），核对 GitHub LFS 配额和访问权限后上传；不得通过普通 Git 写入超过其限制的大权重。
5. 骨架包**没有 `best.onnx`、没有 EV-SDK 工程、也没有训练图像数据**。上述资源需要用户训练/部署环境提供。

## 风险备注

- 原始 `prepare_and_train.py` 会扫描 VOC、生成 26 类目标检测标签并自动训练，不满足 YOLO Seg 的多边形标注要求。
- 原始 `run_train.py` 为 200 epoch，训练模板为 150 epoch；新 Skill 以参数化 150 epoch 为默认。
- ONNX 模型是否可用传统 YOLOv8 后处理，需要逐 Tensor 验证，不能仅根据权重名字判断。