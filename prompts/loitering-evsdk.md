# EV-SDK 人员徘徊检测算法完整开发提示词（含自动化测试）

> 基于用户《人员徘徊.md》原始算法技术文档及《EV-SDK 老鼠入侵检测算法完整开发提示词》结构改写。属于**待执行任务规范**，不是代码或测试通过证明。

## 0. 角色与任务
角色：EV-SDK CV 算法工程师。
任务：在禁止徘徊 ROI 中识别人，维护跨帧稳定人员 ID，判断停留时间/人数条件，在满足连续帧规则时报警，并输出规范 JSON。视频输入为真实帧和时间戳；仅用单帧静态图不可证明徘徊。
数据路径：`/home/data/`。模型：`best.onnx`，完整类别映射 `0: person`。前向推理要求“YOLOv8”；必须对 YOLO26 导出模型的 ONNX 输入输出做适配检查，不得假定二者完全兼容。

## 1. 工程与运行环境约束
仅允许修改 `ev_sdk/config` 与 `ev_sdk/src`，不可改 SDK 接口/模型文件/其他目录。必须先审查已有工程语言、构建系统、已有 ROI 与跟踪接口；核心逻辑与 SDK IO 解耦，便于独立回归。支持多摄像头 `cid` 独立状态，避免跨镜头串号。未提供工程骨架时交付规范，不宣称已实现。

## 2. 业务报警逻辑
### 2.1 逐帧管线
图像 → ONNX 检测 person → 置信度与 ROI 过滤 → 为人员分配稳定 `person_id` → 每 CID/每人更新首次持续驻留时间 → 是否 `duration > time_thresh` → 根据业务人数条件判断 → 可选连续帧投票 → JSON/可视化。
- 目标在 ROI 内持续停留达到时间阈值后进入“徘徊候选”，离开 ROI 时结束该次驻留。原文未定义遮挡与重新进入规则；建议短暂丢帧保留状态超过宽限期重置，具体秒数须业务确认。
- 原文 `less_switch`、`equal_switch`、`threshold_count` 与 `linger_peroson_num` 共同存在但未定义全部组合优先级。**建议默认**：徘徊人员数量满足 `linger_peroson_num` 且持续时间满足阈值为基础条件；启用的其他人数条件以“或”额外触发，未启用不影响基础条件。严格遵守业务自定义口径，不将本建议视为原文事实。
- 若出现“ROI 内无人但人数过少要报警”的独立分支，其 `target_info=[]`、`target_count=0` 也可与 `is_alert=true` 共存；需业务明确，该分支不是“徘徊的某个人”。

### 2.2 阈值与边界
- 原文：`threshold_count` 条件为检测人数**大于**阈值；`less_count_person` 为**小于**；`equal_count_person` 为**等于**。
- 原文：时间判断为**大于**设置的时间阈值（非大于等于）。
- 投票原文描述“赞同票比例大于 90%”，默认 `alert_frame_ratio=0.9`，窗口变量在例子中写作 `alert_time_thresh=10`，字段表没有定义该窗口。为可实现性**建议补充** `alert_time_thresh`（int，默认 10），按严格 `>` 计算，是否包含当前帧与是否要求当前帧有效必须另行确认。
- 参数名 `linger_peroson_num` 属于原文拼写，默认保留以免破坏协议。

### 2.3 逐 CID 状态
`cid -> person_id -> {first_seen, last_seen, entered_roi, dwell_seconds}`。每 CID 独立投票队列。跟踪稳定性要求 ID 不跨镜头复用，时间使用输入时间戳或单调时钟，不使用推理耗时替代驻留秒数。跟踪算法需使用 SDK 已有实现或明确新增适配层（仅在允许路径内）。

## 3. 自定义配置参数
| 参数 | 类型 | 说明 |
|---|---|---|
| less_switch | bool | 是否开启“人数小于”报警 |
| less_count_person | int | 区域内人数 < 该值 |
| equal_switch | bool | 是否开启“人数等于”报警 |
| equal_count_person | int | 区域内人数 == 该值 |
| threshold_count | int | 人数 > 该值触发对应报警条件 |
| linger_peroson_num | int | 区域内徘徊人数阈值（原文拼写） |
| time_thresh | int | 超时阈值（秒） |
| alert_ratio_switch | bool | 默认 false；是否连续帧投票 |
| alert_frame_ratio | float | 默认 0.9；赞同票比例阈值 |
| used_time_visual | bool | 是否画“徘徊：XX 秒”或“滞留：XX 秒” |
| alert_time_thresh | int | **建议新增**投票窗口长度，原文示例提及但未列入字段表 |
| cid | string | **工程环境依赖**，摄像头标识 |
| roi_type / polygon_1 | string / array | **需对齐具体 SDK 标准参数** |

不得强行给未定义字段编造“官方”默认值；参数缺失或格式错误应输出可追踪配置错误并禁用误报警。设置 `labels_name=["person"]`、`thresh_list=[0.3]` 仅为建议示例，须按 SDK 协议校验。

## 4. JSON 输出协议
不报警示例（严格保留原文顶层字段）：
```json
{
  "algorithm_data": {"is_alert": false, "alert_time_s": 10, "target_count": 0, "target_info": []},
  "model_data": {"objects": [{"x": 716, "y": 716, "width": 233, "height": 646, "person_id": "1", "name": "person"}]}
}
```
报警示例：
```json
{
  "algorithm_data": {"is_alert": true, "alert_time_s": 10, "target_count": 1, "target_info": [{"x": 716, "y": 716, "width": 233, "height": 646, "person_id": "1", "name": "person"}]},
  "model_data": {"objects": [{"x": 716, "y": 716, "width": 233, "height": 646, "person_id": "1", "name": "person"}]}
}
```
字段 `person_id` 为 string；`name="person"`；`target_count == len(target_info)`；`model_data.objects` 包含检测到的人员结果，`target_info` 为报警对象。严格按现有 SDK 约定选定坐标系；不能从示例推导模型直接输出。

## 5. 自动化测试
| ID | 场景 | 预期 |
|---|---|---|
| T-01 | 模型 best.onnx 存在、单类别、输出 shape 与前处理一致 | 人框可正确解码；失败需报错 |
| T-02 | ROI 内人，驻留 9 秒，time_thresh=10 | 不报警 |
| T-03 | ROI 内人，驻留 11 秒，time_thresh=10 | 形成徘徊候选 |
| T-04 | 同一人短暂遮挡与回归 | 按明确丢失宽限期处理，不能跨人串 ID |
| T-05 | 两 CID 交错输入 | 各自计时、票队列独立 |
| T-06 | less/equal/threshold_count 三分支与边界 | 分别符合严格 <、==、> |
| T-07 | 多帧窗口 10、比例 0.9 | 严格 >0.9 则 9/10 不报警、10/10 才报警（仅在确认该口径后） |
| T-08 | 空 ROI、非法参数、空帧、回退默认值 | 无崩溃，无误报警 |
| T-09 | 标准 JSON 解析与 target_count | 类型和数组数量正确 |
| T-10 | /home/data 真实图片/视频 | 留存真实推理、标注真值与结果，不用静态图替代时序测试 |

建议为核心状态机增加可注入检测结果及时间戳的接口。没有 EV-SDK 代码和测试数据时这些是测试用例规范，**不是已通过的测试**。

## 6. 工程交付
`ev_sdk/config` 下的参数配置与 schema；`ev_sdk/src` 下的模型推理适配、人员跟踪、时序报警与 JSON 输出代码；在允许的文件范围内以现有构建系统接入自测入口。不得修改其他目录。

## 7. 验收标准
构建通过、测试图 ONNX 推理一致、ROI/时序/人数/多 CID 回归通过、报警 JSON 满足协议、日志可追溯，且修改范围仅 `ev_sdk/config` 与 `ev_sdk/src`。

## 8. 待确认事项
- 业务原文的四种人数报警条件（less/equal/threshold/linger）是否并列、优先级与组合操作；
- ROI 判断用框相交、脚底点还是框中心；跟踪丢失宽限、计时重置与重复报警抑制；
- `alert_time_thresh` 是否为真实配置字段、投票是否需包含当前帧；
- YOLO26 ONNX vs YOLOv8 具体 tensor layout；SDK 版本与引擎、输入帧格式。
