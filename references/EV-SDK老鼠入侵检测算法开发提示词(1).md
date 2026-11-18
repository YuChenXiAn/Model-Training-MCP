# EV-SDK 老鼠入侵检测算法完整开发提示词（含自动化测试）

> 用途：本文件为**完整可复制提示词**，可直接交给大模型 / 开发人员，依据以下规格实现"老鼠入侵检测报警算法"，并交付可自动验证的代码与测试。
> 内容结构：① 角色与任务 → ② 工程约束 → ③ 算法报警逻辑 → ④ 配置参数 → ⑤ JSON 输出协议 → ⑥ 自动化测试 → ⑦ 交付物与验收 → ⑧ 默认口径与待确认事项。

---

## 0. 角色与总体任务

**角色**：计算机视觉算法工程专家，熟悉 EV-SDK（极市平台）开发规范、YOLOv8 ONNX 推理、变化检测、多帧状态机与自动化测试。

**任务**：在给定 EV-SDK 工程中实现"老鼠入侵检测报警算法"，满足以下全部规格：

1. 使用人员检测模型与老鼠检测模型，结合 2 帧队列变化检测，在 ROI 范围内按既定分支逻辑输出报警；
2. 支持多摄像头 cid 维度下的多帧判断（每 cid 独立队列）；
3. 全部业务参数通过配置文件下发，输出协议严格遵循本文档第 5 节；
4. 交付自动化测试：覆盖配置解析、模型加载、判定分支、多帧/cid 状态机、输出协议、边界异常，一键运行可回归。

---

## 1. 工程与运行环境约束

| 项目 | 约束 |
|---|---|
| 修改范围 | 仅允许修改 `ev_sdk/config` 与 `ev_sdk/src` 内的代码；模型文件、SDK 框架、平台接口不得改动 |
| 模型输入 | `best.onnx`，总 label：`['rat', 'person']`，采用 YOLOv8 前向推理 |
| 人员模型选择 | 由 `person_model_choice` 控制：`yolov8`（YOLOv8 自带）／`cvmart_trained`（极市训练模型） |
| 测试图像 | 位于 `/home/data/` 下 |
| 开发语言 | 以 EV-SDK 既有工程语言为准（通常为 C++）；**核心逻辑必须与 SDK 接口解耦**（独立模块 / 纯函数），便于自动化测试；若使用 Python 实现核心逻辑，需通过 SDK 封装为可加载动态库并说明构建方式 |
| 训练 | 不执行训练；输出代码需可直接构建、可直接运行测试 |

---

## 2. 算法报警逻辑

### 2.1 单路算法逻辑

模型输入为 `best.onnx`，对应总 label：`['rat', 'person']`，采用 YOLOv8 前向推理。

**1. 人员检测**
- 使用一个单独的目标检测模型检测人员；
- 设置 string 类型配置参数 `person_model_choice`，用于控制是否检测人 / 选择人员检测模型。

**2. 老鼠检测**
- 使用一个单独的目标检测模型检测老鼠。

**3. 帧队列与变化检测**
- 算法使用一个 **2 帧队列**存储接收到的视频帧，每隔固定时间间隔更新队列，该时间间隔由 float 配置参数 `interval_between_frames` 控制；
- 每次更新帧队列时，使用变化检测算法检测 2 帧图像中的目标变化，并输出变化目标的 2D 框。

**4. ROI 内综合判断**

根据上述 3 个模型的结果，在 ROI 范围内进行如下判断：

- **分支 1：老鼠目标检测模型（rat_det）在 ROI 内检测到了老鼠**
  - 此时算法**不进入老鼠变化检测部分**，并且：
    - **A、** 该老鼠与人员有交集 → **不认为该目标是老鼠，直接过滤掉，不报警**；
    - **B、** 该老鼠与人员没有交集 → **报警该老鼠**。
- **分支 2：老鼠目标检测模型（rat_det）在 ROI 内没有检测到老鼠**
  - 根据是否开启变化检测（`use_change_detection`），进入变化检测部分；若变化检测模型检测到了老鼠，则：
    - **A、** 此时整个画面中有人 → **不对变化检测部分检测到的老鼠报警**；
    - **B、** 此时整个画面中没有人 → **对该老鼠报警**。

### 2.2 多摄像头 cid 下的多帧判断逻辑

1. 对于**每一个 cid** 建立一个队列，记录过去 n 帧内检测到报警目标的情况；满足以下 **2 个条件**才真正报警：
   - **a、** 过去 n 帧内检测到报警目标的帧数达到一定比例；
   - **b、** 当前帧也检测到报警目标。

   由 3 个配置参数控制：
   - `use_frame_switch`（bool）：是否开启多帧判断；
   - `alert_frame_thresh`（int）：多帧判断的帧数，即上述 n 值；
   - `alert_frame_ratio`（float）：在 `alert_frame_thresh` 控制的帧数内，检测到报警的帧数需达到的比例（且当前帧也检测到报警目标）才报警。

2. **cid 轮询隔离（必须实现）**：多摄像头轮询模式下，前后帧可能不来自同一摄像头 cid。算法内部必须建立**哈希表 / 字典**，根据输入参数中的 `cid` 字段，为**每一个 cid 独立存储队列**，记录该 cid 过去 n 帧的情况。

3. **调试输出**：额外设置 bool 配置参数 `display_status`，当其为 `true` 时，在算法 JSON 输出的 `algorithm_data` 部分：
   - **a、** 添加 `current_cid` 字段，显示当前得到的 cid 字符串（调试用）；
   - **b、** 添加 `last_n_frame` 字段，显示过去 n 帧的报警结果，字符串格式，`1` 表示报警、`0` 表示不报警，**字符串最左侧为倒数第 n 帧，最右侧为当前帧**。
     示例：`"11000"` 表示倒数第 5 帧、第 4 帧检测到报警目标，倒数第 3 帧、第 2 帧、当前帧未检测到报警目标。

---

## 3. 自定义配置参数

### 3.1 config.json 模板（默认值）

```json
{
  "cid": "",
  "roi_type": "polygon_1",
  "polygon_1": [],
  "labels_name": ["person", "rat_det", "rat_chg"],
  "thresh_list": [0.3, 0.5, 0],
  "warning_list": ["person", "rat_det", "rat_chg"],
  "interval_between_frames": 0.2,
  "min_pixels": 10,
  "max_pixels": 10000,
  "use_person_model": false,
  "use_rat_det_model": true,
  "use_change_detection": "false",
  "filter_text_from_roi": true,
  "time_interval_to_detect_text": 30.0,
  "draw_text_detection": false,
  "text_rectangle_expanding": [10, 10],
  "use_only_ir": true,
  "use_frame_switch": true,
  "alert_frame_thresh": 5,
  "alert_frame_ratio": 0.8,
  "person_model_choice": "cvmart_trained",
  "display_status": false
}
```

### 3.2 参数说明表

| 字段名称 | 类型 | 默认值 | 备注 |
|----------|------|--------|------|
| cid | string | `""` | 用于接收上层应用下发的摄像头 cid |
| roi_type | string | `"polygon_1"` | 表示本算法用到的 roi 类型 |
| polygon_1 | array[string] | `[]` | 一个多边形的 roi 区域，算法只需要识别这个区域内的入侵异物 |
| labels_name | array | `["person","rat_det","rat_chg"]` | 不可改动，为 thresh_list 显示对应标签，按顺序：`person` 人；`rat_det` 目标检测模型检测到的老鼠；`rat_chg` 变化检测模型检测到的老鼠 |
| thresh_list | array | `[0.3, 0.5, 0]` | 为每个不同标签类型设置置信度阈值，顺序对应 labels_name；变化检测模型不支持阈值，第 3 个阈值当前无用 |
| warning_list | array | `["person","rat_det","rat_chg"]` | 哪些类别需要报警 |
| interval_between_frames | float | `0.2` | 变化检测帧队列中相邻帧的时间间隔；超过该时长时存入一帧新的参考帧 |
| min_pixels | int | `10` | 2D 框像素大小小于 min_pixels 时过滤，不报警 |
| max_pixels | int | `10000` | 2D 框像素大小大于 max_pixels 时过滤，不报警 |
| use_person_model | bool | `false` | 人员检测模型得到的目标 2D 框，是否将符合 ROI 范围的 2D 框输出 |
| use_rat_det_model | bool | `true` | 是否启用目标检测老鼠的模型 |
| use_change_detection | bool(字符串) | `"false"` | 变化检测模型得到的目标 2D 框是否输出。可选值：`"true"` 输出；`"false"` 不输出；`"default"` 按 2.1 第 4 条优先级决定是否输出 |
| filter_text_from_roi | bool | `true` | 是否将画面文字从 ROI 范围内过滤出去，防止文字变化导致变化检测误报 |
| time_interval_to_detect_text | float | `30.0` | 每间隔多长时间检测一次文字（文字范围很少变化，无需一直检测） |
| draw_text_detection | bool | `false` | 是否将画面中检测到的文字用 2D 框显示出来 |
| text_rectangle_expanding | array | `[10,10]` | 对文字 2D 框左右、上下各扩展多少像素（文字检测框偶尔偏小） |
| use_only_ir | bool | `true` | `true` 时只对红外黑白画面识别；`false` 时对彩色和红外黑白画面都识别 |
| use_frame_switch | bool | `true` | 是否开启多帧判断 |
| alert_frame_thresh | int | `5` | 多帧判断的帧数（即 n 值） |
| alert_frame_ratio | float | `0.8` | 在 alert_frame_thresh 帧数内，报警帧数达到该比例（且当前帧也报警）才报警 |
| person_model_choice | string | `cvmart_trained` | 用 YOLOv8 还是极市训练模型检测人员：`yolov8` 用 YOLOv8 自带；`cvmart_trained` 用极市训练 |
| display_status | bool | `false` | 为 true 时，algorithm_data 输出 `current_cid` 与 `last_n_frame` 调试字段 |

### 3.3 参数校验规则

1. 所有参数**必须有默认值**，缺失时回退默认值，不允许崩溃；
2. 类型不合法（如 `alert_frame_thresh` 传字符串）时：报错并回退默认值，或明确返回配置错误信息（二选一，实现时固定口径）；
3. `labels_name` 不可改动，且顺序与 `thresh_list`、`warning_list` 一一对应；
4. `polygon_1` 必须是 ≥3 点的合法多边形，非法时明确报错；
5. `use_change_detection` 仅接受 `"true"` / `"false"` / `"default"` 三值。

---

## 4. JSON 输出协议

> 下面假设 `mask_output_path` 字段不为空。

### 4.1 不告警情形示例

```json
{
    "algorithm_data": {
        "is_alert": false,
        "target_count": 0,
        "target_info": []
    },
    "model_data": {
        "objects": []
    }
}
```

### 4.2 告警情形示例

```json
{
    "algorithm_data": {
        "is_alert": true,
        "target_count": 1,
        "target_info": [
            {
                "x": 397,
                "y": 397,
                "height": 488,
                "width": 215,
                "confidence": 0.8623,
                "name": "rat"
            }
        ]
    },
    "model_data": {
        "objects": [
            {
                "x": 397,
                "y": 397,
                "height": 488,
                "width": 215,
                "confidence": 0.8623,
                "name": "rat"
            }
        ]
    }
}
```

### 4.3 字段详解

| 字段名称 | 类型 | 备注 |
|----------|------|------|
| algorithm_data | object | 业务相关的输出信息 |
| is_alert | bool | 报警标识，true 告警，false 不告警 |
| target_count | int | target_info 中对象数目 |
| target_info | array | 报警信息 |
| model_data | object | 模型相关的输出信息 |

**target_info 字段：**

| 字段名称 | 类型 | 备注 |
|----------|------|------|
| x | int | 目标外接矩形左上角 X 坐标 |
| y | int | 目标外接矩形左上角 Y 坐标 |
| width | int | 目标外接矩形宽度 |
| height | int | 目标外接矩形高度 |
| confidence | float | 置信度 |
| name | string | 检测到告警时目标名称。老鼠：`rat` |

**model_data.objects 字段：**

| 字段名称 | 类型 | 备注 |
|----------|------|------|
| x | int | 识别框左上角 X 坐标 |
| y | int | 识别框左上角 Y 坐标 |
| width | int | 识别框宽度 |
| height | int | 识别框高度 |
| confidence | float | 置信度 |
| name | string | 模型检测到的目标名称：老鼠 `rat`；人员 `person` |

### 4.4 协议规则

1. `algorithm_data` 对应业务数据，**仅当检测到至少 1 个告警目标时 `target_info` 才非空**；
2. 报警目标的 `name` 统一为 `rat`（或 `person`，见 8-① 待确认口径），**不得输出内部标签** `rat_det` / `rat_chg`；
3. `model_data.objects` 输出模型检测到的全部目标（含人员与老鼠，按 `use_person_model` / `use_rat_det_model` 等开关）；
4. `display_status=true` 时，`algorithm_data` 额外包含 `current_cid`（当前 cid 字符串）与 `last_n_frame`（n 位 0/1 字符串，最左为倒数第 n 帧、最右为当前帧）。

---

## 5. 自动化测试内容

### 5.1 测试目标

1. 核心判定逻辑**分支全覆盖**（2.1 第 4 条四种分支）；
2. 多帧判断状态机与**多 cid 隔离**正确性；
3. JSON 输出协议**逐字段校验**（类型、取值范围、枚举值）；
4. 配置解析、模型加载、ROI/尺寸过滤、文字过滤、红外模式覆盖；
5. 回归保护：修改 `ev_sdk/config` 或 `ev_sdk/src` 后一键运行全部测试。

### 5.2 测试架构要求

1. **核心逻辑与 SDK 解耦**：将判定逻辑、2 帧队列、多帧状态机、cid 哈希表实现为独立模块（不依赖摄像头 / SDK 输入），暴露纯函数 / 纯类接口（如 `AlarmEngine`），SDK 层仅做数据搬运与 JSON 组装；
2. 测试代码放置：优先 `ev_sdk/test/`；若与"仅修改 config/src"约束冲突，放工程根目录 `tests/`，作为开发自测代码，不影响交付物；
3. 测试数据：优先使用 `/home/data/` 下真实图像；分支覆盖不足时使用**合成图像补齐**（构造方法见 5.6）；
4. 运行方式：`pytest` 一键运行，并输出覆盖率报告（核心模块 branch 覆盖率）。

### 5.3 测试用例总表

#### A. 配置解析与模型加载

| 用例ID | 用例描述 | 输入 / 场景 | 预期结果 | 优先级 |
|---|---|---|---|---|
| T-CFG-01 | 默认配置加载 | 空配置 / 只传最小字段 | 全部参数等于默认值表（3.2） | P0 |
| T-CFG-02 | 参数类型校验 | `alert_frame_thresh="abc"` | 明确报错或回退默认值并告警，不崩溃 | P1 |
| T-CFG-03 | 缺失字段回退 | 缺省 `polygon_1` | 按空 ROI 处理：不误报、不崩溃 | P1 |
| T-CFG-04 | 多边形合法性 | 合法 ≥3 点 / 非法（<3 点、自相交） | 合法正常解析；非法明确报错 | P1 |
| T-MODEL-01 | best.onnx 加载 | 指定权重路径 | 加载成功，类别 = `['rat','person']` | P0 |
| T-MODEL-02 | person 模型选择 | `person_model_choice=yolov8` / `cvmart_trained` | 两种取值均加载对应模型并推理 | P1 |
| T-MODEL-03 | 模型缺失处理 | 路径不存在 | 返回明确错误信息，不静默崩溃 | P1 |

#### B. 检测与过滤

| 用例ID | 用例描述 | 输入 / 场景 | 预期结果 | 优先级 |
|---|---|---|---|---|
| T-DET-01 | 人 / 鼠检出 | `/home/data/` 真实测试图 | rat 与 person 均正确检出并输出到 model_data.objects | P0 |
| T-ROI-01 | ROI 内目标参与判断 | 目标框与 ROI 相交 | 参与报警判断（口径见 8-②） | P0 |
| T-ROI-02 | ROI 外目标过滤 | 目标框完全在 ROI 外 | 不参与报警判断 | P0 |
| T-FILTER-01 | min_pixels 过滤 | 框面积 < 10 | 过滤，不报警 | P1 |
| T-FILTER-02 | max_pixels 过滤 | 框面积 > 10000 | 过滤，不报警 | P1 |
| T-FILTER-03 | 阈值边界 | 面积恰等于 10 / 10000 | 保留（≥min 且 ≤max），口径固定 | P2 |
| T-THRESH-01 | thresh_list 生效 | person 阈值 0.3、rat_det 阈值 0.5 | 低于阈值的候选被剔除 | P1 |

#### C. 判定逻辑（对应 2.1 真值表，见 5.4）

| 用例ID | 用例描述 | 输入 / 场景 | 预期结果 | 优先级 |
|---|---|---|---|---|
| T-LOGIC-01 | 分支 1A | ROI 内 rat_det 检出鼠 ∧ 人鼠有交集 | 过滤，不报警 | P0 |
| T-LOGIC-02 | 分支 1B | ROI 内 rat_det 检出鼠 ∧ 无交集 | 报警 `rat` | P0 |
| T-LOGIC-03 | 分支 2A | ROI 内无鼠 ∧ 变化检测检出鼠 ∧ 画面有人 | 不报警 | P0 |
| T-LOGIC-04 | 分支 2B | ROI 内无鼠 ∧ 变化检测检出鼠 ∧ 画面无人 | 报警 `rat` | P0 |
| T-LOGIC-05 | 变化检测开关 | `use_change_detection="false"` | 不进入分支 2，变化检测结果不参与报警 | P1 |
| T-LOGIC-06 | 模型开关组合 | `use_person_model` / `use_rat_det_model` 各取值 | 开关行为与参数表一致 | P1 |
| T-LOGIC-07 | warning_list 生效 | 调整 warning_list | 仅 warning_list 内类别可进入 target_info | P1 |
| T-LOGIC-08 | person 告警口径 | `warning_list` 含 person 且检出人员 | 按第 8 节①默认口径行为一致（默认仅鼠报警） | P2 |

#### D. 多帧判断与 cid 隔离

| 用例ID | 用例描述 | 输入 / 场景 | 预期结果 | 优先级 |
|---|---|---|---|---|
| T-MULTI-01 | 关闭多帧判断 | `use_frame_switch=false` | 当前帧判定即报警，无延迟 | P0 |
| T-MULTI-02 | 比例达标 | n=5, ratio=0.8，过去 5 帧 ≥4 帧命中 ∧ 当前帧命中 | 报警 | P0 |
| T-MULTI-03 | 比例不达标 | 同参数，仅 2/5 帧命中 | 不报警 | P0 |
| T-MULTI-04 | cid 队列隔离 | cam_A 达标、cam_B 不达标，交替输入 | cam_A 报警、cam_B 不报警，互不影响 | P0 |
| T-MULTI-05 | 轮询交错场景 | 帧序列 cid 交替（A,B,A,B,…） | 各 cid 独立计数，结果正确 | P0 |
| T-MULTI-06 | display_status=true | 输出含调试字段 | `current_cid` 正确；`last_n_frame` 为 n 位 0/1 串，最左=倒数第 n 帧、最右=当前帧（如 `"11000"`） | P0 |
| T-MULTI-07 | display_status=false | 输出不含调试字段 | 无 `current_cid` / `last_n_frame` | P1 |
| T-QUEUE-01 | 帧入队间隔 | Δt ≥ `interval_between_frames` | 存入新参考帧并触发变化检测 | P1 |
| T-QUEUE-02 | 帧入队间隔（未达） | Δt < `interval_between_frames` | 不更新队列 | P1 |
| T-QUEUE-03 | 队列容量 | 持续输入 | 队列恒为 2 帧 | P2 |

#### E. 文字过滤与红外模式

| 用例ID | 用例描述 | 输入 / 场景 | 预期结果 | 优先级 |
|---|---|---|---|---|
| T-TEXT-01 | 文字区域防误报 | ROI 内静态文字，两帧间文字不变 | 变化检测不因文字误报 | P1 |
| T-TEXT-02 | 文字框扩展 | `text_rectangle_expanding=[10,10]` | 文字框左右/上下各扩展 10px 后参与过滤 | P2 |
| T-TEXT-03 | 文字检测间隔 | `time_interval_to_detect_text=30.0` | 30s 内不重复检测文字 | P2 |
| T-TEXT-04 | 文字框显示 | `draw_text_detection=true` | 输出文字 2D 框（调试用途） | P2 |
| T-IR-01 | 仅红外识别 | `use_only_ir=true` | 彩色帧跳过识别，红外黑白帧正常识别 | P1 |
| T-IR-02 | 全部识别 | `use_only_ir=false` | 彩色与红外黑白帧均识别 | P2 |

#### F. 输出协议与稳定性

| 用例ID | 用例描述 | 输入 / 场景 | 预期结果 | 优先级 |
|---|---|---|---|---|
| T-OUT-01 | 不告警输出结构 | 无报警目标 | `is_alert=false`、`target_count=0`、`target_info=[]` | P0 |
| T-OUT-02 | 告警字段校验 | 告警场景 | x,y ≥ 0；width,height > 0；0 ≤ confidence ≤ 1；name ∈ {rat, person} | P0 |
| T-OUT-03 | 协议一致性 | 告警场景 | target_info 与 model_data.objects 中对应目标框一致（或按 8-③ 口径） | P0 |
| T-OUT-04 | name 归一化 | 内部标签 rat_det / rat_chg 场景 | 对外 name 统一为 `rat`，不出现内部标签 | P0 |
| T-EDGE-01 | 空帧 / 无效帧 | 空图像、全黑帧 | 不崩溃，`is_alert=false` | P1 |
| T-EDGE-02 | 无检测结果 | 无目标帧 | objects 为空数组，正常输出协议结构 | P1 |
| T-EDGE-03 | ROI 边界目标 | 目标框落在 ROI 边界 | 行为固定（按 8-② 口径，相交即算入） | P2 |
| T-PERF-01 | 单帧耗时 | 640 分辨率连续帧 | 单帧处理耗时 ≤ 阈值（建议 50ms，可按平台调整） | P1 |
| T-PERF-02 | 内存 / 显存稳定 | 连续运行 ≥1000 帧 | 无持续增长（无泄漏） | P1 |

### 5.4 判定真值表（2.1 分支全覆盖）

> 实现与测试均须与此表一致。`-` 表示该条件下不参与判断（分支 1 命中时不进入变化检测）。

| 编号 | rat_det 在 ROI 内检出鼠 | 鼠与人有交集 | 画面中有人 | 变化检测检出鼠 | 是否报警 | 对应逻辑 |
|---|---|---|---|---|---|---|
| ① | 是 | 是 | - | - | **否**（过滤） | 2.1-4-1A |
| ② | 是 | 否 | - | - | **是** | 2.1-4-1B |
| ③ | 否 | - | 是 | 是 | **否** | 2.1-4-2A |
| ④ | 否 | - | 否 | 是 | **是** | 2.1-4-2B |
| ⑤ | 否 | - | - | 否 | **否** | 无报警目标 |
| ⑥ | 否 | - | 否 | 是（但 `use_change_detection="false"`） | **否** | 未启用变化检测 |

### 5.5 关键用例细化（多帧 / cid / last_n_frame）

**T-MULTI-05 轮询交错示例**（cid 各自独立）：

```text
输入帧序列（frame_hit 表示该帧判定为报警目标）：
  帧1: cid=A, hit=1    帧2: cid=B, hit=1    帧3: cid=A, hit=1
  帧4: cid=B, hit=0    帧5: cid=A, hit=1    帧6: cid=B, hit=1
  帧7: cid=A, hit=0    帧8: cid=B, hit=1    帧9: cid=A, hit=1
  帧10: cid=B, hit=1

配置：use_frame_switch=true, alert_frame_thresh=5, alert_frame_ratio=0.8

cid=A 最近 5 帧（帧5~帧9，不含其他 cid）：1,1,0,1 → 4 帧命中？→ 实际窗口内 A 帧序列：
  帧1,3,5,7,9 = 1,1,1,0,1 → 4/5 命中且当前帧(帧9)命中 → 帧9 处 cid=A 报警
cid=B 最近 5 帧：帧2,4,6,8,10 = 1,0,1,1,1 → 4/5 命中且当前帧命中 → 帧10 处 cid=B 报警

断言：帧9 输出 cid=A 报警；帧10 输出 cid=B 报警；两者互不串扰。
```

**T-MULTI-06 last_n_frame 格式示例**：

```text
配置：alert_frame_thresh=5，cid=cam_1，最近 5 帧命中序列为 [1,1,0,0,0]
输出：last_n_frame = "11000"
解释：最左 '1' = 倒数第 5 帧（命中），最右 '0' = 当前帧（未命中）
```

### 5.6 测试数据构造方法

1. **基础场景图**（优先取自 `/home/data/`）：含人、含鼠、人鼠同框、空背景四类；再按以下维度组合：
   - ROI 内 / ROI 外 / ROI 边界；
   - 目标面积 < min_pixels / 正常 / > max_pixels；
   - 置信度高 / 低（低于 / 高于 thresh_list）；
   - 红外黑白 / 彩色；
2. **变化检测触发帧对**：同一场景两张图，第二张在 ROI 内叠加小目标（合成差异区域），使变化检测输出目标，用于覆盖分支 2；
3. **文字干扰帧对**：ROI 内叠加静态文字区域，验证 `filter_text_from_roi` 不因文字变化误报；
4. **命名规范**：`t_<case_id>_<scenario>.jpg` + 配套 `expected_<case_id>.json`（期望输出），测试读取期望文件断言。

### 5.7 pytest 骨架示例（最小可用，接口按最终实现调整）

```python
# ev_sdk/test/test_core_logic.py（示例骨架）
# 核心逻辑模块 core_logic 需与 SDK 解耦，统一暴露 AlarmEngine
from core_logic import AlarmEngine  # 按实际实现调整导入


def make_engine(**overrides):
    cfg = {
        "polygon_1": ["0,0", "100,0", "100,100", "0,100"],
        "thresh_list": [0.3, 0.5, 0.0],
        "min_pixels": 10,
        "max_pixels": 10000,
        "use_frame_switch": True,
        "alert_frame_thresh": 5,
        "alert_frame_ratio": 0.8,
        "use_change_detection": "default",
    }
    cfg.update(overrides)
    return AlarmEngine(cfg)


def test_rat_in_roi_without_person_alerts():
    # 分支 2.1-1B：ROI 内检出鼠、与人员无交集 → 报警
    engine = make_engine(use_frame_switch=False)
    result = engine.step(cid="cam_1", rat_in_roi=True, person_in_scene=False,
                         rat_person_intersect=False, change_rat=False)
    assert result["is_alert"] is True
    assert result["target_info"][0]["name"] == "rat"


def test_rat_intersect_with_person_filtered():
    # 分支 2.1-1A：与人员有交集 → 过滤，不报警
    engine = make_engine(use_frame_switch=False)
    result = engine.step(cid="cam_1", rat_in_roi=True, person_in_scene=True,
                         rat_person_intersect=True, change_rat=False)
    assert result["is_alert"] is False


def test_change_detection_with_person_not_alert():
    # 分支 2.1-2A：变化检测检出鼠但画面有人 → 不报警
    engine = make_engine(use_frame_switch=False)
    result = engine.step(cid="cam_1", rat_in_roi=False, person_in_scene=True,
                         rat_person_intersect=False, change_rat=True)
    assert result["is_alert"] is False


def test_change_detection_without_person_alerts():
    # 分支 2.1-2B：变化检测检出鼠且画面无人 → 报警
    engine = make_engine(use_frame_switch=False)
    result = engine.step(cid="cam_1", rat_in_roi=False, person_in_scene=False,
                         rat_person_intersect=False, change_rat=True)
    assert result["is_alert"] is True


def test_multi_frame_switch_and_cid_isolation():
    # n=5, ratio=0.8：过去 5 帧 ≥4 帧命中 且 当前帧命中 → 报警；cid 队列互相独立
    engine = make_engine()
    cam_a_seq = [True, True, True, False, True]    # 4/5 命中，当前帧命中 → 报警
    cam_b_seq = [True, True, False, False, False]  # 2/5 命中 → 不报警
    r_a = r_b = None
    for hit_a, hit_b in zip(cam_a_seq, cam_b_seq):
        r_a = engine.step(cid="cam_A", frame_hit=hit_a, rat_in_roi=hit_a,
                          person_in_scene=False, rat_person_intersect=False)
        r_b = engine.step(cid="cam_B", frame_hit=hit_b, rat_in_roi=hit_b,
                          person_in_scene=False, rat_person_intersect=False)
    assert r_a["is_alert"] is True
    assert r_b["is_alert"] is False


def test_last_n_frame_debug_output():
    # display_status=true：输出 current_cid 与 last_n_frame（最左=倒数第 n 帧，最右=当前帧）
    engine = make_engine(display_status=True)
    for hit in [True, True, False, False, False]:
        engine.step(cid="cam_1", frame_hit=hit, rat_in_roi=hit,
                    person_in_scene=False, rat_person_intersect=False)
    debug = engine.debug_state("cam_1")
    assert debug["current_cid"] == "cam_1"
    assert debug["last_n_frame"] == "11000"
```

### 5.8 测试运行与验收标准

**运行方式**（以工程实际为准）：

```bash
# C++ 工程：构建后运行
cd ev_sdk && mkdir -p build && cd build && cmake .. && make
# 自动化测试（核心逻辑为 Python 时）
cd ev_sdk && python -m pytest test -v --cov=src --cov-report=term-missing
```

**验收标准**：

1. **P0 用例 100% 通过**，P1 通过率 ≥ 95%（未通过的需在交付说明中注明原因与影响）；
2. 核心判定函数**分支覆盖率 100%**（以 5.4 真值表为准）；
3. JSON 输出协议**逐字段校验**通过（类型、取值范围、枚举值，见 4.3）；
4. 多帧 / cid 用例覆盖**轮询交错场景**（T-MULTI-05）；
5. 性能用例满足单帧耗时上限，内存 / 显存无持续增长；
6. 修改 `ev_sdk/config` 或 `ev_sdk/src` 后，全部测试可一键回归。

---

## 6. 交付物清单

| 交付物 | 说明 |
|---|---|
| `ev_sdk/config/config.json` | 含全部参数及默认值（见第 3 节模板） |
| `ev_sdk/src` 全部实现 | 模型推理封装（best.onnx / person 模型）、2 帧队列、变化检测、ROI 判定逻辑、多帧 cid 状态机、文字过滤、红外模式、JSON 组装 |
| 核心逻辑独立模块 | 与 SDK 解耦，便于测试（如 `src/core_logic.*`） |
| 自动化测试代码 | `ev_sdk/test/`（或工程根 `tests/`），含用例与期望文件 |
| 测试数据构造脚本 | 合成变化检测帧对 / 文字干扰帧对的生成脚本 |
| 构建与运行说明 | README 或代码注释：构建方式、测试命令、参数口径说明 |

---

## 7. 总体验收标准（汇总）

1. 仅修改 `ev_sdk/config` 与 `ev_sdk/src`，SDK 框架与模型文件未改动；
2. 输入 `best.onnx`（labels `['rat','person']`），YOLOv8 前向推理，人员模型可按 `person_model_choice` 切换；
3. 2 帧队列 + 变化检测按 `interval_between_frames` 更新，ROI 内判定逻辑与 2.1 / 5.4 真值表完全一致；
4. 多 cid 各自独立多帧队列，`use_frame_switch` / `alert_frame_thresh` / `alert_frame_ratio` 生效，轮询交错场景正确；
5. JSON 输出协议与第 4 节完全一致；`display_status=true` 时输出 `current_cid` / `last_n_frame`；
6. 自动化测试按第 5 节覆盖并通过 5.8 验收标准。

---

## 8. 默认口径与待确认事项

> 以下口径在需求中未完全明确，**默认按下列方式实现**；如与实际需求不符，请在交付说明中标注差异。

| 编号 | 事项 | 默认实现口径 |
|---|---|---|
| ① | `warning_list` 含 person 时，人员是否独立报警 | **默认仅鼠类报警**：person 仅参与判断（人鼠交集 / 画面是否有人），不进入 target_info；如需 person 告警，需另行确认 ROI 与阈值口径 |
| ② | ROI 判定口径 | 目标 2D 框与 ROI **相交即算入 ROI**（含边界接触） |
| ③ | 变化检测输出置信度 | 变化检测模型无阈值，输出目标 `confidence` 固定为 `1.0`，`thresh_list[2]` 不使用 |
| ④ | min_pixels / max_pixels 作用对象 | 应用于所有参与报警判定的候选 2D 框（rat_det 与变化检测输出）；person 不参与尺寸过滤 |
| ⑤ | 红外帧判定依据 | 按通道数 / 灰度占比判断（如图像为单通道或灰度占比 > 90% 视为红外黑白帧） |
| ⑥ | 语言与构建方式 | 以 EV-SDK 既有工程语言为准；若核心逻辑用 Python，需说明动态库封装与构建方式 |
| ⑦ | 变化检测算法选型 | 由实现者确定（帧差 / 光流 / 轻量分割等），需满足性能指标（T-PERF-01），并支持"仅输出变化目标的 2D 框" |