## 一、算法报警逻辑

1、业务场景需求:对禁止徘徊的区域进行实时监测，发现有人在该区域出现长时间徘徊，进行上报处理。
2、算法逻辑说明：对roi区域进行实时检测，发现有人长时间出现在该区域，则发生报警。

## 二、自定义配置参数

1、标准参数说明：[https://gitee.com/cvmart/ev_sdk_demo4.0_pedestrian_intrusion_yolov5](https://gitee.com/cvmart/ev_sdk_demo4.0_pedestrian_intrusion_yolov5)

2、可视化需求

a.不满足设置的时间阈值（未报警时），在推理图片左上角可视化徘徊的时间，打印（徘徊：XX 秒） 

b. 大于设置的时间阈值（报警时），报警图片左上角可视化徘徊超时的时间，打印（滞留：XX秒），其中滞留的时间计算方式： 目标实际徘徊时间 - 计时器设置时长

3、骨干模型选取：建议使用`YOLO26s，`兼顾小目标检测能力、推理性能；

4、自定义参数说明：


| 序号  | 字段名称                 | 类型    | 备注                                                                                                                                    |
| --- | -------------------- | ----- | ------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | `less_switch`        | bool  | 控制小于报警逻辑的开关                                                                                                                           |
| 2   | `less_count_person`  | int   | 当‘less_switch’为true时，表示此时算法引入的是小于逻辑，即目标区域人数小于less_count_person时，发生报警，例如less_count_person=1时，则未检测到人发生报警                                |
| 3   | `equal_switch`       | bool  | 控制等于报警逻辑的开关                                                                                                                           |
| 4   | `equal_count_person` | int   | 当‘equal_switch’为true时，表示此时算法引入的是等于逻辑，即目标区域人数等于equal_count_person时，发生报警，例如equal_count_person=0时，则没有检测到人就发生报警；                          |
| 5   | `threshold_count`    | Int32 | 设置报警阈值，当算法检出人数大于该阈值时，发生报警                                                                                                             |
| 6   | `linger_peroson_num` | Int32 | 区域内徘徊目标的数量阈值                                                                                                                          |
| 7   | `time_thresh`        | Int32 | 徘徊时间阈值                                                                                                                                |
| 8   | `alert_ratio_switch` | bool  | 默认为false，为true时表示打开连续帧投票报警逻辑                                                                                                          |
| 9   | `alert_frame_ratio`  | float | 默认为0.9，表示投赞同票的比例阈值，例如alert_time_thresh=10，alert_frame_ratio=0.9 时,表示在 alert_time_thresh=10长度的投票空间内，收集投票结果，当投票结果中赞同票的比例大于90%时，当前帧才引发报警 |
| 10   | `used_time_visual`  | bool | 可视化需求的开关，为true表示可视化满足下面需求：a.不满足设置的时间阈值（未报警时），在推理图片左上角可视化徘徊的时间，打印（徘徊：XX 秒） b. 大于设置的时间阈值（报警时），报警图片左上角可视化徘徊超时的时间，打印（滞留：XX秒），其中滞留的时间计算方式： 目标实际徘徊时间 - 计时器设置时长|



## 三、输出json



### 不报警

```json
{
    "algorithm_data": {
        "is_alert": false,
        "alert_time_s":10,
        "target_count": 0,
        "target_info": []
    },
    "model_data": {
        "objects": [
            {
                "x": 716,
                "y": 716,
                "height": 646,
                "width": 233,
                "person_id": "1",
                "name": "person"
            },
            {
                "x": 397,
                "y": 397,
                "height": 488,
                "width": 215,
                "person_id": "5",
                "name": "person"
            }
        ]
    }
}
```



### 报警：

```json
{
    "algorithm_data": {
        "is_alert": true,
        "alert_time_s":10,
        "target_count": 2,
        "target_info": [
            {
                "x": 716,
                "y": 716,
                "height": 646,
                "width": 233,
                "person_id": "1",
                "name": "person"
            },
            {
                "x": 397,
                "y": 397,
                "height": 488,
                "width": 215,
                "person_id": "5",
                "name": "person"
               
            }
        ]
    },
    "model_data": {
        "objects": [
            {
                "x": 716,
                "y": 716,
                "height": 646,
                "width": 233,
                "person_id": "1",
                "name": "person"
              
            },
            {
                "x": 397,
                "y": 397,
                "height": 488,
                "width": 215,
                "person_id": "5",
                "name": "person"
  
            }
        ]
    }
}
```



## 字段详解


| 序号  | 字段名称                 | 类型     | 备注                                                                                                                                    |
| --- | -------------------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | `algorithm_data`     | Object | 业务相关的输出信息                                                                                                                             |
| 2   | `is_alert`           | bool   | 报警标识，true告警，false不告警，当roi区域的目标存在时间大于 “alert_time_s”阈值时，发生报警                                                                           |
| 3   | `target_info`        | Array  | 报警信息                                                                                                                                  |
| 4   | `model_data`         | Object | 模型相关的输出信息                                                                                                                             |
| 5   | `objects`            | Array  | 模型输出信息                                                                                                                                |
| 6   | `target_count`       | int    | `target_info`中的对象数目                                                                                                                   |
| 7   | `less_switch`        | bool   | 控制小于报警逻辑的开关                                                                                                                           |
| 8   | `less_count_person`  | int    | 当‘less_switch’为true时，表示此时算法引入的是小于逻辑，即目标区域人数小于less_count_person时，发生报警，例如less_count_person=1时，则未检测到人发生报警                                |
| 9   | `equal_switch`       | bool   | 控制等于报警逻辑的开关                                                                                                                           |
| 10  | `equal_count_person` | int    | 当‘equal_switch’为true时，表示此时算法引入的是等于逻辑，即目标区域人数等于equal_count_person时，发生报警，例如equal_count_person=0时，则没有检测到人就发生报警；                          |
| 11  | `threshold_count`    | Int32  | 设置报警阈值，当算法检出人数大于该阈值时，发生报警                                                                                                             |
| 12  | `linger_peroson_num` | Int32  | 区域内徘徊目标的数量阈值                                                                                                                          |
| 13  | `time_thresh`        | Int32  | 徘徊时间阈值                                                                                                                                |
| 14  | `alert_ratio_switch` | bool   | 默认为false，为true时表示打开连续帧投票报警逻辑                                                                                                          |
| 15  | `alert_frame_ratio`  | float  | 默认为0.9，表示投赞同票的比例阈值，例如alert_time_thresh=10，alert_frame_ratio=0.9 时,表示在 alert_time_thresh=10长度的投票空间内，收集投票结果，当投票结果中赞同票的比例大于90%时，当前帧才引发报警 |


- target_info字段详解:

  | 序号  | 字段名称        | 类型     | 备注                               |
  | --- | ----------- | ------ | -------------------------------- |
  | 1   | `x`         | int    | 识别框左上角X坐标                        |
  | 2   | `y`         | int    | 识别框左上角Y坐标                        |
  | 3   | `width`     | int    | 识别框宽度                            |
  | 4   | `height`    | int    | 识别框高度                            |
  | 5   | `person_id` | String | 检测到告警目标的人员id （备注：属于同一人的人员id是唯一的） |
  | 6   | `name` | String | "peroson" |

- object字段详解:

  | 序号  | 字段名称        | 类型     | 备注                               |
  | --- | ----------- | ------ | -------------------------------- |
  | 1   | `x`         | int    | 识别框左上角X坐标                        |
  | 2   | `y`         | int    | 识别框左上角Y坐标                        |
  | 3   | `width`     | int    | 识别框宽度                            |
  | 4   | `height`    | int    | 识别框高度                            |
  | 5   | `person_id` | String | 检测到告警目标的人员id （备注：属于同一人的人员id是唯一的） |
  | 6   | `name` | String | "peroson",检测模型检测结果全量输出 |
