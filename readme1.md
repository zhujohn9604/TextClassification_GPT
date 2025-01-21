# 导航规划输入数据描述

本文档详细描述了 `planner_navi_input` 数据结构，这是一个包含 12 列的单行向量，数据类型为 `float32`。每一列表示一个特定的特征，具体说明如下。

## `planner_navi_input` 数据结构

### 基本信息
- **Shape**：`[bsz, 1, 12]`
- **输入时数据类型**：`float32`

### 字段概览

| 字段                     | 原始数据类型      | 范围 / 描述                          |
|--------------------------|---------------|---------------------------------------|
| `road_class`             | `int`         | `[0, 9] + [-1]`，表示道路分类         |
| `road_type`              | `int`         | `[0, 30] + [-1]`，表示道路类型        |
| `main_action`            | `int`         | `[0, 20]`，表示当前主动作      |
| `assist_action`          | `int`         | `[0, 91]`，表示当前副动作      |
| `guide_main_action`      | `int`         | `[0, 20]`，表示引导主动作      |
| `guide_assist_action`    | `int`         | `[0, 91]`，表示引导副动作      |
| `guide_distance`         | `float`       | `[-1]` 表示引导距离，`-1` 表示未定义 |
| `traffic_light_direction`| `int`         | `[0, 3] + [-1]`，表示交通信号灯方向  |
| `traffic_light_type`     | `int`         | `[0, 2] + [-1]`，表示交通信号灯类型  |
| `traffic_light_countdown`| `int`         | `[0, 34] + [-1]`，表示信号灯倒计时   |
| `traffic_light_distance` | `float`       | `[-1]` 表示交通信号灯的距离          |
| `speedLimit`             | `float`       | `[-1]` 表示当前道路的限速值          |


### 字段及特殊值说明
该数据可用于自动驾驶车辆的导航规划，其中：
- `road_class` 和 `road_type` 表示道路类型。
- `main_action` 和 `assist_action` 表示当前时刻的主、副action。
- `guide_main_action`、`guide_assist_action` 和 `guide_distance` 表示引导信号和引导距离。
- `traffic_light_*` 特征提供关于附近交通信号灯的信息，包括方向、类型、倒计时和距离。
- `speedLimit` 表示当前道路的限速值。
- **`-1`**：表示该特征的值未定义或不可用。

### 数据映射规则

#### `road_class` 映射规则
`road_class` 表示道路分类，采用以下映射规则：

| 道路类别           | 映射值 |
|--------------------|--------|
| `R_NONE`          | `0`    |
| `HIGHWAY`         | `1`    |
| `NATIONALROAD`    | `2`    |
| `PROVINCIALROAD`  | `3`    |
| `COUNTRYROAD`     | `4`    |
| `TOWNROAD`        | `5`    |
| `NONNAVIGATIONROAD` | `6`  |
| `WALKINGROAD`     | `7`    |
| `FERRY`           | `8`    |
| `R_MAX`           | `9`    |

#### `traffic_light_direction` 映射规则
`traffic_light_direction` 表示交通信号灯的方向，采用以下映射规则：

| 方向    | 映射值 |
|---------|--------|
| `左转`   | `0`    |
| `右转`   | `1`    |
| `调头`   | `2`    |
| `直行`   | `3`    |

#### `traffic_light_type` 映射规则
`traffic_light_type` 表示交通信号灯的类型，采用以下映射规则：

| 类型              | 映射值 |
|-------------------|--------|
| `红灯倒计时`       | `0`    |
| `绿灯可通行`       | `1`    |
| `即将变红灯`       | `2`    |

### 注意事项
- 请在数据预处理时确保输入数据与上述映射规则一致。
- 如果值未定义或不可用，请使用 `-1` 表示。
