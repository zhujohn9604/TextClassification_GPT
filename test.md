# 导航规划输入数据描述

本文档描述了 `planner_navi_input` 以及 `highlight_lane` 数据结构，具体说明如下。
映射关系: https://ad-gitlab.nioint.com/ad/data-algorithm/pad/wm-prompt-export/-/blob/master/mapping_config.py?ref_type=heads

## `planner_navi_input` 数据结构

### 基本信息
- **Shape**：`[bsz, 1, 12]`
- **输入时数据类型**：`float32`

### 字段概览

| 字段                     | 原始数据类型      | 范围 / 描述                          |
|--------------------------|---------------|---------------------------------------|
| `road_class`             | `int`         | `[0, 11] + [-1]`，表示道路分类         |
| `road_type`              | `int`         | `[0, 30] + [-1]`，表示道路类型        |
| `main_action`            | `int`         | `[0, 20]`，表示当前主动作      |
| `assist_action`          | `int`         | `[0, 91]`，表示当前副动作      |
| `guide_main_action`      | `int`         | `[0, 20]`，表示引导主动作      |
| `guide_assist_action`    | `int`         | `[0, 91]`，表示引导副动作      |
| `guide_distance`         | `float`       | `float or [-1]` 表示引导距离，`-1` 表示未定义 |
| `traffic_light_direction`| `int`         | `[0, 3] + [-1]`，表示交通信号灯方向  |
| `traffic_light_type`     | `int`         | `[0, 2] + [-1]`，表示交通信号灯类型  |
| `traffic_light_countdown`| `int`         | `[0, 34] + [-1]`，表示信号灯倒计时   |
| `traffic_light_distance` | `float`       | `float or [-1]` 表示交通信号灯的距离          |
| `speedLimit`             | `float`       | `float or [-1]` 表示当前道路的限速值          |


### 字段及特殊值说明
该数据可用于自动驾驶车辆的导航规划，其中：
- `road_class` 和 `road_type` 表示道路类型。
- `main_action` 和 `assist_action` 表示当前时刻的主、副action。
- `guide_main_action`、`guide_assist_action` 和 `guide_distance` 表示引导信号和引导距离。
- `traffic_light_*` 特征提供关于附近交通信号灯的信息，包括方向、类型、倒计时和距离。
- `speedLimit` 表示当前道路的限速值。
- **`-1`**：表示该特征的值未定义或不可用。

### 数据映射规则

#### 1. `road_class` 映射规则
`road_class` 表示道路分类，采用以下映射规则：

| 道路类别             | 映射值 |
|----------------------|--------|
| `R_NONE`             | `0`    |
| `HIGHWAY`            | `1`    |
| `URBANHIGHWAY`       | `2`    |
| `NATIONALROAD`       | `3`    |
| `PROVINCIALROAD`     | `4`    |
| `COUNTRYROAD`        | `5`    |
| `COUNTYROAD`         | `5`    |
| `TOWNROAD`           | `6`    |
| `OTHERROAD`          | `7`    |
| `NONNAVIGATIONROAD`  | `8`    |
| `WALKINGROAD`        | `9`    |
| `FERRY`              | `10`   |
| `R_MAX`              | `11`   |

#### 2. `main_action` 及 `guide_main_action` 映射规则
`main_action` 及 `guide_main_action` 表示当前时刻和引导时刻的导航主信号，采用以下映射规则：

| 信号类型             | 映射值 | 描述                 |
|---------------------|--------|-----------------------|
| `NULL`             | `0`    | `无动作`              |
| `TURN_LEFT`        | `1`    | `左转`                |
| `TURN_RIGHT`       | `2`    | `右转`                |
| `SLIGHT_LEFT`      | `3`    | `向左前方行驶`        |
| `SLIGHT_RIGHT`     | `4`    | `向右前方行驶`        |
| `TURN_HARDLEFT`    | `5`    | `向左后方行驶`        |
| `TURN_HARDRIGHT`   | `6`    | `向右后方行驶`        |
| `UTURN`            | `7`    | `左转调头`            |
| `CONTINUE`         | `8`    | `直行`                |
| `MERGE_LEFT`       | `9`    | `靠左`                |
| `MERGE_RIGHT`      | `10`   | `靠右`                |
| `ENTRY_RING`       | `11`   | `进入环岛`            |
| `LEAVE_RING`       | `12`   | `离开环岛`            |
| `SLOW`             | `13`   | `减速`                |
| `PLUG_CONTINUE`    | `14`   | `插入直行`            |
| `ENTER_BUILDING`   | `15`   | `进入建筑物`          |
| `LEAVE_BUILDING`   | `16`   | `离开建筑物`          |
| `BY_ELEVATOR`      | `17`   | `电梯换层`            |
| `BY_STAIR`         | `18`   | `楼梯换层`            |
| `BY_ESCALATOR`     | `19`   | `扶梯换层`            |
| `COUNT`            | `20`   | `导航主动作最大个数`  |


#### 3. `assist_action` 及 `guide_assist_action` 映射规则
`assist_action` 及 `guide_assist_action` 表示当前时刻和引导时刻的导航副信号，采用以下映射规则：

| 信号类型                 | 映射值 | 描述                         |
|--------------------------|--------|------------------------------|
| `NULL`                   | `0`    | `无动作`                     |
| `ENTRY_MAIN`             | `1`    | `进入主路`                   |
| `ENTRY_SIDEROAD`         | `2`    | `进入辅路`                   |
| `ENTRY_FREEWAY`          | `3`    | `进入高速`                   |
| `ENTRY_SLIP`             | `4`    | `进入匝道`                   |
| `ENTRY_TUNNEL`           | `5`    | `进入隧道`                   |
| `ENTRY_CENTERBRANCH`     | `6`    | `进入中间岔道`               |
| `ENTRY_RIGHTBRANCH`      | `7`    | `进入右岔路`                 |
| `ENTRY_LEFTBRANCH`       | `8`    | `进入左岔路`                 |
| `ENTRY_RIGHTROAD`        | `9`    | `进入右转专用道`             |
| `ENTRY_LEFTROAD`         | `10`   | `进入左转专用道`             |
| `ENTRY_MERGE_CENTER`     | `11`   | `进入中间道路`               |
| `ENTRY_MERGE_RIGHT`      | `12`   | `进入右侧道路`               |
| `ENTRY_MERGE_LEFT`       | `13`   | `进入左侧道路`               |
| `ENTRY_MERGE_RIGHTSILD`  | `14`   | `靠右进入辅路`               |
| `ENTRY_MERGE_LEFTSILD`   | `15`   | `靠左进入辅路`               |
| `ENTRY_MERGE_RIGHTMAIN`  | `16`   | `靠右进入主路`               |
| `ENTRY_MERGE_LEFTMAIN`   | `17`   | `靠左进入主路`               |
| `ENTRY_MERGE_RIGHTRIGHT` | `18`   | `靠右进入右转专用道`         |
| `ENTRY_FERRY`            | `19`   | `到达航道`                   |
| `LEFT_FERRY`             | `20`   | `驶离轮渡`                   |
| `ALONG_ROAD`             | `21`   | `沿当前道路行驶`             |
| `ALONG_SILD`             | `22`   | `沿辅路行驶`                 |
| `ALONG_MAIN`             | `23`   | `沿主路行驶`                 |
| `ARRIVE_EXIT`            | `24`   | `到达出口`                   |
| `ARRIVE_SERVICEAREA`     | `25`   | `到达服务区`                 |
| `ARRIVE_TOLLGATE`        | `26`   | `到达收费站`                 |
| `ARRIVE_WAY`             | `27`   | `到达途径地`                 |
| `ARRIVE_DESTINATION`     | `28`   | `到达目的地`                 |
| `ARRIVE_CHARGINGSTATION` | `29`   | `到达充电站`                 |
| `ENTRY_RINGLEFT`         | `30`   | `沿环岛左转`                 |
| `ENTRY_RINGRIGHT`        | `31`   | `绕环岛右转`                 |
| `ENTRY_RINGCONTINUE`     | `32`   | `绕环岛直行`                 |
| `ENTRY_RINGUTURN`        | `33`   | `绕环岛掉头`                 |
| `SMALLRING_NOTCOUNT`     | `34`   | `小环岛不数个数`             |
| `RIGHT_BRANCH1`          | `35`   | `复杂路口，走右边第一出口`   |
| `RIGHT_BRANCH2`          | `36`   | `复杂路口，走右边第二出口`   |
| `RIGHT_BRANCH3`          | `37`   | `复杂路口，走右边第三出口`   |
| `RIGHT_BRANCH4`          | `38`   | `复杂路口，走右边第四出口`   |
| `RIGHT_BRANCH5`          | `39`   | `复杂路口，走右边第五出口`   |
| `LEFT_BRANCH1`           | `40`   | `复杂路口，走左边第一出口`   |
| `LEFT_BRANCH2`           | `41`   | `复杂路口，走左边第二出口`   |
| `LEFT_BRANCH3`           | `42`   | `复杂路口，走左边第三出口`   |
| `LEFT_BRANCH4`           | `43`   | `复杂路口，走左边第四出口`   |
| `LEFT_BRANCH5`           | `44`   | `复杂路口，走左边第五出口`   |
| `ENTER_ULINE`            | `45`   | `进入调头专用路`             |
| `PASS_CROSSWALK`         | `46`   | `通过人行横道`               |
| `PASS_OVERPASS`          | `47`   | `通过过街天桥`               |
| `PASS_UNDERGROUND`       | `48`   | `通过地下通道`               |
| `PASS_SQUARE`            | `49`   | `通过广场`                   |
| `PASS_PARK`              | `50`   | `通过公园`                   |
| `PASS_STAIRCASE`         | `51`   | `通过扶梯`                   |
| `PASS_LIFT`              | `52`   | `通过直梯`                   |
| `PASS_CABLEWAY`          | `53`   | `通过索道`                   |
| `PASS_SKYCHANNEL`        | `54`   | `通过空中通道`               |
| `PASS_CHANNEL`           | `55`   | `通过建筑物穿越通道`         |
| `PASS_WALKROAD`          | `56`   | `通过行人道路`               |
| `PASS_BOATLINE`          | `57`   | `通过游船路线`               |
| `PASS_SIGHTSEEING_LINE`  | `58`   | `通过观光路线`               |
| `PASS_SKIDWAY`           | `59`   | `通过滑道`                   |
| `PASS_LADDER`            | `60`   | `通过梯子`                   |
| `PASS_SLOP`              | `61`   | `通过斜坡`                   |
| `PASS_BRIDGE`            | `62`   | `通过桥梁`                   |
| `PASS_FERRY`             | `63`   | `通过渡轮`                   |
| `PASS_SUBWAY`            | `64`   | `通过地铁`                   |
| `SOON_ENTER_BUILDING`    | `65`   | `即将进入建筑物`             |
| `SOON_LEAVE_BUILDING`    | `66`   | `即将离开建筑物`             |
| `ENTER_ROUNDABOUT`       | `67`   | `进入环岛`                   |
| `LEAVE_ROUNDABOUT`       | `68`   | `离开环岛`                   |
| `ENTER_PATH`             | `69`   | `进入小路`                   |
| `ENTER_INNER`            | `70`   | `进入内部路`                 |
| `ENTER_LEFT_BRANCH_TWO`  | `71`   | `进入左边第二岔路`           |
| `ENTER_LEFT_BRANCH_THREE`| `72`   | `进入左边第三岔路`           |
| `ENTER_RIGHT_BRANCH_TWO` | `73`   | `进入右边第二岔路`           |
| `ENTER_RIGHT_BRANCH_THREE`| `74`  | `进入右边第三岔路`           |
| `ENTER_GAS_STATION`      | `75`   | `进入加油站道路`             |
| `ENTER_HOUSING_ESTATE`   | `76`   | `进入小区道路`               |
| `ENTER_PARK_ROAD`        | `77`   | `进入园区道路`               |
| `ENTER_OVERHEAD`         | `78`   | `上高架`                     |
| `ENTER_CENTER_BRANCH_OVERHEAD` | `79` | `走中间岔路上高架`       |
| `ENTER_RIGHT_BRANCH_OVERHEAD` | `80`  | `走最右侧岔路上高架`     |
| `ENTER_LEFT_BRANCH_OVERHEAD`  | `81` | `走最左侧岔路上高架`     |
| `ALONE_STRAIGHT`         | `82`   | `沿当前道路直行`             |
| `DOWN_OVERHEAD`          | `83`   | `下高架`                     |
| `ENTER_LEFT_OVERHEAD`    | `84`   | `进入左侧高架`               |
| `ENTER_RIGHT_OVERHEAD`   | `85`   | `进入右侧高架`               |
| `UPTO_BRIDGE`            | `86`   | `进入桥梁`                   |
| `ENTER_PARKING`          | `87`   | `进入停车场`                 |
| `ENTER_OVERPASS`         | `88`   | `进入立交桥`                 |
| `ENTER_BRIDGE`           | `89`   | `进入桥梁`                   |
| `ENTER_UNDERPASS`        | `90`   | `进入地下通道`               |
| `MAX`                    | `91`   | `最大辅助动作`               |

#### 4. `traffic_light_direction` 映射规则
`traffic_light_direction` 表示交通信号灯的方向，采用以下映射规则：

| 方向    | 映射值 |
|---------|--------|
| `左转`   | `0`    |
| `右转`   | `1`    |
| `调头`   | `2`    |
| `直行`   | `3`    |

#### 5. `traffic_light_type` 映射规则
`traffic_light_type` 表示交通信号灯的类型，采用以下映射规则：

| 类型              | 映射值 |
|-------------------|--------|
| `红灯倒计时`       | `0`    |
| `绿灯可通行`       | `1`    |
| `即将变红灯`       | `2`    |

##
## `highlight_lane` 数据结构

### 基本信息
`highlight_lane` 数据由 `highlight_lane_attrs` 和 `highlight_lane_distance` 组成：
- **Shape**：
  - `highlight_lane_attrs`: `[bsz, 10, 6]`
  - `highlight_lane_distance`: `[bsz, 1]`
- **输入时数据类型**：`float32`


### 字段概览

| 字段                                     | 原始数据类型  | 范围 / 描述                                |
|------------------------------------------|---------------|-------------------------------------------|
| `highlight_lane_attrs::recommend`        | `int`         | `[0, 1] + [-1]`，表示是否推荐该车道                 |
| `highlight_lane_attrs::can_drive`        | `int`         | `[0, 1] + [-1]`，表示是否允许车辆行驶               |
| `highlight_lane_attrs::lane_type`        | `int`         | `[0, 11] + [-1]`，表示车道类型              |
| `highlight_lane_attrs::lane_direction`   | `int`         | `[0, 31] + [-1]`，表示车道行驶方向            |
| `highlight_lane_attrs::lane_highlight_direction` | `int`         | `[0, 31] + [-1]`，表示高亮车道行驶方向           |
| `highlight_lane_attrs::lane_change_type` | `int`         | `[0, 4] + [-1]`，表示车道变换类型                 |
| `highlight_lane_distance`               | `float`       | `float or [-1]`，表示与高亮车道的距离，`-1` 表示未定义   |

### 数据映射规则

### 1. `recommend` 映射规则
`recommend` 表示是否推荐，采用以下映射规则：

| 映射值 | 车道类型 |
|--------|----------|
| `0`    | `不推荐` |
| `1`    | `推荐`   |

### 2. `can_drive` 映射规则
`can_drive` 表示是否可以行驶/高亮，采用以下映射规则：

| 映射值 | 车道类型 |
|--------|----------|
| `0`    | `不可`   |
| `1`    | `可`     |

### 3. `lane_type` 映射规则
`lane_type` 表示车道类型，采用以下映射规则：

| 原始值        | 映射值 | 车道类型        |
|-----------------|--------|--------------|
| `000000`        | `0`    | `无效车道`   |
| `000001`        | `1`    | `普通车道`   |
| `000010`        | `2`    | `公交车道`   |
| `000011`        | `3`    | `公交车道文字` |
| `000100`        | `4`    | `可变车道`   |
| `000101`        | `5`    | `HOV`        |
| `000110`        | `6`    | `潮汐车道文字` |
| `000111`        | `7`    | `潮汐车道前行剪头` |
| `001000`        | `8`    | `潮汐车道叉号` |
| `001001`        | `9`    | `ETC`        |
| `001010`        | `10`   | `专用车道线` |
| `001011`        | `11`   | `省略号`     |

### 4. `lane_direction` 和 `lane_highlight_direction` 映射规则
`lane_direction` 和 `lane_highlight_direction` 表示车道和高亮车道行驶方向，采用以下映射规则：

| 原始值    | 映射值 | 行驶方向                             |
|-----------|--------|------------------------------------|
| `00000`   | `0`    | `None`                             |
| `00001`   | `1`    | `右掉头`                           |
| `00010`   | `2`    | `右转`                             |
| `00011`   | `3`    | `右掉头 + 右转`                    |
| `00100`   | `4`    | `直行`                             |
| `00101`   | `5`    | `右掉头 + 直行`                    |
| `00110`   | `6`    | `右转 + 直行`                      |
| `00111`   | `7`    | `右掉头 + 右转 + 直行`             |
| `01000`   | `8`    | `左转`                             |
| `01001`   | `9`    | `右掉头 + 左转`                    |
| `01010`   | `10`   | `右转 + 左转`                      |
| `01011`   | `11`   | `右掉头 + 右转 + 左转`             |
| `01100`   | `12`   | `直行 + 左转`                      |
| `01101`   | `13`   | `右掉头 + 直行 + 左转`             |
| `01110`   | `14`   | `右转 + 直行 + 左转`               |
| `01111`   | `15`   | `右掉头 + 右转 + 直行 + 左转`      |
| `10000`   | `16`   | `左掉头`                           |
| `10001`   | `17`   | `右掉头 + 左掉头`                  |
| `10010`   | `18`   | `右转 + 左掉头`                    |
| `10011`   | `19`   | `右掉头 + 右转 + 左掉头`           |
| `10100`   | `20`   | `直行 + 左掉头`                    |
| `10101`   | `21`   | `右掉头 + 直行 + 左掉头`           |
| `10110`   | `22`   | `右转 + 直行 + 左掉头`             |
| `10111`   | `23`   | `右掉头 + 右转 + 直行 + 左掉头`    |
| `11000`   | `24`   | `左转 + 左掉头`                    |
| `11001`   | `25`   | `右掉头 + 左转 + 左掉头`           |
| `11010`   | `26`   | `右转 + 左转 + 左掉头`             |
| `11011`   | `27`   | `右掉头 + 右转 + 左转 + 左掉头`    |
| `11100`   | `28`   | `直行 + 左转 + 左掉头`             |
| `11101`   | `29`   | `右掉头 + 直行 + 左转 + 左掉头`    |
| `11110`   | `30`   | `右转 + 直行 + 左转 + 左掉头`      |
| `11111`   | `31`   | `右掉头 + 右转 + 直行 + 左转 + 左掉头` |

### 5. `lane_change_type` 映射规则
`lane_change_type` 表示车道变换类型，采用以下映射规则：

| 原始值    | 映射值 | 车道变换类型     |
|-------------|--------|----------|
| `0000`      | `0`    | `无`     |
| `1000`      | `1`    | `左侧拓展` |
| `0100`      | `2`    | `左侧收窄` |
| `0010`      | `3`    | `右侧收窄` |
| `0001`      | `4`    | `右侧拓展` |



