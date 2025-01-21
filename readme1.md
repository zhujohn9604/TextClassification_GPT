# Planner Navigation Input Data Description

This document provides a detailed description of the `planner_navi_input` data structure, which is a single-row, 12-column vector of type `float32`. Each column represents a specific feature, as described below.

## Data Structure

### General Information
- **Shape**: `[1, 12]`
- **Data Type**: `float32`

### Columns and Their Descriptions

| Column Name              | Data Type      | Range / Description                  |
|--------------------------|----------------|---------------------------------------|
| `road_class`             | `int`          | `[0, 30] + [-1]`                     |
| `road_type`              | `int`          | `[0, 30] + [-1]`                     |
| `main_action`            | `int`          | `[0, 30] + [-1]`                     |
| `assist_action`          | `int`          | `[0, 50] + [-1]`                     |
| `guide_main_action`      | `int`          | `[0, 30] + [-1]`                     |
| `guide_assist_action`    | `int`          | `[0, 110] + [-1]`                    |
| `guide_distance`         | `float`        | `[-1]` for undefined distances       |
| `traffic_light_direction`| `int`          | `[0, 10] + [-1]`                     |
| `traffic_light_type`     | `int`          | `[0, 10] + [-1]`                     |
| `traffic_light_countdown`| `int`          | `[0, 35] + [-1]`                     |
| `traffic_light_distance` | `float`        | `[-1]` for undefined distances       |
| `speedLimit`             | `float`        | `[-1]` for undefined limits          |

### Explanation of Special Values
- **`-1`**: Indicates an undefined or unavailable value for the respective feature.

## Example Use Case
This dataset can be used for navigation planning in autonomous vehicles, where:
- `road_class`, `road_type`, and `main_action` represent the type of road and the primary action to be taken.
- `assist_action` and `guide_*` columns provide additional guidance for the navigation system.
- `traffic_light_*` features provide information about nearby traffic lights, including their direction, type, countdown timer, and distance.
- `speedLimit` indicates the current speed limit on the road.

## Notes
- Ensure that special values (e.g., `-1`) are handled appropriately during data preprocessing or model training.
- Features such as `guide_distance` and `traffic_light_distance` are provided as floating-point values, while other categorical features are represented as integers.

## Contact
For any questions or clarifications, feel free to reach out to the dataset maintainer.
