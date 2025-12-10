# Step Numbering Logic - 步骤编号逻辑

## Overview - 概述

MapReAct-VLN系统使用统一的步骤编号来管理所有可视化文件（RGB、地图、检测等）。

## Step Counter Flow - 步骤计数器流程

### 初始化
- Episode开始时：`current_step = 0`

### 环视阶段 (Look Around)
```python
for step in range(12):
    execute_action(TURN_LEFT)
    save_files(step)  # 保存 step-0 到 step-11
    
# 循环结束后
current_step = 12  # 下次动作将保存为 step-12
```

**结果**：
- 保存了12个文件：`step-0.png` 到 `step-11.png`
- `current_step = 12`（指向下一个要保存的步骤）

### 导航阶段 (Navigation Loop)

每次迭代：

1. **VLM决策** (`execute_action_with_vlm()`)
   - 使用 `current_step - 1` 查找已保存的文件
   - 例如：`current_step = 12`，则读取 `step-11.png`
   - VLM基于这些文件决策下一个动作

2. **执行动作** (`step_with_vlm()`)
   - 调用 `self.step(action)` 执行动作
   - 保存新文件：`step-{current_step}.png`
   - 例如：`current_step = 12`，保存 `step-12.png`
   - 自动递增：`current_step += 1` → `current_step = 13`

## File Naming Convention - 文件命名约定

所有可视化文件使用统一的步骤编号：

```
data/vlm_navigation/episode_{id}/
├── rgb/
│   ├── step-0.png    # 第1次TURN_LEFT后的RGB
│   ├── step-1.png    # 第2次TURN_LEFT后的RGB
│   └── step-11.png   # 第12次TURN_LEFT后的RGB
│   └── step-12.png   # 导航阶段第1次动作后的RGB
├── global_map/
│   ├── step-0.png    # 第1次TURN_LEFT后的全局地图
│   └── step-11.png   # 第12次TURN_LEFT后的全局地图
│   └── step-12.png   # 导航阶段第1次动作后的全局地图
├── local_map/
│   └── step-{N}.png  # 局部地图
└── detection/
    └── step-{N}.png  # 目标检测可视化
```

## Key Points - 关键点

1. **step-0 的含义**：
   - ✅ 第一次动作执行后的状态
   - ❌ 不是初始状态（执行动作前）

2. **current_step 的含义**：
   - 总是指向**下一个**要保存的步骤编号
   - 读取文件时使用 `current_step - 1`

3. **文件查找规则**：
   - VLM决策前：读取 `step-{current_step - 1}.png`
   - 动作执行后：保存 `step-{current_step}.png`，然后递增

## Example Timeline - 示例时间线

```
初始化：current_step = 0

环视阶段：
  iteration 0: TURN_LEFT → save step-0.png → current_step = 1
  iteration 1: TURN_LEFT → save step-1.png → current_step = 2
  ...
  iteration 11: TURN_LEFT → save step-11.png → current_step = 12

导航阶段：
  Turn 1:
    - VLM reads step-11.png (current_step - 1 = 11)
    - VLM decides: MOVE_FORWARD
    - Execute MOVE_FORWARD → save step-12.png → current_step = 13
    
  Turn 2:
    - VLM reads step-12.png (current_step - 1 = 12)
    - VLM decides: TURN_RIGHT
    - Execute TURN_RIGHT → save step-13.png → current_step = 14
```

## Code References - 代码引用

### Look Around Loop
```python
# vlnce_baselines/vlm_navigation_controller.py:183-227
for step in range(12):
    execute_turn_left()
    save_visualization(step=step)
    
self.current_step = 12  # 设置为下一个步骤
```

### Generate Initial Subtask
```python
# vlnce_baselines/vlm_navigation_controller.py:328-337
last_saved_step = self.current_step - 1  # 12
global_map = f'step-{last_saved_step}.png'  # step-11.png
```

### Execute Action
```python
# vlnce_baselines/vlm_navigation_controller.py:572-600
last_step = self.current_step - 1
fp_image = f'step-{last_step}.png'  # 读取上一步保存的文件
detection_image = f'step-{last_step}.png'
local_map = f'step-{last_step}.png'
```

### Step with VLM
```python
# vlnce_baselines/vlm_navigation_controller.py:646
result = self.step(action, save_vis)  # 保存 step-{current_step}.png
# 然后父类自动递增 current_step
```

## Troubleshooting - 故障排除

### "Global map not found: step-12.png"

**原因**：在环视结束后，`current_step = 12`，但最后保存的文件是 `step-11.png`

**解决方案**：读取文件时使用 `current_step - 1`

```python
# ❌ 错误
map_path = f'step-{self.current_step}.png'

# ✅ 正确
map_path = f'step-{self.current_step - 1}.png'
```

### "Step numbering mismatch"

**检查清单**：
1. 环视循环是否是 `range(12)`（不是 `range(1, 13)`）
2. 环视结束后是否设置 `current_step = 12`
3. VLM决策时是否使用 `current_step - 1` 读取文件
4. 父类 `step()` 是否在保存后递增 `current_step`
