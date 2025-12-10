# Step 命名逻辑分析

## 问题背景
Map: 14 张文件，但查找 step-12.png 失败。

## 时间线分析

### 初始化 (reset_episode)
```python
self.envs.reset()
self.current_step = 0  # ✅
```
- **不保存文件**
- current_step = 0

### 环视建图 (look_around_and_collect)
```python
for step in range(12):  # step = 0, 1, 2, ..., 11
    envs.step(TURN_LEFT)  # 执行动作
    save_step_visualization(step=step)  # 保存文件
```

**保存的文件**:
- Iteration 0: 执行动作 → 保存 `step-0.png`
- Iteration 1: 执行动作 → 保存 `step-1.png`
- ...
- Iteration 11: 执行动作 → 保存 `step-11.png`

**结果**: 12 张文件 (step-0 到 step-11)

```python
self.current_step = 11  # ✅ 指向最后保存的文件
```

### 正常导航 (step 方法)
```python
envs.step(action)  # 执行动作
save_step_visualization(step=self.current_step)  # 保存
self.current_step += 1  # 递增
```

**示例**:
- current_step=11 → 执行动作 → 保存 `step-11.png` → current_step=12
- current_step=12 → 执行动作 → 保存 `step-12.png` → current_step=13

## 命名约定

### 文件命名格式
所有文件使用统一格式: `step-{step}.png`

### 目录结构
```
results/episode_{id}/
├── rgb/
│   ├── step-0.png
│   ├── step-1.png
│   └── ...
├── global_map/
│   ├── step-0.png
│   ├── step-1.png
│   └── ...
├── local_map/
│   ├── step-0.png
│   ├── step-1.png
│   └── ...
└── detection/
    ├── step-0.png
    ├── step-1.png
    └── ...
```

## Step 值含义

**step = N** 表示：
- 已执行 N+1 次动作（包括环视）
- 这是第 N 次动作后的状态
- 保存的文件名为 `step-N.png`

## 验证

### 环视结束后
- 文件: step-0.png ~ step-11.png (12张)
- current_step: 11
- 查找地图: step-11.png ✅

### 第一次正常导航
- current_step: 11
- 执行动作 MOVE_FORWARD
- 保存: step-11.png (会覆盖环视的最后一张？)
- current_step: 12

⚠️ **潜在问题**: step-11.png 会被覆盖！

## 解决方案

### 方案1: 环视后递增 current_step
```python
# look_around_and_collect 结束后
self.current_step = 12  # 下次保存会是 step-12.png
```

但这样需要在查找地图时使用 `current_step - 1`:
```python
last_step = self.current_step - 1  # 11
global_map = f'step-{last_step}.png'  # step-11.png
```

### 方案2: 保持 current_step = 11 (当前实现)
```python
# look_around_and_collect 结束后
self.current_step = 11  # 指向最后保存的文件
```

查找地图时直接使用:
```python
global_map = f'step-{self.current_step}.png'  # step-11.png ✅
```

但第一次正常导航会覆盖 step-11.png！

### 方案3: 修改 step() 方法逻辑 (推荐)
```python
# step() 方法中
self.current_step += 1  # 先递增
save_step_visualization(step=self.current_step)  # 再保存
```

这样：
- 环视保存 step-0 到 step-11
- current_step = 11
- 第一次导航: current_step=12 → 保存 step-12.png ✅

但这会改变整个系统的逻辑！

## 推荐方案

**采用方案2，但在环视后递增一次**:

```python
# look_around_and_collect 结束后
self.current_step = 12  # 表示环视完成，准备开始导航

# generate_initial_subtask 中查找地图
last_step = 11  # 或 self.current_step - 1
global_map = f'step-{last_step}.png'  # step-11.png
```

这样：
- ✅ 环视保存 step-0 到 step-11
- ✅ 查找地图时使用 step-11.png
- ✅ 第一次导航保存 step-12.png（不覆盖）
- ✅ 符合直觉：current_step 表示"即将执行的步数"

## 当前实现 (修复后)

```python
# look_around_and_collect
for step in range(12):
    save_step_visualization(step=step)  # step-0 到 step-11
self.current_step = 11  # ✅

# generate_initial_subtask  
global_map = f'step-{self.current_step}.png'  # step-11.png ✅

# 第一次 step_with_vlm
step(action)  # 内部会保存 step-11.png 然后 current_step=12
```

⚠️ **问题**: step-11.png 会被第一次导航覆盖！
