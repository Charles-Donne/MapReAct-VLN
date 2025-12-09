"""
动作执行提示词模板
==================
用于VLM低层动作决策的提示词模板

动作参数与interactive_navigation保持一致：
- TURN_LEFT/RIGHT: 30°（12步×30°=360°）
- MOVE_FORWARD: 0.25m
"""

ACTION_EXECUTION_PROMPT = """You are the action execution module for Vision-Language Navigation. Analyze the environment and decide the next action.

# Current Subtask
**Destination**: {subtask_destination}
**Instruction**: {subtask_instruction}

# Progress Summary
{progress_summary}

# Visual Observations

You are provided with 3 images:

**IMAGE 1: First-person RGB View** - Current facing direction view
**IMAGE 2: Object Detection View** - Detected objects with bounding boxes (landmark: {detected_landmarks})
**IMAGE 3: Local Semantic Map** - Nearby region top-down view

# Local Map

**Map Orientation**: 
- Top of map = Agent's Front direction
- Map rotates with agent - front is always up
- Agent is at center

**Color Legend**:
- **White**: Unexplored/unknown areas
- **Black**: Obstacles (walls, furniture) - AVOID
- **Green**: Floor areas (safe to navigate) - OK TO MOVE
- **Orange line**: Recent trajectory 
- **Red arrow at center**: Agent position and facing direction (arrow = Front)
- **Blue semi-circle**: Current field of view
  - Opening direction = Front view
  - Objects within blue region are visible in IMAGE 1

# Your Task

Analyze the 3 images to decide the next action for collision avoidance and navigation.

**Decision Process**:
1. **RGB View**: What do you see? Where is the destination?
2. **Detection View**: Are there relevant landmarks detected?
3. **Local Map**: 
   - Check immediate path ahead (black = obstacle)
   - Verify direction to destination
   - Plan collision-free path
4. **Distance Estimation**: How far to destination? (e.g., "~3m", "<0.5m")
5. **Action Decision**: Choose safest action toward destination

**STOP Conditions** (ALL required):
- Moved ≥2 times
- Destination within 0.5m

**Safety Priority**: Avoid obstacles shown as black regions on local map

# Available Actions
- MOVE_FORWARD: Move {move_distance}m forward
- TURN_LEFT: Rotate {turn_angle}° counterclockwise
- TURN_RIGHT: Rotate {turn_angle}° clockwise
- STOP: Declare arrival at subtask destination

# Output Format (JSON only)

{{
    "reasoning": "Logic: (1) Destination location and distance (2) Movement count (3) Action decision",
    "action": "MOVE_FORWARD" | "TURN_LEFT" | "TURN_RIGHT" | "STOP",
    "progress_summary": "Updated action history for current subtask"
}}

# Examples

**Ex1 - Clear path ahead**
{{
    "reasoning": "Local map shows safe green floor ahead. Destination visible. Move forward.",
    "action": "MOVE_FORWARD",
    "progress_summary": "Moved forward 1x toward doorway"
}}

**Ex2 - Obstacle detected**
{{
    "reasoning": "Local map shows black obstacle directly ahead. Must turn to find clear path.",
    "action": "TURN_RIGHT",
    "progress_summary": "Turned right to avoid chair obstacle"
}}

**Ex3 - Approaching destination**
{{
    "reasoning": "Movement: 3, Distance: ~1m. Local map clear. Continue approach.",
    "action": "MOVE_FORWARD",
    "progress_summary": "Moved forward 4x approaching sofa"
}}

**Ex4 - At destination**
{{
    "reasoning": "Movement: 4 (✓≥2), Distance: <0.5m (✓), Fills view (✓). ALL MET.",
    "action": "STOP",
    "progress_summary": "Moved forward 4x, reached sofa"
}}

**Critical Rules**:
- Move ≥2 times before STOP
- STOP only when distance ≤0.5m
- When uncertain, MOVE_FORWARD
"""


def get_action_execution_prompt(subtask_destination: str,
                                subtask_instruction: str,
                                turn_angle: float,
                                move_distance: float,
                                progress_summary: str = "",
                                detected_landmarks: str = None) -> str:
    """
    获取动作执行提示词
    
    Args:
        subtask_destination: 子任务目的地
        subtask_instruction: 子任务指令
        turn_angle: 转向角度（度）- 默认30°
        move_distance: 前进距离（米）- 默认0.25m
        progress_summary: 当前子任务进度摘要
        detected_landmarks: 已检测到的landmark类别字符串
        
    Returns:
        格式化的提示词字符串
    """
    if not detected_landmarks:
        detected_landmarks = "No landmarks detected yet"
        
    return ACTION_EXECUTION_PROMPT.format(
        subtask_destination=subtask_destination,
        subtask_instruction=subtask_instruction,
        turn_angle=turn_angle,
        move_distance=move_distance,
        progress_summary=progress_summary if progress_summary else "(Just started - no actions yet)",
        detected_landmarks=detected_landmarks
    )
