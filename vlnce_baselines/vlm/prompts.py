"""
VLM规划提示词模板
================
用于LLM高层规划的提示词模板
"""

# 初始规划提示词 - 在任务开始时生成第一个子任务
INITIAL_PLANNING_PROMPT = """You are a Vision-Language Navigation planning module. Analyze the environment and design the next navigation subtask.

# Navigation Task
{instruction}

# Visual Observations
You are provided with 4 first-person RGB views and 2 bird's-eye view maps:

**IMAGE 1: Front View (0°)**
**IMAGE 2: Left View (90°)**
**IMAGE 3: Back View (180°)**
**IMAGE 4: Right View (270°)** 
**IMAGE 5: Global Semantic Map** - Top-down view of full explored area
**IMAGE 6: Local Semantic Map** - Top-down view of nearby region (focused on agent)

# Map Interpretation Guide

**Map Orientation**: 
- Top of map = Agent's current Front direction (IMAGE 1)
- Map rotates with agent - front is always up

**Global Map**:
- **White**: Unexplored/unknown areas
- **Black**: Obstacles (walls, furniture, barriers, furniture)
- **Green**: Confirmed floor areas (safe to navigate)
- **Orange line**: Trajectory from subtask start to current position
- **Red circle with arrow**: Current position, arrow points to Front direction
- **Purple markers with labels**: Detected landmark objects: {detected_landmarks}
  
**Local Map** (zoomed view around agent, same color legend as Global Map):
- Shows finer details in immediate vicinity for precise navigation
- **Blue semi-circle**: Agent's current field of view (Front direction visibility range)
  - The opening of the semi-circle indicates Front view direction
  - Objects within this blue region are currently visible in IMAGE 1 (Front View)
- Better for planning nearby movements and obstacle avoidance

# Your Task

1. **Analyze environment**: Use 4-directional views + semantic map to identify landmarks and obstacles
2. **Plan subtask**: Break down global task into achievable intermediate waypoints
3. **Provide instructions**: Action sequence starting from Front view using concrete landmarks

**Available Actions**: {action_space}

# Output Format (JSON only)

{{
    "current_area_type": "Room type (e.g., 'living room', 'bedroom', 'hallway', 'kitchen')",
    "current_location": "A brief description of the current environment (1-2 sentences)",
    "waypoint_sequence": "A(Current) → B → C → ... → Goal",
    "subtask_destination": "Next immediate waypoint",
    "subtask_instruction": "Step-by-step actions from Front view",
    "subtask_landmark": "Key landmark to note in the sub-instruction (e.g., 'cabinet', 'door', 'bed') for map marking",
    "completion_criteria": {{
        "landmark_detection": "What landmark should be detected at destination (e.g., 'cabinet visible and detected in any view')",
        "destination_reached": "Whether arrived at specified location (e.g., 'reached kitchen entrance', 'arrived at cabinet area')",
        "spatial_relationship": "Trajectory and orientation check (e.g., 'orange trajectory reaches target area', 'agent facing towards cabinet')"
    }},
    "is_final_subtask": false,
    "reasoning": "Logic: (1) Environment analysis, (2) Waypoint selection, (3) Action plan"
}}

**Critical Requirements**:
- Start all actions from Front view
"""


# 验证和重规划提示词 - 验证子任务完成并生成下一步规划
VERIFICATION_REPLANNING_PROMPT = """You are a Vision-Language Navigation verification module. Verify subtask completion and plan the next navigation step.

# Navigation Task
{instruction}

# Previous Subtask Context
**Waypoint Sequence**: {waypoint_sequence}
**Subtask Destination**: {subtask_destination}
**Subtask Instruction**: {subtask_instruction}
**Completion Criteria**: {completion_criteria}

# Visual Observations
The agent has performed a 360° scan. You are provided with 4 first-person RGB views and 2 bird's-eye view maps:

**IMAGE 1: Front View (0°)**
**IMAGE 2: Left View (90°)**
**IMAGE 3: Back View (180°)**
**IMAGE 4: Right View (270°)**
**IMAGE 5: Global Semantic Map** - Top-down view of full explored area (updated)
**IMAGE 6: Local Semantic Map** - Top-down view of nearby region (focused on agent)

# Map Interpretation Guide

**Map Orientation**: 
- Top of map = Agent's current Front direction (IMAGE 1)
- Map rotates with agent - front is always up

**Global Map**:
- **White**: Unexplored/unknown areas
- **Black**: Obstacles (walls, furniture, barriers)
- **Green**: Confirmed floor areas (safe to navigate)
- **Orange line**: Trajectory from subtask start to current position
- **Red circle with arrow**: Current position, arrow points to Front direction
- **Purple markers with labels**: Detected landmark objects: {detected_landmarks}
  
**Local Map** (zoomed view around agent, same color legend as Global Map):
- Shows finer details in immediate vicinity for precise navigation
- **Blue semi-circle**: Agent's current field of view (Front direction visibility range)
  - The opening of the semi-circle indicates Front view direction
  - Objects within this blue region are currently visible in IMAGE 1 (Front View)
- Better for planning nearby movements and obstacle avoidance

# Your Task

1. **Verify completion**: Compare current observations with completion_criteria (landmark detection, destination arrival, trajectory/orientation)
2. **Make decision**: 
   - **is_completed = true**: Subtask finished → plan NEXT waypoint
   - **is_completed = false**: Not finished → continue SAME subtask
3. **Plan next step**: If completed, update waypoint_sequence and define new subtask; if not, adjust current subtask

**Available Actions**: {action_space}

# Output Format (JSON only)

{{
    "is_completed": true/false,
    "current_area_type": "Room type (e.g., 'living room', 'bedroom', 'hallway', 'kitchen')",
    "current_location": "Concise position (1 sentence with 1-2 key landmarks)",
    "current_observation": "Brief spatial description (1-2 sentences: visible objects, layout)",
    "waypoint_sequence": "A(✓) → B(Current) → C → Goal",
    "subtask_destination": "Next waypoint or same if not completed",
    "subtask_instruction": "Step-by-step actions from Front view",
    "subtask_landmark": "Key landmark at destination (e.g., 'cabinet', 'door', 'bed') for map marking",
    "completion_criteria": {{
        "landmark_detection": "What landmark should be detected at destination (e.g., 'cabinet visible and detected in any view')",
        "destination_reached": "Whether arrived at specified location (e.g., 'reached kitchen entrance', 'arrived at cabinet area')",
        "spatial_relationship": "Trajectory and orientation check (e.g., 'orange trajectory reaches target area', 'agent facing towards cabinet')"
    }},
    "is_final_subtask": true/false,
    "reasoning": "Logic: (1) Completion verification, (2) Progress analysis, (3) Next action plan"
}}

**Critical Requirements**:
- Verify **completion_criteria** 3 checks: (1) landmark detected in 4 views, (2) destination arrived, (3) trajectory/orientation on map
- Analyze all 4 views for 360° understanding
- Mark completed waypoints with (✓)
- Start all actions from Front view
"""


def get_initial_planning_prompt(instruction: str, 
                               direction_names: list, 
                               action_space: str,
                               detected_landmarks: str = None) -> str:
    """
    获取初始规划提示词
    
    Args:
        instruction: 完整导航指令
        direction_names: 方向名称列表（4个方向）
        action_space: 动作空间描述
        detected_landmarks: 已检测到的landmark类别字符串
        
    Returns:
        格式化的提示词字符串
    """
    if not detected_landmarks:
        detected_landmarks = "No landmarks detected yet"
    
def get_verification_replanning_prompt(instruction: str,
                                       waypoint_sequence: str,
                                       subtask_destination: str,
                                       subtask_instruction: str,
                                       completion_criteria: str,
                                       direction_names: list,
                                       action_space: str,
                                       detected_landmarks: str = None) -> str:
    """
    获取验证和重规划提示词
    
    Args:
        instruction: 完整导航指令
        waypoint_sequence: 当前路径点序列
        subtask_destination: 当前子任务目的地
        subtask_instruction: 当前子任务指令
        completion_criteria: 完成条件
        direction_names: 方向名称列表
        action_space: 动作空间描述
        detected_landmarks: 已检测到的landmark类别字符串
        
    Returns:
        格式化的提示词字符串
    """
    if not detected_landmarks:
        detected_landmarks = "No landmarks detected yet"
    
    return VERIFICATION_REPLANNING_PROMPT.format(
        instruction=instruction,
        waypoint_sequence=waypoint_sequence,
        subtask_destination=subtask_destination,
        subtask_instruction=subtask_instruction,
        completion_criteria=completion_criteria,
        direction_names=', '.join(direction_names),
        action_space=action_space,
        detected_landmarks=detected_landmarks
    )