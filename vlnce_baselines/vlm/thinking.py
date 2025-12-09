"""
LLM规划模块
===========
高层规划：分析环境生成子任务
"""
from typing import Dict, List, Tuple, Optional
from vlnce_baselines.vlm.api_client import APIConfig, BaseAPIClient
from vlnce_baselines.vlm.prompts import (
    get_initial_planning_prompt,
    get_verification_replanning_prompt
)


class LLMPlanner(BaseAPIClient):
    """LLM规划器 - 负责子任务生成和验证"""
    
    REQUIRED_FIELDS_INITIAL = ['subtask_destination', 'subtask_destination_landmark',
                               'subtask_instruction', 'completion_criteria',
                               'current_observation', 'current_room_type']
    REQUIRED_FIELDS_VERIFY = ['is_completed', 'subtask_destination', 'subtask_destination_landmark',
                              'subtask_instruction', 'completion_criteria',
                              'current_observation', 'current_room_type']
    
    # completion_criteria 子字段（嵌套结构）
    REQUIRED_CRITERIA_FIELDS = ['landmark_detection', 'destination_reached', 'spatial_relationship']
    
    def __init__(self, config_path: str = "vlnce_baselines/vlm/llm_config.yaml", 
                 action_space: str = None):
        """
        初始化LLM规划器
        
        Args:
            config_path: LLM配置文件路径
            action_space: 动作空间描述（如 "MOVE_FORWARD (0.25m), TURN_LEFT (30°), ..."）
        """
        config = APIConfig(config_path)
        super().__init__(config)
        
        # 默认动作空间与interactive_navigation一致
        self.action_space = action_space or "MOVE_FORWARD (0.25m), TURN_LEFT (30°), TURN_RIGHT (30°), STOP"
        
        print(f"✓ LLM Planner initialized")
        print(f"  Model: {self.config.model}")
        print(f"  Action space: {self.action_space}")
    
    def validate_response(self, response: Dict, mode: str = 'initial') -> bool:
        """验证响应字段"""
        required = self.REQUIRED_FIELDS_INITIAL if mode == 'initial' else self.REQUIRED_FIELDS_VERIFY
        
        # 先验证基础字段
        if not self.validate_fields(response, required):
            return False
        
        # 验证completion_criteria嵌套字段
        criteria = response.get('completion_criteria')
        if criteria and isinstance(criteria, dict):
            for field in self.REQUIRED_CRITERIA_FIELDS:
                if field not in criteria:
                    print(f"⚠️ Missing completion_criteria field: {field}")
                    return False
        else:
            print(f"⚠️ completion_criteria should be a dict with fields: {self.REQUIRED_CRITERIA_FIELDS}")
            return False
        
        return True
    
    def generate_initial_subtask(self,
                                instruction: str,
                                observation_images: List[str],
                                direction_names: List[str],
                                global_map_image: str,
                                local_map_image: str = None,
                                detected_landmarks: List[str] = None) -> Optional[Dict]:
        """
        生成初始子任务
        
        Args:
            instruction: 完整导航指令
            observation_images: 4方向图像路径列表 [前, 左, 后, 右]
            direction_names: 方向名称列表 ['Front (0°)', 'Left (90°)', 'Back (180°)', 'Right (270°)']
            global_map_image: 全局语义地图路径（global_map/step-N.png）- 必需
            local_map_image: 局部语义地图路径（local_map/step-N.png）- 可选
            detected_landmarks: 已检测到的landmark类别列表 - 可选
            
        Returns:
            LLM响应字典或None
        """
        if not global_map_image:
            print("✗ Error: global_map_image is required")
            return None
        
        # 格式化检测到的landmark信息
        landmarks_str = None
        if detected_landmarks:
            landmarks_str = f"Detected landmarks: {', '.join(sorted(detected_landmarks))}"
        
        prompt = get_initial_planning_prompt(
            instruction, 
            direction_names, 
            self.action_space,
            detected_landmarks=landmarks_str
        )
        
        # 组合图像：4方向观察 + 全局地图 + 局部地图（如果有）
        images = observation_images.copy()
        images.append(global_map_image)
        
        if local_map_image:
            images.append(local_map_image)
            print(f"  📍 Images: 4 directions + Global map + Local map")
        else:
            print(f"  📍 Images: 4 directions + Global map")
        
        response = self.call_api(prompt, images)
    def verify_and_replan(self,
                         instruction: str,
                         current_subtask: Dict,
                         observation_images: List[str],
                         direction_names: List[str],
                         global_map_image: str,
                         local_map_image: str = None,
                         detected_landmarks: List[str] = None) -> Tuple[Optional[Dict], bool]:
        """
        验证子任务完成并规划下一步
        
        Args:
            instruction: 完整导航指令
            current_subtask: 当前子任务字典
            observation_images: 4方向图像路径列表（当前位置重新环视获得）
            direction_names: 方向名称列表
            global_map_image: 更新后的全局语义地图路径 - 必需
            local_map_image: 更新后的局部语义地图路径 - 可选
            detected_landmarks: 已检测到的landmark类别列表 - 可选
            
        Returns:
            (response字典, is_completed标志)
        """
        if not global_map_image:
            print("✗ Error: global_map_image is required")
            return None, False
        
        # 获取当前子任务信息
        waypoint_sequence = current_subtask.get('waypoint_sequence', 'Unknown')
        subtask_destination = current_subtask.get('subtask_destination', 'Unknown')
        subtask_instruction = current_subtask.get('subtask_instruction', 'Unknown')
        completion_criteria = current_subtask.get('completion_criteria', 'Unknown')
        
        # 格式化检测到的landmark信息
        landmarks_str = None
        if detected_landmarks:
            landmarks_str = f"Detected landmarks: {', '.join(sorted(detected_landmarks))}"
        
        prompt = get_verification_replanning_prompt(
            instruction,
            waypoint_sequence,
            subtask_destination,
            subtask_instruction,
            completion_criteria,
            direction_names,
            self.action_space,
            detected_landmarks=landmarks_str
        )
        
        # 组合图像：当前位置4方向 + 全局地图 + 局部地图（如果有）
        images = observation_images.copy()
        images.append(global_map_image)
        
        if local_map_image:
            images.append(local_map_image)
            print(f"  📍 Images: 4 directions (updated) + Global map + Local map")
        else:
            print(f"  📍 Images: 4 directions (updated) + Global map")
        
        response = self.call_api(prompt, images)
        images = observation_images.copy()
        if map_image:
            images.append(map_image)
            print(f"  📍 Images: 4 directions (updated) + Global map (with trajectory)")
        else:
            print(f"  📍 Images: 4 directions only")
        
        response = self.call_api(prompt, images)
        
        if response and self.validate_response(response, mode='verify'):
            is_completed = response.get('is_completed', False)
            return response, is_completed
        
        return None, False
