"""
VLM动作执行模块
===============
低层动作决策：基于视觉和地图输出具体动作
"""
from typing import Dict, Tuple, Optional
from vlnce_baselines.vlm.api_client import APIConfig, BaseAPIClient
from vlnce_baselines.vlm.action_prompt import get_action_execution_prompt


class ActionExecutor(BaseAPIClient):
    """VLM动作执行器 - 负责低层动作决策"""
    
    REQUIRED_FIELDS = ['reasoning', 'action', 'progress_summary']
    
    def __init__(self, config_path: str = "vlnce_baselines/vlm/vlm_config.yaml", 
                 turn_angle: float = 30.0, 
                 move_distance: float = 0.25):
        """
        初始化动作执行器
        
        Args:
            config_path: VLM配置文件路径
            turn_angle: 每次转向角度（度）- 与interactive_navigation一致：30°
            move_distance: 每次前进距离（米）- 与interactive_navigation一致：0.25m
        """
        config = APIConfig(config_path)
        super().__init__(config)
        
        self.turn_angle = turn_angle
        self.move_distance = move_distance
        
        print(f"✓ Action Executor initialized")
        print(f"  Model: {self.config.model}")
        print(f"  Parameters: turn={turn_angle}°, move={move_distance}m")
    
    def validate_response(self, response: Dict) -> bool:
        """验证VLM响应是否包含所有必需字段"""
        return self.validate_fields(response, self.REQUIRED_FIELDS)
    
    def decide_action(self,
                     subtask_destination: str,
                     subtask_instruction: str,
                     first_person_image: str,
                     action_mapping: Dict[str, int],
                     progress_summary: str = "",
                     detection_image: str = None,
                     local_map_image: str = None,
                     detected_landmarks: str = None) -> Tuple[Optional[int], Optional[str], Optional[str], Optional[Dict]]:
        """
        基于第一人称视角、检测结果和局部地图决策下一步动作
        
        Args:
            subtask_destination: 子任务目的地
            subtask_instruction: 子任务指令
            first_person_image: 第一人称RGB图像路径
            action_mapping: 动作名称到ID的映射
            progress_summary: 当前子任务进度摘要
            detection_image: 目标检测图像路径（可选）
            local_map_image: 局部语义地图路径（可选）
            detected_landmarks: 已检测landmark类别字符串（可选）
            
        Returns:
            (action_id, action_name, updated_progress, full_response)
        """
        # 构建prompt
        prompt = get_action_execution_prompt(
            subtask_destination=subtask_destination,
            subtask_instruction=subtask_instruction,
            turn_angle=self.turn_angle,
            move_distance=self.move_distance,
            progress_summary=progress_summary,
            detected_landmarks=detected_landmarks
        )
        
        # 组合图像：RGB + Detection + Local Map
        images = [first_person_image]
        if detection_image:
            images.append(detection_image)
        if local_map_image:
            images.append(local_map_image)
        
        # 调用API
        response = self.call_api(prompt, images)
        
        if not response:
            print("✗ No response from VLM")
            return None, None, None, None
        
        # 验证响应
        if not self.validate_response(response):
            return None, None, None, None
        
        # 提取动作
        action_name = response['action']
        if action_name not in action_mapping:
            print(f"✗ Invalid action: {action_name}")
            print(f"✗ Valid actions: {list(action_mapping.keys())}")
            return None, None, None, None
        
        action_id = action_mapping[action_name]
        updated_progress = response['progress_summary']
        
        # 打印推理过程
        print(f"🤔 Reasoning: {response['reasoning']}")
        print(f"🎯 Action: {action_name}")
        
        return action_id, action_name, updated_progress, response
