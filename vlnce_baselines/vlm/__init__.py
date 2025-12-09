"""
VLM (Vision-Language Model) 模块
================================
集成大语言模型进行导航规划和动作决策

模块组成:
- api_client: API配置和客户端基类
- prompts: 规划提示词模板
- action_prompt: 动作执行提示词模板
- thinking: LLM规划器
- action: VLM动作执行器
- observation_collector: 观察收集和可视化
"""

from vlnce_baselines.vlm.api_client import APIConfig, BaseAPIClient
from vlnce_baselines.vlm.thinking import LLMPlanner
from vlnce_baselines.vlm.action import ActionExecutor
from vlnce_baselines.vlm.observation_collector import ObservationCollector

__all__ = [
    'APIConfig',
    'BaseAPIClient', 
    'LLMPlanner',
    'ActionExecutor',
    'ObservationCollector',
]
