"""
配置管理工具 - ConfigHelper
=============================
职责：
1. 配置初始化和验证
2. 配置参数设置
3. Episode配置

设计原则：
- 简化Controller初始化
- 集中配置逻辑
- 提高可维护性
"""

from typing import List, Optional
from habitat import Config


class ConfigHelper:
    """配置管理助手"""
    
    @staticmethod
    def setup_navigation_config(
        config: Config,
        torch_gpu_id: Optional[int] = None,
        num_environments: Optional[int] = None
    ) -> Config:
        """
        配置导航相关参数
        
        Args:
            config: Habitat配置对象
            torch_gpu_id: GPU设备ID（可选，从config读取）
            num_environments: 环境数量（可选，从config读取）
        
        Returns:
            config: 配置后的Config对象
        """
        config.defrost()
        
        # 从config读取默认值
        if torch_gpu_id is None:
            torch_gpu_id = config.TORCH_GPU_ID
        if num_environments is None:
            num_environments = config.NUM_ENVIRONMENTS
        
        # 配置MAP参数
        config.MAP.DEVICE = torch_gpu_id
        config.MAP.HFOV = config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.HFOV
        config.MAP.AGENT_HEIGHT = config.TASK_CONFIG.SIMULATOR.AGENT_0.HEIGHT
        config.MAP.NUM_ENVIRONMENTS = num_environments
        config.MAP.RESULTS_DIR = config.RESULTS_DIR
        
        config.freeze()
        return config
    
    @staticmethod
    def setup_episode_config(
        config: Config,
        episode_ids: List[int],
        num_environments: int = 1
    ) -> Config:
        """
        配置Episode相关参数
        
        Args:
            config: Habitat配置对象
            episode_ids: 要运行的episode ID列表
            num_environments: 环境数量
        
        Returns:
            config: 配置后的Config对象
        """
        config.defrost()
        config.TASK_CONFIG.DATASET.EPISODES_ALLOWED = episode_ids
        config.NUM_ENVIRONMENTS = num_environments
        config.freeze()
        return config
    
    @staticmethod
    def setup_results_dir(
        config: Config,
        results_dir: str
    ) -> Config:
        """
        设置结果保存目录
        
        Args:
            config: Habitat配置对象
            results_dir: 结果保存路径
        
        Returns:
            config: 配置后的Config对象
        """
        config.defrost()
        config.RESULTS_DIR = results_dir
        # 同步更新MAP配置
        if hasattr(config, 'MAP'):
            config.MAP.RESULTS_DIR = results_dir
        config.freeze()
        return config
    
    @staticmethod
    def validate_config(config: Config) -> bool:
        """
        验证配置完整性
        
        Args:
            config: Habitat配置对象
        
        Returns:
            is_valid: 配置是否有效
        """
        required_attrs = [
            'TORCH_GPU_ID',
            'NUM_ENVIRONMENTS',
            'RESULTS_DIR',
            'TASK_CONFIG',
            'MAP'
        ]
        
        for attr in required_attrs:
            if not hasattr(config, attr):
                print(f"[Config验证] 缺少必要属性: {attr}")
                return False
        
        # 验证MAP配置
        map_required = ['MAP_SIZE_CM', 'MAP_RESOLUTION', 'FRAME_WIDTH', 'FRAME_HEIGHT']
        for attr in map_required:
            if not hasattr(config.MAP, attr):
                print(f"[Config验证] MAP缺少必要属性: {attr}")
                return False
        
        return True
    
    @staticmethod
    def print_config_summary(config: Config):
        """
        打印配置摘要
        
        Args:
            config: Habitat配置对象
        """
        print("\n" + "="*60)
        print("配置摘要")
        print("="*60)
        print(f"GPU设备: {config.TORCH_GPU_ID}")
        print(f"环境数量: {config.NUM_ENVIRONMENTS}")
        print(f"结果目录: {config.RESULTS_DIR}")
        print(f"\n地图配置:")
        print(f"  - 地图大小: {config.MAP.MAP_SIZE_CM} cm")
        print(f"  - 分辨率: {config.MAP.MAP_RESOLUTION} cm/pixel")
        print(f"  - 帧尺寸: {config.MAP.FRAME_WIDTH}×{config.MAP.FRAME_HEIGHT}")
        print(f"\n传感器配置:")
        print(f"  - RGB尺寸: {config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.WIDTH}×{config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.HEIGHT}")
        print(f"  - HFOV: {config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.HFOV}°")
        print(f"  - 深度范围: {config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH}-{config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH} m")
        
        if hasattr(config.TASK_CONFIG.DATASET, 'EPISODES_ALLOWED'):
            episodes = config.TASK_CONFIG.DATASET.EPISODES_ALLOWED
            print(f"\nEpisode配置:")
            print(f"  - 允许的Episodes: {episodes[:5]}{'...' if len(episodes) > 5 else ''}")
            print(f"  - 总数: {len(episodes)}")
        
        print("="*60 + "\n")


# ========== 便捷函数 ==========

def create_navigation_config(
    base_config: Config,
    episode_ids: List[int],
    results_dir: Optional[str] = None,
    validate: bool = True
) -> Config:
    """
    一键创建导航配置（组合多个配置步骤）
    
    Args:
        base_config: 基础配置
        episode_ids: Episode ID列表
        results_dir: 结果目录（可选）
        validate: 是否验证配置
    
    Returns:
        config: 配置完成的Config对象
    """
    config = base_config
    
    # 设置Episode
    config = ConfigHelper.setup_episode_config(config, episode_ids)
    
    # 设置结果目录
    if results_dir:
        config = ConfigHelper.setup_results_dir(config, results_dir)
    
    # 配置导航参数
    config = ConfigHelper.setup_navigation_config(config)
    
    # 验证配置
    if validate and not ConfigHelper.validate_config(config):
        raise ValueError("配置验证失败")
    
    return config
