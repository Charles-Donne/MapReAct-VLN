"""
配置系统 - Config System
=========================
统一管理所有配置参数、常量和类别

模块组织：
- constants.py: 全局常量（颜色、阈值、类别列表等）
- setup.py: 配置初始化和验证
- categories.py: 类别管理（mapping、landmark、detection）
"""

from vlnce_baselines.config_system.constants import *
from vlnce_baselines.config_system.setup import ConfigHelper
from vlnce_baselines.config_system.categories import CategoryConfig, create_category_config

__all__ = [
    # 常量
    'color_palette',
    'legend_color_palette', 
    'detection_colors',
    'navigable_classes',
    'map_channels',
    'landmark_min_area_threshold',
    'landmark_min_total_pixels',
    'detection_thickness',
    'landmark_marker_color',
    'landmark_marker_border',
    'landmark_marker_radius',
    
    # 配置工具
    'ConfigHelper',
    
    # 类别管理
    'CategoryConfig',
    'create_category_config',
]
