"""
语义建图器 - SemanticMapper
============================
职责：
1. 语义地图更新逻辑
2. Floor区域提取
3. 轨迹管理
4. 地图状态查询

设计原则：
- 单一职责：只负责建图和地图状态管理
- 解耦：独立于可视化和控制器
- 封装：隐藏地图内部实现细节
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import torch

from skimage.morphology import disk, remove_small_objects
from skimage.morphology import binary_closing as _binary_closing_compat

from vlnce_baselines.config_system.constants import navigable_classes, map_channels


class SemanticMapper:
    """语义建图器 - 管理地图构建和更新逻辑"""
    
    def __init__(self, 
                 mapping_module,
                 map_shape: Tuple[int, int],
                 resolution: int = 5):
        """
        Args:
            mapping_module: Semantic_Mapping实例
            map_shape: 地图尺寸 (H, W)
            resolution: 地图分辨率 (cm/pixel)
        """
        self.mapping_module = mapping_module
        self.map_shape = map_shape
        self.resolution = resolution
        
        # 轨迹管理（存储轨迹坐标列表，用于绘制平滑曲线）
        self.trajectory_points = []  # [(x, y), ...] 像素坐标列表
        self.enable_trajectory = True  # 轨迹开关
        
        # 地图缓存
        self.floor = np.zeros(map_shape)
        self.full_map = None
        self.full_pose = None
    
    def reset(self):
        """重置建图器状态"""
        self.trajectory_points = []  # 清空轨迹点列表
        self.floor = np.zeros(self.map_shape)
        self.full_map = None
        self.full_pose = None
        self.mapping_module.reset()
    
    def init_map_and_pose(self, num_detected_classes: int):
        """初始化地图和位姿"""
        self.mapping_module.init_map_and_pose(num_detected_classes=num_detected_classes)
    
    def update_map(self, 
                  batch_obs: torch.Tensor,
                  poses: torch.Tensor,
                  step: int,
                  detected_classes: List[str],
                  episode_id: int) -> Dict[str, Any]:
        """
        更新语义地图
        
        Args:
            batch_obs: 批量观察 [B, C, H, W]
            poses: 位姿变化 [B, 3] [Δx, Δy, Δθ]
            step: 当前步数
            detected_classes: 已检测类别列表
            episode_id: episode ID
        
        Returns:
            map_state: 地图状态字典
                - full_map: [C, H, W]
                - full_pose: [3] (x, y, orientation)
                - floor: [H, W]
                - visited_vis: [H, W]
        """
        # 1. 调用底层mapping_module更新
        self.mapping_module(batch_obs, poses)
        
        # 2. 获取更新后的地图
        full_map, full_pose, one_step_full_map = self.mapping_module.update_map(
            step, detected_classes, episode_id
        )
        
        # 转换为numpy（如果是tensor）
        if torch.is_tensor(full_map):
            self.full_map = full_map[0].cpu().numpy()  # [C, H, W]
        else:
            self.full_map = full_map[0]
        
        if torch.is_tensor(full_pose):
            self.full_pose = full_pose[0].cpu().numpy()  # [3]
        else:
            self.full_pose = full_pose[0]
        
        # 3. 提取floor区域
        self.floor = self.extract_floor(self.full_map, detected_classes)
        
        # 4. 更新轨迹
        if self.enable_trajectory:
            self.update_trajectory(self.full_pose)
        
        # 5. 清空单步地图（准备下一步）
        self.mapping_module.one_step_full_map.fill_(0.)
        self.mapping_module.one_step_local_map.fill_(0.)
        
        return {
            'full_map': self.full_map,
            'full_pose': self.full_pose,
            'floor': self.floor,
            'trajectory_points': self.trajectory_points
        }
    
    def extract_floor(self, 
                     full_map: np.ndarray,
                     detected_classes: List[str]) -> np.ndarray:
        """
        从full_map提取floor区域（已弃用：floor现在是语义类别）
        
        注意：按照ZS_Evaluator的方式，floor现在是full_map[4+]中的第一个语义类别，
        不再需要通过形态学方法提取。这个方法保留仅用于向后兼容。
        
        Args:
            full_map: [C, H, W] 全局地图
            detected_classes: 已检测类别列表（全局累计的类别，可能多于当前步的检测）
        
        Returns:
            floor: [H, W] floor地图（现在主要用于向后兼容，实际floor在semantic layer）
        """
        # 使用阈值过滤小区域
        full_map_filtered = remove_small_objects(full_map.astype(bool), min_size=16)
        
        # 提取地图通道
        obstacles = full_map_filtered[0, ...].astype(bool)  # 障碍物
        explored_area = full_map_filtered[1, ...].astype(bool)  # 已探索区域
        
        # 提取语义层（从第 map_channels 个通道开始）
        semantic_layers = full_map_filtered[map_channels:, ...]
        
        # 关键修复：使用full_map的实际通道数，而不是detected_classes的长度
        # detected_classes是全局累计的，但每步的full_map只包含当前步检测到的类别
        num_semantic_channels = semantic_layers.shape[0]
        
        # 如果没有语义通道，直接返回基于explored的简单floor
        if num_semantic_channels == 0:
            # 简单处理：explored且非障碍物的区域
            floor = np.logical_and(explored_area, np.logical_not(obstacles))
            return floor.astype(np.uint8)
        
        # 区分可导航和不可导航的类别（只处理当前步实际存在的类别）
        navigable_index = []
        not_navigable_index = []
        
        for i in range(num_semantic_channels):
            # 由于detected_classes可能多于semantic_layers，需要安全索引
            if i < len(detected_classes):
                cls_name = detected_classes[i]
                if cls_name in navigable_classes:
                    navigable_index.append(i)
                else:
                    not_navigable_index.append(i)
        
        # 不可导航物体
        if len(not_navigable_index) > 0:
            objects = np.sum(semantic_layers[not_navigable_index], axis=0).astype(bool)
        else:
            objects = np.zeros_like(obstacles)
        
        # 可导航区域（如floor, stairs等）
        if len(navigable_index) > 0:
            navigable = np.logical_or.reduce(semantic_layers[navigable_index])
            navigable = np.logical_and(navigable, np.logical_not(objects))
        else:
            navigable = np.zeros_like(obstacles)
        
        # 计算自由空间
        free_mask = 1 - np.logical_or(obstacles, objects)
        free_mask = np.logical_or(free_mask, navigable)
        floor = explored_area * free_mask
        
        # 过滤小floor区域并形态学闭运算
        floor = remove_small_objects(floor, min_size=100).astype(bool)
        floor = _binary_closing_compat(floor, disk(3))
        
        # 静默返回，不输出调试信息
        return floor.astype(np.uint8)
    
    def update_trajectory(self, full_pose: np.ndarray):
        """
        更新轨迹坐标列表（记录位置点，稍后绘制平滑曲线）
        
        Args:
            full_pose: [3] (x, y, orientation) 当前位姿（米）
        """
        if not self.enable_trajectory:
            return
        
        # 转换位置到像素坐标
        position = full_pose[:2] * 100 / self.resolution  # 米 → 像素
        y = int(np.clip(position[0], 0, self.map_shape[0] - 1))
        x = int(np.clip(position[1], 0, self.map_shape[1] - 1))
        
        # 添加到轨迹点列表（避免重复添加相同位置）
        if len(self.trajectory_points) == 0 or self.trajectory_points[-1] != (x, y):
            self.trajectory_points.append((x, y))
    
    def toggle_trajectory(self):
        """切换轨迹绘制开关"""
        self.enable_trajectory = not self.enable_trajectory
        status = "启用" if self.enable_trajectory else "禁用"
        return status
    
    def clear_trajectory(self):
        """
        清空当前轨迹
        
        使用场景：
        - 子任务完成时：清空上一子任务的轨迹，开始记录新子任务轨迹
        - 每个子任务都有独立的轨迹显示，不会累积
        
        注意：
        - 轨迹是动态绘制在地图上的，不会写入底图
        - landmark标注同样是动态绘制，更新landmark_classes即可替换
        """
        self.trajectory_points = []
    
    def get_map_state(self) -> Dict[str, Any]:
        """
        获取当前地图状态
        
        注意：floor字段保留用于向后兼容，但实际floor渲染现在从full_map[4+]的
        语义类别中自动获取（floor是第一个mapping_class，索引为0）
        
        Returns:
            state: 地图状态字典
        """
        return {
            'full_map': self.full_map,
            'full_pose': self.full_pose,
            'floor': self.floor,
            'trajectory_points': self.trajectory_points,
            'map_shape': self.map_shape,
            'resolution': self.resolution
        }
    
    def get_current_pose(self) -> Optional[Tuple[float, float, float]]:
        """获取当前位姿"""
        if self.full_pose is None:
            return None
        return tuple(self.full_pose)


# ========== 便捷函数 ==========

def create_mapper(mapping_module, 
                 map_shape: Tuple[int, int],
                 resolution: int = 5) -> SemanticMapper:
    """创建SemanticMapper实例"""
    return SemanticMapper(mapping_module, map_shape, resolution)
