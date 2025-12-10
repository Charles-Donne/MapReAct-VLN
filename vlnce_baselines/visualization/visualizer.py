"""
地图可视化工具 - MapVisualizer
================================
职责：
1. 地图渲染（全局地图、局部地图）
2. 检测结果可视化
3. 轨迹绘制
4. 文件保存

设计原则：
- 单一职责：只负责可视化和保存，不涉及建图逻辑
- 解耦：独立于Controller和Mapper
- 可复用：支持多种可视化场景
"""

import os
import cv2
import copy
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional, Dict, Any

from vlnce_baselines.visualization import rendering as vu
from vlnce_baselines.config_system.constants import (
    color_palette, 
    detection_colors,
    detection_thickness,
    landmark_marker_color,
    landmark_marker_border,
    landmark_marker_radius,
)


class MapVisualizer:
    """地图可视化器 - 统一管理所有可视化和保存逻辑"""
    
    def __init__(self, 
                 results_dir: str,
                 resolution: int = 5,
                 map_shape: Tuple[int, int] = (480, 480)):
        """
        Args:
            results_dir: 保存根目录（如：data/manual_navigation）
            resolution: 地图分辨率（cm/pixel）
            map_shape: 地图尺寸
        """
        self.results_dir = results_dir
        self.resolution = resolution
        self.map_shape = map_shape
        self.color_palette = [int(x * 255.) for x in color_palette]
        
        # 注意：不在初始化时创建目录，而是在保存时根据episode_id动态创建
    
    def _create_episode_directories(self, episode_id: int):
        """为特定episode创建保存目录"""
        episode_dir = os.path.join(self.results_dir, f'episode_{episode_id}')
        dirs = ['rgb', 'global_map', 'local_map', 'detection']
        for dir_name in dirs:
            os.makedirs(os.path.join(episode_dir, dir_name), exist_ok=True)
        return episode_dir
    
    # ========== 渲染方法 ==========
    
    def render_global_map(self,
                         full_map: np.ndarray,
                         trajectory_points: List[Tuple[int, int]],
                         detected_classes: List[str],
                         floor: Optional[np.ndarray] = None,
                         current_pose: Optional[Tuple[float, float, float]] = None,
                         landmark_classes: Optional[List[str]] = None,
                         landmark_config: Optional[Dict] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        渲染全局地图（严格按照ZS_Evaluator的渲染逻辑 + 平滑轨迹线）
        
        Args:
            full_map: [C, H, W] 全局地图
                [0] = obstacle map (障碍物)
                [1] = explored map (已探索)
                [2] = current position
                [3] = history position
                [4+] = semantic classes (用于landmark标注，不用于floor渲染)
            trajectory_points: [(x, y), ...] 轨迹坐标列表（像素坐标）
            detected_classes: 已检测类别列表
            floor: [H, W] floor地图（通过形态学方法计算，像ZS_Evaluator）
            current_pose: (x, y, orientation) 当前位姿
            landmark_classes: landmark类别列表
            landmark_config: landmark配置 {min_total_pixels, min_area_threshold}
        
        Returns:
            (sem_map_vis, global_map_rotated, landmarks)
            - sem_map_vis: 基础渲染地图 (480×480)
            - global_map_rotated: 旋转调整后的地图 (480×480)，箭头朝上
            - landmarks: [(x, y, class_name), ...] 标注列表
        
        渲染层次（严格按照ZS_Evaluator）:
            - 白色(0): 未探索区域
            - 浅灰色(2): 已探索自由空间（先渲染）
            - 黑色(1): 障碍物（覆盖已探索）
            - 浅绿色(5): Floor（通过形态学计算，覆盖障碍物）
            - 橙色(3): Agent轨迹（最后覆盖）
            
        注意：不渲染bed/chair等语义类别的颜色，只用于landmark标注
        """
        obstacle_map = full_map[0, ...]
        explored_map = full_map[1, ...]
        h, w = obstacle_map.shape
        
        # ===== 阶段1: 创建语义地图（严格按照ZS_Evaluator的layer顺序）=====
        semantic_map = np.zeros((h, w), dtype=np.uint8)
        
        obstacle_mask = np.rint(obstacle_map) == 1
        explored_mask = np.rint(explored_map) == 1
        
        # ===== 正确的渲染顺序：防止障碍物被覆盖 =====
        # 正确顺序：已探索自由空间(底层) → Floor(中层) → 障碍物(顶层)
        # 这样确保障碍物始终可见，不会被Floor覆盖
        
        # Layer 1: 已探索自由空间（浅灰色）- 先绘制底层
        explored_free_mask = np.logical_and(explored_mask, ~obstacle_mask)
        semantic_map[explored_free_mask] = 2
        
        # Layer 2: Floor（浅绿色）- 覆盖部分自由空间
        if floor is not None:
            floor_mask = floor.astype(bool)
            floor_display_mask = np.logical_and(floor_mask, explored_mask)
            semantic_map[floor_display_mask] = 5  # 浅绿色
        
        # Layer 3: 障碍物（黑色）- 最后绘制，确保覆盖Floor和自由空间
        # 这样障碍物永远可见，防止碰撞
        semantic_map[obstacle_mask] = 1
        
        # ===== 阶段2: PIL调色板渲染 =====
        sem_map_vis = Image.new("P", (w, h))
        sem_map_vis.putpalette(self.color_palette)
        sem_map_vis.putdata(semantic_map.flatten().astype(np.uint8))
        sem_map_vis = sem_map_vis.convert("RGB")
        
        # 坐标系变换：翻转Y轴 + RGB→BGR
        sem_map_vis = np.flipud(sem_map_vis)
        sem_map_vis = np.array(sem_map_vis)
        sem_map_vis = sem_map_vis[:, :, [2, 1, 0]]  # RGB → BGR
        sem_map_vis = cv2.resize(sem_map_vis, (480, 480), interpolation=cv2.INTER_NEAREST)
        
        # ===== 阶段3: 提取Landmark位置（但不绘制）=====
        landmarks = []
        if landmark_classes and landmark_config:
            landmarks = self._extract_landmarks(
                full_map, detected_classes, landmark_classes,
                landmark_config['min_total_pixels'],
                landmark_config['min_area_threshold']
            )
        
        # ===== 阶段4: 旋转调整（箭头朝上，居中240,240）=====
        global_map_rotated = None
        if current_pose is not None:
            current_x, current_y, current_o = current_pose
            
            # 计算agent在地图中的位置
            map_x = current_x * 100.0 / self.resolution
            map_y = current_y * 100.0 / self.resolution
            agent_x = map_x * 480 / h
            agent_y = (w - map_y) * 480 / w
            
            # 旋转使箭头朝正上方
            rotation_angle = 90 - current_o
            rotation_center = (agent_x, agent_y)
            rotation_matrix = cv2.getRotationMatrix2D(rotation_center, rotation_angle, 1.0)
            
            # ✅ 添加平移步骤：将旋转后的agent移动到(240, 240)
            target_center = np.array([240, 240, 1])
            current_center = np.array([agent_x, agent_y, 1])
            translation = target_center[:2] - rotation_matrix @ current_center
            rotation_matrix[0, 2] += translation[0]
            rotation_matrix[1, 2] += translation[1]
            
            global_map_rotated = cv2.warpAffine(
                sem_map_vis, rotation_matrix, (480, 480),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255, 255, 255)
            )
            
            # ===== 阶段5: 创建global_map的显示副本（用于绘制轨迹和landmark）=====
            global_map_with_trajectory = global_map_rotated.copy()
            
            # 先在副本上绘制轨迹线（底层）
            if len(trajectory_points) >= 2:
                # 转换轨迹点到旋转后的坐标系
                rotated_trajectory = []
                for x, y in trajectory_points:
                    # 原始地图坐标 -> 翻转Y轴 -> 缩放到480x480
                    display_x = y * 480 / w
                    display_y = (h - 1 - x) * 480 / h
                    
                    # 应用旋转变换
                    point = np.array([display_x, display_y, 1])
                    rotated_point = rotation_matrix @ point
                    rotated_trajectory.append([int(round(rotated_point[0])), int(round(rotated_point[1]))])
                
                # 绘制实心轨迹线（2像素宽）
                if len(rotated_trajectory) >= 2:
                    trajectory_array = np.array(rotated_trajectory, dtype=np.int32)
                    cv2.polylines(global_map_with_trajectory, [trajectory_array], isClosed=False,
                                 color=(0, 165, 255), thickness=2, lineType=cv2.LINE_8)
            
            # 再在中心绘制箭头（顶层，覆盖轨迹）
            center_x, center_y = 240, 240
            arrow_angle = np.deg2rad(-90)  # 朝上
            agent_pos = (center_x, center_y, arrow_angle)
            agent_arrow = vu.get_contour_points(agent_pos, origin=(0, 0), size=12)
            cv2.drawContours(global_map_with_trajectory, [agent_arrow], 0, (0, 0, 255), -1)
            
            # ===== 阶段6: 在显示副本上绘制Landmark标记 =====
            if len(landmarks) > 0:
                landmark_summary = {}
                for marker_x, marker_y, cls_name in landmarks:
                    # 计算像素数
                    cls_idx = detected_classes.index(cls_name)
                    semantic_channel_idx = 4 + cls_idx
                    if semantic_channel_idx < full_map.shape[0]:
                        cls_mask = full_map[semantic_channel_idx, ...] > 0.5
                        pixel_count = int(cls_mask.sum())
                        if cls_name not in landmark_summary:
                            landmark_summary[cls_name] = {'count': 0, 'total_pixels': 0}
                        landmark_summary[cls_name]['count'] += 1
                        landmark_summary[cls_name]['total_pixels'] += pixel_count
                    
                    # 转换landmark坐标到旋转后的坐标系
                    display_x = marker_x * 480 / w
                    display_y = (h - 1 - marker_y) * 480 / h
                    point = np.array([display_x, display_y, 1])
                    rotated_point = rotation_matrix @ point
                    
                    # 绘制紫色圆球（在显示副本上）
                    cv2.circle(global_map_with_trajectory, 
                              (int(rotated_point[0]), int(rotated_point[1])), 
                              landmark_marker_radius, landmark_marker_color, -1)
                    cv2.circle(global_map_with_trajectory, 
                              (int(rotated_point[0]), int(rotated_point[1])), 
                              landmark_marker_radius, landmark_marker_border, 1)
                
                # 静默处理，不输出标注统计
        
        # 返回：基础地图 + 显示副本（带轨迹和landmark） + 无轨迹的旋转地图（供local_map裁剪）
        return sem_map_vis, global_map_with_trajectory, landmarks, global_map_rotated
    
    def render_local_map(self, 
                        global_map_clean: np.ndarray,
                        trajectory_points: List[Tuple[int, int]],
                        current_pose: Tuple[float, float, float],
                        hfov: float = 90.0) -> np.ndarray:
        """
        渲染局部地图（从无轨迹的全局地图裁剪，独立绘制local轨迹）
        
        Args:
            global_map_clean: 旋转后的全局地图（无轨迹，480×480），agent已居中在(240,240)
            trajectory_points: [(x, y), ...] 原始轨迹坐标列表（地图像素坐标）
            current_pose: (x, y, orientation) 当前位姿（米）
            hfov: 水平视野角度（默认90度）
        
        Returns:
            local_map: 局部地图 (400×400)，裁剪自480×480
        """
        if global_map_clean is None:
            return None
        
        # ===== 阶段1: 从无轨迹的地图裁剪中心240×240区域（对应12m×12m）=====
        center_x, center_y = 240, 240
        crop_size = 240
        crop_half = crop_size // 2
        
        x1 = center_x - crop_half
        x2 = center_x + crop_half
        y1 = center_y - crop_half
        y2 = center_y + crop_half
        
        local_map = global_map_clean[y1:y2, x1:x2].copy()
        
        # ===== 阶段2: 扩充到480×480 =====
        local_map = cv2.resize(local_map, (480, 480), interpolation=cv2.INTER_NEAREST)
        
        # ===== 阶段3: 在local_map上独立绘制轨迹线 =====
        if len(trajectory_points) >= 2:
            # 计算裁剪区域在global_map中的边界
            h, w = global_map_clean.shape[:2]
            
            # 转换轨迹点到旋转后的global坐标系，然后映射到local坐标
            local_trajectory = []
            for x, y in trajectory_points:
                # 原始地图坐标 -> 翻转Y轴 -> 缩放到480x480
                display_x = y * 480 / w
                display_y = (h - 1 - x) * 480 / h
                
                # 应用旋转变换（使用global_map的旋转矩阵）
                current_x, current_y, current_o = current_pose
                map_x = current_x * 100.0 / self.resolution
                map_y = current_y * 100.0 / self.resolution
                agent_x = map_x * 480 / h
                agent_y = (w - map_y) * 480 / w
                rotation_angle = 90 - current_o
                rotation_center = (agent_x, agent_y)
                rotation_matrix = cv2.getRotationMatrix2D(rotation_center, rotation_angle, 1.0)
                
                # 添加平移到中心
                target_center = np.array([240, 240, 1])
                current_center = np.array([agent_x, agent_y, 1])
                translation = target_center[:2] - rotation_matrix @ current_center
                rotation_matrix[0, 2] += translation[0]
                rotation_matrix[1, 2] += translation[1]
                
                point = np.array([display_x, display_y, 1])
                rotated_point = rotation_matrix @ point
                
                # 转换到local_map坐标系（裁剪区域120-360映射到0-480）
                local_x = (rotated_point[0] - 120) * 2  # 240区域放大2倍到480
                local_y = (rotated_point[1] - 120) * 2
                
                # 只添加在可见范围内的点
                if 0 <= local_x < 480 and 0 <= local_y < 480:
                    local_trajectory.append([int(round(local_x)), int(round(local_y))])
            
            # 绘制平滑轨迹线（3像素宽）
            if len(local_trajectory) >= 2:
                trajectory_array = np.array(local_trajectory, dtype=np.int32)
                cv2.polylines(local_map, [trajectory_array], isClosed=False,
                             color=(0, 165, 255), thickness=3, lineType=cv2.LINE_8)
        
        # ===== 阶段4: 绘制FOV扇形（5米视野半径）=====
        # 480像素 = 12m，所以1像素 = 2.5cm
        # 5米 = 500cm ÷ 2.5cm/pixel = 200像素
        fov_center_x, fov_center_y = 240, 240
        fov_radius = 200  # 5米视野半径
        
        # Agent朝上（-90度），FOV扇形中心线也朝上
        fov_center_angle = -90
        fov_start_angle = fov_center_angle - hfov / 2
        fov_end_angle = fov_center_angle + hfov / 2
        
        # 绘制FOV扇形轮廓（深蓝色，粗线）
        fov_outline_color = (255, 128, 0)  # 深蓝色BGR
        fov_outline_thickness = 3
        cv2.ellipse(local_map, (fov_center_x, fov_center_y), (fov_radius, fov_radius),
                   0, fov_start_angle, fov_end_angle, fov_outline_color, fov_outline_thickness)
        
        # 绘制扇形两条边线
        import math
        # 左边线
        left_angle_rad = math.radians(fov_start_angle)
        left_end_x = int(fov_center_x + fov_radius * math.cos(left_angle_rad))
        left_end_y = int(fov_center_y + fov_radius * math.sin(left_angle_rad))
        cv2.line(local_map, (fov_center_x, fov_center_y), (left_end_x, left_end_y),
                fov_outline_color, fov_outline_thickness)
        
        # 右边线
        right_angle_rad = math.radians(fov_end_angle)
        right_end_x = int(fov_center_x + fov_radius * math.cos(right_angle_rad))
        right_end_y = int(fov_center_y + fov_radius * math.sin(right_angle_rad))
        cv2.line(local_map, (fov_center_x, fov_center_y), (right_end_x, right_end_y),
                fov_outline_color, fov_outline_thickness)
        
        # ===== 阶段5: 绘制朝上的大箭头（size=24）=====
        arrow_color = (0, 0, 255)  # 亮红色BGR
        arrow_angle = np.deg2rad(-90)  # 朝上
        agent_pos = (fov_center_x, fov_center_y, arrow_angle)
        agent_arrow = vu.get_contour_points(agent_pos, origin=(0, 0), size=24)
        cv2.drawContours(local_map, [agent_arrow], 0, arrow_color, -1)
        
        # ===== 阶段6: 裁剪中心400×400 =====
        local_map_cropped = local_map[40:440, 40:440].copy()
        
        return local_map_cropped
    
    def render_detection_bbox(self, 
                              rgb: np.ndarray,
                              detections,  # sv.Detections object
                              labels: List[str],
                              landmark_classes: Optional[List[str]] = None,
                              mapping_classes: Optional[List[str]] = None) -> np.ndarray:
        """
        直接在RGB上渲染边界框（只标注Landmark类别）
        
        Args:
            rgb: RGB图像 (H, W, 3) BGR格式
            detections: supervision Detections对象
            labels: 标签列表 (例如: ["chair 0.85", "table 0.92"])
            landmark_classes: Landmark类别列表（只标注这些类别）
            mapping_classes: Mapping类别列表（不标注，仅用于建图）
        
        Returns:
            detection_vis: 检测可视化图像（只显示Landmark边界框）
        """
        detection_vis = rgb.copy()
        
        if detections is None or len(detections.xyxy) == 0:
            return detection_vis
        
        # 统计检测到的landmark
        detected_landmarks = []
        
        for i in range(len(detections.xyxy)):
            bbox = detections.xyxy[i]
            label = labels[i] if i < len(labels) else f"object_{i}"
            
            # 提取类别名和置信度
            parts = label.split()
            label_name = parts[0] if len(parts) > 0 else "unknown"
            confidence = float(parts[-1]) if len(parts) > 1 else 0.0
            
            # 只标注在landmark_classes中的类别
            is_landmark = landmark_classes and label_name in landmark_classes
            if not is_landmark:
                continue  # 跳过非Landmark类别
            
            detected_landmarks.append((label_name, confidence))
            
            # 使用醒目的黄色粗框标注Landmark
            color = detection_colors["landmark"]
            thickness = detection_thickness["landmark"]
            
            # 画边界框
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(detection_vis, (x1, y1), (x2, y2), color, thickness)
            
            # 准备标签文本（Landmark专属）
            text = f"{label_name} {confidence:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 2
            
            # 计算文本尺寸
            (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
            
            # 画标签背景（黄色）
            cv2.rectangle(detection_vis, 
                         (x1, y1 - text_h - baseline - 5), 
                         (x1 + text_w + 5, y1), 
                         color, -1)
            
            # 画标签文字（黑色，清晰易读）
            cv2.putText(detection_vis, text, 
                       (x1 + 2, y1 - baseline - 2),
                       font, font_scale, (0, 0, 0), font_thickness)
        
        # 静默处理，不输出检测统计
        
        return detection_vis
    
    # ========== 保存方法 ==========
    
    def save_rgb(self, step: int, episode_id: int, rgb: np.ndarray) -> str:
        """
        保存原始RGB帧
        
        Args:
            step: 步数
            episode_id: episode ID
            rgb: RGB图像 (H, W, 3) BGR格式
        
        Returns:
            save_path: 保存路径
        """
        episode_dir = self._create_episode_directories(episode_id)
        save_path = os.path.join(episode_dir, 'rgb', f'step-{step}.png')
        cv2.imwrite(save_path, rgb)
        return save_path
    
    def save_global_map(self, 
                       step: int,
                       episode_id: int,
                       global_map: np.ndarray) -> str:
        """
        保存全局地图（裁剪为400×400）
        
        Args:
            step: 步数
            episode_id: episode ID
            global_map: 旋转后的全局地图 (480×480)
        
        Returns:
            save_path: 保存路径
        """
        if global_map is None:
            return None
        
        # 裁剪中心400×400
        global_map_cropped = global_map[40:440, 40:440]
        
        # 简化路径：data/manual_navigation/episode_X/global_map/step-Y.png
        episode_dir = self._create_episode_directories(episode_id)
        save_path = os.path.join(episode_dir, 'global_map', f'step-{step}.png')
        cv2.imwrite(save_path, global_map_cropped, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        return save_path
    
    def save_local_map(self,
                      step: int,
                      episode_id: int,
                      local_map: np.ndarray) -> str:
        """
        保存局部地图
        
        Args:
            step: 步数
            episode_id: episode ID
            local_map: 局部地图 (400×400)
        
        Returns:
            save_path: 保存路径
        """
        if local_map is None:
            return None
        
        # 简化路径：data/manual_navigation/episode_X/local_map/step-Y.png
        episode_dir = self._create_episode_directories(episode_id)
        save_path = os.path.join(episode_dir, 'local_map', f'step-{step}.png')
        cv2.imwrite(save_path, local_map, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        return save_path
    
    def save_detection(self,
                      step: int,
                      episode_id: int,
                      detection_vis: np.ndarray) -> str:
        """
        保存检测可视化
        
        Args:
            step: 步数
            episode_id: episode ID
            detection_vis: 检测可视化图像
        
        Returns:
            save_path: 保存路径
        """
        if detection_vis is None:
            return None
        
        # 简化路径：data/manual_navigation/episode_X/detection/step-Y.png
        episode_dir = self._create_episode_directories(episode_id)
        save_path = os.path.join(episode_dir, 'detection', f'step-{step}.png')
        cv2.imwrite(save_path, detection_vis)
        return save_path
    
    # ========== 一键保存方法 ==========
    
    def save_step_visualization(self, 
                               step: int,
                               episode_id: int,
                               rgb: np.ndarray,
                               full_map: np.ndarray,
                               trajectory_points: List[Tuple[int, int]],
                               detected_classes: List[str],
                               current_pose: Tuple[float, float, float],
                               floor: Optional[np.ndarray] = None,
                               hfov: float = 90.0,
                               detections=None,  # sv.Detections对象（新）
                               labels: Optional[List[str]] = None,
                               landmark_classes: Optional[List[str]] = None,
                               mapping_classes: Optional[List[str]] = None,  # 新增
                               landmark_config: Optional[Dict] = None,
                               masks: Optional[np.ndarray] = None) -> Dict[str, str]:  # 兼容旧参数
        """
        一键保存当前步骤的所有可视化（支持新detection渲染 + 平滑轨迹线）
        
        Args:
            trajectory_points: [(x, y), ...] 轨迹坐标列表（像素坐标）
            floor: [H, W] floor地图（通过形态学方法计算，像ZS_Evaluator）
            detections: supervision Detections对象（优先使用）
            masks: 检测掩码（向后兼容，已废弃）
            mapping_classes: Mapping类别列表
            landmark_classes: Landmark类别列表
        
        Returns:
            paths: 保存路径字典 {'rgb', 'global_map', 'local_map', 'detection'}
            landmarks: Landmark列表
            
        注意：floor通过形态学方法计算（像ZS_Evaluator._process_map）
        """
        paths = {}
        
        # 1. 保存RGB
        paths['rgb'] = self.save_rgb(step, episode_id, rgb)
        
        # 2. 渲染并保存全局地图（floor通过形态学方法计算 + 平滑轨迹线）
        _, global_map_with_trajectory, landmarks, global_map_clean = self.render_global_map(
            full_map, trajectory_points, detected_classes, floor,
            current_pose, landmark_classes, landmark_config
        )
        paths['global_map'] = self.save_global_map(step, episode_id, global_map_with_trajectory)
        
        # 3. 渲染并保存局部地图（从无轨迹地图裁剪，独立绘制轨迹）
        local_map = self.render_local_map(global_map_clean, trajectory_points, current_pose, hfov)
        paths['local_map'] = self.save_local_map(step, episode_id, local_map)        # 4. 渲染并保存检测结果
        if detections is not None and labels is not None:
            detection_vis = self.render_detection_bbox(
                rgb, detections, labels, 
                landmark_classes, mapping_classes
            )
            paths['detection'] = self.save_detection(step, episode_id, detection_vis)
        
        return paths, landmarks
    
    # ========== 辅助方法 ==========
    
    def _extract_landmarks(self,
                          full_map: np.ndarray,
                          detected_classes: List[str],
                          landmark_classes: List[str],
                          min_total_pixels: int,
                          min_area_threshold: int) -> List[Tuple[int, int, str]]:
        """提取landmark标记位置
        
        流程：
        1. 遍历landmark_classes（如cabinet）
        2. 检查是否在detected_classes中
        3. 计算语义通道索引：semantic_channel_idx = 4 + detected_classes.index(cls_name)
        4. 从full_map[semantic_channel_idx]提取mask
        5. 形态学闭运算：填补间隙，合并相近区域
        6. 连通域分析，过滤面积 < min_area_threshold
        7. 空间合并（距离 < landmark_merge_distance）
        
        Returns:
            List of (cx, cy, class_name)
        """
        if not landmark_classes or len(detected_classes) == 0:
            return []
        
        spatial_regions = {}
        landmark_found = False
        
        for cls_name in landmark_classes:
            if cls_name not in detected_classes:
                continue
            
            cls_idx = detected_classes.index(cls_name)
            semantic_channel_idx = 4 + cls_idx
            
            if semantic_channel_idx >= full_map.shape[0]:
                continue
            
            cls_mask = full_map[semantic_channel_idx, ...] > 0.5
            num_pixels = cls_mask.sum()
            
            if num_pixels < min_total_pixels:
                continue
            
            # 形态学闭运算：填补间隙，合并相近区域
            # 使用7×7核，可以填补距离3-4像素的间隙，合并被Winner-Takes-All分割的区域
            cls_mask_uint8 = cls_mask.astype(np.uint8)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            cls_mask_closed = cv2.morphologyEx(cls_mask_uint8, cv2.MORPH_CLOSE, kernel)
            
            # 连通性分析（在闭运算后的mask上）
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                cls_mask_closed, connectivity=8)
            
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                cx, cy = int(centroids[i][0]), int(centroids[i][1])
                
                if area < min_area_threshold:
                    continue
                
                # 检查空间合并（使用constant.py配置的距离）
                from vlnce_baselines.config_system.constants import landmark_merge_distance
                merged = False
                for existing_pos in list(spatial_regions.keys()):
                    ex_cx, ex_cy = existing_pos
                    dist = np.sqrt((cx - ex_cx)**2 + (cy - ex_cy)**2)
                    if dist < landmark_merge_distance:
                        spatial_regions[existing_pos].append((area, cls_name))
                        merged = True
                        break
                
                if not merged:
                    spatial_regions[(cx, cy)] = [(area, cls_name)]
                    if not landmark_found:
                        landmark_found = True
        landmarks = []
        for (cx, cy), candidates in spatial_regions.items():
            candidates.sort(key=lambda x: x[0], reverse=True)
            dominant_class = candidates[0][1]
            area = candidates[0][0]
            landmarks.append((cx, cy, dominant_class))
            print(f"  📍 {dominant_class} @({cx},{cy}) - {area}px")
        
        return landmarks


# ========== 便捷函数 ==========

def create_visualizer(results_dir: str, 
                     resolution: int = 5,
                     map_shape: Tuple[int, int] = (480, 480)) -> MapVisualizer:
    """创建MapVisualizer实例"""
    return MapVisualizer(results_dir, resolution, map_shape)
