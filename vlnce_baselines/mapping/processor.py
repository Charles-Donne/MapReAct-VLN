"""
语义处理器 - SemanticProcessor
================================
职责：
1. Winner-Takes-All算法（防止语义重叠）
2. 掩码处理和合并
3. 标签解析

设计原则：
- 单一职责：只负责语义分割后处理
- 无状态：纯函数式设计，易于测试
- 可复用：独立于Controller和Mapper
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from collections import defaultdict


class SemanticProcessor:
    """语义处理器 - 提供语义分割后处理工具"""
    
    @staticmethod
    def apply_winner_takes_all(
        masks: np.ndarray,
        labels: List[str],
        confidences: Optional[List[float]] = None,
        height: int = 224,
        width: int = 224
    ) -> np.ndarray:
        """
        应用Winner-Takes-All机制：每个像素只属于置信度最高的类别
        
        关键改进：
        1. 每个像素只属于置信度最高的那个类别
        2. 防止 "既是桌子又是椅子" 的重叠标注
        3. 确保语义地图和landmark标注的准确性
        
        Args:
            masks: [N, H, W] - N个检测结果的mask
            labels: ["chair", "table", ...] - 类别名列表（已去掉置信度）
            confidences: [N] - 每个检测的置信度分数（可选）
            height: 输出高度
            width: 输出宽度
        
        Returns:
            final_masks: [num_classes, H, W] - 每个类别的mask（无重叠）
        """
        if masks.shape == (0,) or len(masks) == 0:
            # 创建唯一类别列表
            unique_classes = list(set(labels)) if labels else []
            return np.zeros((len(unique_classes), height, width))
        
        h, w = masks.shape[1], masks.shape[2]
        num_detections = masks.shape[0]
        
        # ===== 阶段1: 构建置信度权重 =====
        if confidences is None:
            # 使用mask面积作为权重
            confidences = np.array([masks[i].sum() for i in range(num_detections)])
        else:
            confidences = np.array(confidences)
        
        # ===== 阶段2: 为每个像素找到置信度最高的类别 =====
        winner_class = np.full((h, w), -1, dtype=np.int32)  # -1表示无检测
        winner_confidence = np.zeros((h, w), dtype=np.float32)
        
        for i in range(num_detections):
            mask_i = masks[i] > 0.5  # 二值化
            conf_i = confidences[i]
            
            # 更新winner：当前置信度 > 已有置信度
            update_mask = mask_i & (conf_i > winner_confidence)
            winner_class[update_mask] = i
            winner_confidence[update_mask] = conf_i
        
        # ===== 阶段3: 按类别分组（合并同类检测）=====
        class_to_detection_ids = defaultdict(list)
        for i, label in enumerate(labels):
            class_to_detection_ids[label].append(i)
        
        # 创建唯一类别列表（保持顺序）
        unique_classes = []
        seen = set()
        for label in labels:
            if label not in seen:
                unique_classes.append(label)
                seen.add(label)
        
        # ===== 阶段4: 构建最终的类别mask =====
        final_masks = np.zeros((len(unique_classes), h, w), dtype=np.float32)
        
        for class_name, detection_ids in class_to_detection_ids.items():
            # 找到该类别在unique_classes中的索引
            if class_name not in unique_classes:
                continue
            class_idx = unique_classes.index(class_name)
            
            # 合并该类别的所有检测：像素属于该类 当且仅当 winner是该类的某个检测
            class_mask = np.zeros((h, w), dtype=bool)
            for det_id in detection_ids:
                class_mask |= (winner_class == det_id)
            
            final_masks[class_idx] = class_mask.astype(np.float32)
        
        return final_masks
    
    @staticmethod
    def process_masks_simple(
        masks: np.ndarray,
        labels: List[str],
        unique_classes: List[str]
    ) -> np.ndarray:
        """
        简单的掩码处理：合并相同类别（不使用Winner-Takes-All）
        
        Args:
            masks: [N, H, W] - N个检测结果的mask
            labels: 标签列表
            unique_classes: 唯一类别列表
        
        Returns:
            final_masks: [num_classes, H, W]
        """
        if masks.shape == (0,):
            return np.zeros((len(unique_classes), *masks.shape[1:]))
        
        class_to_indexes = defaultdict(list)
        for i, label in enumerate(labels):
            class_to_indexes[label].append(i)
        
        idx = [unique_classes.index(label) for label in class_to_indexes.keys() 
               if label in unique_classes]
        final_masks = np.zeros((len(unique_classes), *masks.shape[1:]))
        
        for i, label in enumerate(class_to_indexes.keys()):
            if label not in unique_classes:
                continue
            class_idx = unique_classes.index(label)
            mask_indices = class_to_indexes[label]
            combined_mask = np.zeros(masks.shape[1:])
            for mask_idx in mask_indices:
                combined_mask = np.maximum(combined_mask, masks[mask_idx, ...])
            final_masks[class_idx, ...] = combined_mask
        
        return final_masks
    
    @staticmethod
    def parse_labels(labels: List[str]) -> Tuple[List[str], List[float]]:
        """
        解析标签：提取类别名和置信度
        
        Args:
            labels: ["chair 0.85", "table 0.92", ...]
        
        Returns:
            (class_names, confidences)
        """
        class_names = []
        confidences = []
        
        for label in labels:
            parts = label.split()
            if len(parts) >= 2:
                # 类别名可能有多个单词，最后一个是置信度
                try:
                    conf = float(parts[-1])
                    class_name = " ".join(parts[:-1])
                except ValueError:
                    # 如果无法解析置信度，整个当作类别名
                    class_name = label
                    conf = 0.5
            else:
                class_name = label
                conf = 0.5
            
            class_names.append(class_name)
            confidences.append(conf)
        
        return class_names, confidences
    
    @staticmethod
    def extract_unique_classes(labels: List[str]) -> List[str]:
        """
        提取唯一类别列表（保持顺序）
        
        Args:
            labels: 标签列表（可能包含重复）
        
        Returns:
            unique_classes: 唯一类别列表
        """
        unique = []
        seen = set()
        for label in labels:
            if label not in seen:
                unique.append(label)
                seen.add(label)
        return unique


# ========== 便捷函数 ==========

def process_detection_masks(
    masks: np.ndarray,
    labels: List[str],
    use_winner_takes_all: bool = True,
    height: int = 224,
    width: int = 224
) -> Tuple[np.ndarray, List[str]]:
    """
    处理检测掩码的便捷函数
    
    Args:
        masks: 原始mask [N, H, W]
        labels: 标签列表（可能包含置信度）
        use_winner_takes_all: 是否使用Winner-Takes-All
        height: 输出高度
        width: 输出宽度
    
    Returns:
        (processed_masks, unique_classes)
    """
    # 解析标签
    class_names, confidences = SemanticProcessor.parse_labels(labels)
    unique_classes = SemanticProcessor.extract_unique_classes(class_names)
    
    # 处理掩码
    if use_winner_takes_all:
        processed_masks = SemanticProcessor.apply_winner_takes_all(
            masks, class_names, confidences, height, width
        )
    else:
        processed_masks = SemanticProcessor.process_masks_simple(
            masks, class_names, unique_classes
        )
    
    return processed_masks, unique_classes
