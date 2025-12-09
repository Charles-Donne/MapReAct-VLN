"""
类别配置管理 - CategoryConfig
==============================
职责：
1. 统一管理mapping和landmark类别
2. 提供类别查询接口
3. 动态类别集合管理

设计原则：
- 封装类别系统复杂性
- 提供清晰的分类接口
- 支持运行时类别更新
"""

from typing import List, Set

# OrderedSet implementation (previously in data_utils.py)
class OrderedSet:
    """有序集合 - 保持插入顺序的set"""
    def __init__(self, iterable=None):
        self._dict = {}
        if iterable:
            for item in iterable:
                self.add(item)
    
    def add(self, item):
        self._dict[item] = None
    
    def __contains__(self, item):
        return item in self._dict
    
    def __iter__(self):
        return iter(self._dict.keys())
    
    def __len__(self):
        return len(self._dict)
    
    def __repr__(self):
        return f"OrderedSet({list(self._dict.keys())})"


class CategoryConfig:
    """类别配置管理器"""
    
    def __init__(self, 
                 mapping_classes: List[str],
                 landmark_classes: List[str]):
        """
        Args:
            mapping_classes: 建图基础类别（固定）
            landmark_classes: Landmark类别（动态）
        """
        # 基础类别（不变）
        self._mapping_classes = mapping_classes.copy()
        self._landmark_classes = landmark_classes.copy()
        
        # 完整检测类别（mapping + landmark）
        self._detection_classes = self._mapping_classes + self._landmark_classes
        
        # 运行时检测到的类别（动态更新）
        self._detected_classes = OrderedSet()
    
    # ========== 属性访问 ==========
    
    @property
    def mapping_classes(self) -> List[str]:
        """建图基础类别（floor, stairs, wall等）"""
        return self._mapping_classes.copy()
    
    @property
    def landmark_classes(self) -> List[str]:
        """Landmark类别（bed, chair, table等）"""
        return self._landmark_classes.copy()
    
    @property
    def detection_classes(self) -> List[str]:
        """完整检测类别（mapping + landmark）"""
        return self._detection_classes.copy()
    
    @property
    def detected_classes(self) -> List[str]:
        """运行时检测到的类别"""
        return list(self._detected_classes)
    
    # ========== 类别查询 ==========
    
    def is_mapping_class(self, class_name: str) -> bool:
        """判断是否为建图类别"""
        return class_name in self._mapping_classes
    
    def is_landmark_class(self, class_name: str) -> bool:
        """判断是否为Landmark类别"""
        return class_name in self._landmark_classes
    
    def is_detected(self, class_name: str) -> bool:
        """判断是否已检测到"""
        return class_name in self._detected_classes
    
    def get_class_type(self, class_name: str) -> str:
        """
        获取类别类型
        
        Returns:
            'mapping', 'landmark', 或 'unknown'
        """
        if self.is_mapping_class(class_name):
            return 'mapping'
        elif self.is_landmark_class(class_name):
            return 'landmark'
        else:
            return 'unknown'
    
    # ========== 类别管理 ==========
    
    def add_detected_class(self, class_name: str):
        """添加检测到的类别"""
        if class_name in self._detection_classes:
            self._detected_classes.add(class_name)
    
    def add_detected_classes(self, class_names: List[str]):
        """批量添加检测到的类别"""
        for name in class_names:
            self.add_detected_class(name)
    
    def reset_detected(self):
        """重置检测记录"""
        self._detected_classes = OrderedSet()
    
    def get_detected_by_type(self, class_type: str) -> List[str]:
        """
        按类型获取检测到的类别
        
        Args:
            class_type: 'mapping' 或 'landmark'
        
        Returns:
            检测到的指定类型类别列表
        """
        if class_type == 'mapping':
            return [c for c in self._detected_classes if c in self._mapping_classes]
        elif class_type == 'landmark':
            return [c for c in self._detected_classes if c in self._landmark_classes]
        else:
            return []
    
    # ========== 统计信息 ==========
    
    def get_statistics(self) -> dict:
        """
        获取类别统计信息
        
        Returns:
            统计字典
        """
        detected_mapping = self.get_detected_by_type('mapping')
        detected_landmark = self.get_detected_by_type('landmark')
        
        return {
            'total_mapping': len(self._mapping_classes),
            'total_landmark': len(self._landmark_classes),
            'total_detection': len(self._detection_classes),
            'detected_total': len(self._detected_classes),
            'detected_mapping': len(detected_mapping),
            'detected_landmark': len(detected_landmark),
            'mapping_classes': self._mapping_classes,
            'landmark_classes': self._landmark_classes,
            'detected_classes': list(self._detected_classes),
            'detected_mapping_list': detected_mapping,
            'detected_landmark_list': detected_landmark
        }
    
    def print_summary(self):
        """打印类别配置摘要"""
        stats = self.get_statistics()
        
        print("\n" + "="*60)
        print("类别配置摘要")
        print("="*60)
        print(f"建图类别 (Mapping): {stats['total_mapping']} 个")
        print(f"  {', '.join(self._mapping_classes)}")
        print(f"\nLandmark类别: {stats['total_landmark']} 个")
        print(f"  {', '.join(self._landmark_classes)}")
        print(f"\n完整检测类别: {stats['total_detection']} 个")
        
        if stats['detected_total'] > 0:
            print(f"\n已检测到: {stats['detected_total']} 个")
            if stats['detected_mapping']:
                print(f"  - Mapping: {', '.join(stats['detected_mapping_list'])}")
            if stats['detected_landmark']:
                print(f"  - Landmark: {', '.join(stats['detected_landmark_list'])}")
        
        print("="*60 + "\n")
    
    # ========== 复制和重置 ==========
    
    def copy(self) -> 'CategoryConfig':
        """创建副本"""
        new_config = CategoryConfig(
            self._mapping_classes,
            self._landmark_classes
        )
        new_config._detected_classes = OrderedSet(self._detected_classes)
        return new_config


# ========== 便捷函数 ==========

def create_category_config() -> CategoryConfig:
    """
    创建默认类别配置（从constant.py读取）
    
    Returns:
        CategoryConfig实例
    """
    from vlnce_baselines.config_system.constants import mapping_classes, landmark_classes
    return CategoryConfig(mapping_classes, landmark_classes)


def create_custom_category_config(
    mapping_classes: List[str],
    landmark_classes: List[str]
) -> CategoryConfig:
    """
    创建自定义类别配置
    
    Args:
        mapping_classes: 自定义建图类别
        landmark_classes: 自定义Landmark类别
    
    Returns:
        CategoryConfig实例
    """
    return CategoryConfig(mapping_classes, landmark_classes)
