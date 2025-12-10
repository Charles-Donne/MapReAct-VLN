"""
检测模块 - Detection Module
============================
目标检测和实例分割

模块：
- grounded_sam.py: GroundedSAM (GroundingDINO + SAM/RepViTSAM)
- RepViTSAM/: 轻量级SAM模型
"""

from vlnce_baselines.detection.grounded_sam import GroundedSAM

__all__ = ['GroundedSAM']
