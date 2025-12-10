"""
观察收集模块
=============
负责收集4方向观察图像、保存RGB+俯视图拼接可视化、生成GIF

与Sub-VLM-VLN的ObservationCollector类似，但适配4方向观察
"""
import os
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from habitat.utils.visualizations import maps

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False
    print("⚠️  imageio not installed, GIF generation disabled")


class ObservationCollector:
    """
    观察收集器
    
    负责:
    - 4方向图像采集和保存
    - RGB + 俯视图拼接可视化
    - GIF动画生成
    """
    
    # 4个方向的名称
    DIRECTION_NAMES = [
        "Front (0°)",
        "Right (90°)",
        "Back (180°)",
        "Left (270°)"
    ]
    
    def __init__(self, output_dir: str):
        """
        初始化收集器
        
        Args:
            output_dir: 输出目录（用于保存4方向观察图像）
        """
        self.output_dir = output_dir
        self.maps_dir = None
        self.video_frames = []
        os.makedirs(output_dir, exist_ok=True)
    
    def setup_maps_dir(self, episode_dir: str):
        """
        设置地图可视化目录
        
        Args:
            episode_dir: Episode输出根目录
        """
        self.maps_dir = os.path.join(episode_dir, "maps")
        os.makedirs(self.maps_dir, exist_ok=True)
        self.video_frames = []
    
    def save_direction_image(self, rgb: np.ndarray, direction_idx: int, 
                             prefix: str = "observe") -> str:
        """
        保存单个方向的观察图像
        
        Args:
            rgb: RGB图像 (H, W, 3)，RGB格式
            direction_idx: 方向索引 (0-3)
            prefix: 文件名前缀
            
        Returns:
            保存的图像路径
        """
        direction_name = self.DIRECTION_NAMES[direction_idx].split()[0].lower()
        filename = f"{prefix}_dir{direction_idx}_{direction_name}.jpg"
        filepath = os.path.join(self.output_dir, filename)
        
        # RGB转BGR后保存
        cv2.imwrite(filepath, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        
        return filepath
    
    def save_step_visualization(self,
                                observations: Dict,
                                info: Dict,
                                step: int,
                                instruction: str,
                                current_subtask: str = None,
                                distance: float = 0.0,
                                action: str = "") -> Optional[str]:
        """
        保存单步可视化：左边第一人称视角 + 右边俯视图 + 文本信息
        
        Args:
            observations: 环境观测字典（需包含"rgb"键）
            info: 环境指标字典（需包含"top_down_map_vlnce"键）
            step: 当前步数
            instruction: 全局导航指令
            current_subtask: 当前子任务指令（可选）
            distance: 到目标距离
            action: 当前执行的动作名称
            
        Returns:
            保存的图像路径，失败返回None
        """
        if not self.maps_dir or "rgb" not in observations:
            return None
        
        # 获取第一人称RGB
        rgb = observations["rgb"]
        
        # 获取俯视图（环境提供）
        if "top_down_map_vlnce" in info:
            top_down_map = maps.colorize_draw_agent_and_fit_to_height(
                info["top_down_map_vlnce"], rgb.shape[0]
            )
        else:
            # 如果没有地图，创建空白占位
            top_down_map = np.zeros_like(rgb)
        
        # 拼接：左边RGB + 右边俯视图
        combined = np.concatenate((rgb, top_down_map), axis=1)
        
        # 添加文本信息
        combined = self._add_text_overlay(
            combined, 
            instruction, 
            current_subtask, 
            step, 
            distance,
            action
        )
        
        # 保存（RGB格式需要转换为BGR）
        filename = f"step{step:04d}_visualization.jpg"
        filepath = os.path.join(self.maps_dir, filename)
        cv2.imwrite(filepath, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
        
        # 记录到视频帧列表（保持RGB格式用于GIF）
        self.video_frames.append(combined)
        
        return filepath
    
    def _add_text_overlay(self,
                          image: np.ndarray,
                          instruction: str,
                          current_subtask: Optional[str],
                          step: int,
                          distance: float,
                          action: str = "") -> np.ndarray:
        """
        在图像底部添加文本信息
        
        Args:
            image: 拼接后的RGB图像
            instruction: 全局指令
            current_subtask: 当前子任务
            step: 步数
            distance: 距离
            action: 动作名称
            
        Returns:
            添加文本后的图像
        """
        img = image.copy()
        h, w = img.shape[:2]
        
        # 创建文本区域
        text_height = 100
        text_area = np.zeros((text_height, w, 3), dtype=np.uint8)
        
        # 字体设置
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        color = (255, 255, 255)  # RGB白色
        
        # 计算最大文本宽度
        max_width = w - 20
        
        # 第1行：Step + Distance + Action
        line1 = f"Step: {step}  |  Distance: {distance:.2f}m  |  Action: {action}"
        cv2.putText(text_area, line1, (10, 25), font, font_scale, color, thickness)
        
        # 第2行：Instruction（可能需要截断）
        instr_text = f"Instruction: {instruction}"
        if len(instr_text) > 100:
            instr_text = instr_text[:100] + "..."
        # 自动换行处理
        lines = self._wrap_text(instr_text, font, font_scale, max_width)
        y = 50
        for line in lines[:2]:  # 最多显示2行
            cv2.putText(text_area, line, (10, y), font, font_scale, color, thickness)
            y += 20
        
        # 第3行：Subtask（如果有）
        if current_subtask:
            subtask_text = f"Subtask: {current_subtask}"
            if len(subtask_text) > 80:
                subtask_text = subtask_text[:80] + "..."
            cv2.putText(text_area, subtask_text, (10, y), font, font_scale * 0.9, 
                       (200, 200, 200), thickness)
        
        # 拼接文本区域到图像底部
        result = np.concatenate((img, text_area), axis=0)
        
        return result
    
    def _wrap_text(self, text: str, font, font_scale: float, 
                   max_width: int) -> List[str]:
        """
        文本自动换行
        
        Args:
            text: 要换行的文本
            font: OpenCV字体
            font_scale: 字体缩放
            max_width: 最大宽度（像素）
            
        Returns:
            换行后的文本列表
        """
        words = text.split(' ')
        lines = []
        current_line = ""
        
        for word in words:
            test_line = f"{current_line} {word}".strip()
            (text_width, _), _ = cv2.getTextSize(test_line, font, font_scale, 1)
            
            if text_width <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        
        if current_line:
            lines.append(current_line)
        
        return lines
    
    def save_gif(self, output_path: str = None, fps: int = 2) -> Optional[str]:
        """
        将所有帧保存为GIF动画
        
        Args:
            output_path: 输出路径（可选，默认在maps目录下）
            fps: 帧率
            
        Returns:
            GIF路径，失败返回None
        """
        if not self.video_frames:
            print("⚠️  No frames to save")
            return None
        
        if not HAS_IMAGEIO:
            print("⚠️  imageio not installed, cannot create GIF")
            return None
        
        if output_path is None and self.maps_dir:
            output_path = os.path.join(self.maps_dir, "navigation.gif")
        
        if not output_path:
            return None
        
        try:
            # 转换帧为uint8格式
            frames_rgb = []
            for frame in self.video_frames:
                if frame.dtype != np.uint8:
                    frame = frame.astype(np.uint8)
                frames_rgb.append(frame)
            
            # 计算每帧持续时间
            duration = 1.0 / fps
            
            # 保存GIF
            imageio.mimsave(output_path, frames_rgb, duration=duration, loop=0)
            
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                print(f"✅ GIF saved: {output_path} ({len(self.video_frames)} frames, {fps} fps)")
                return output_path
            else:
                print("✗ GIF file creation failed")
                return None
                
        except Exception as e:
            print(f"✗ Error saving GIF: {e}")
            return None
    
    def clear_frames(self):
        """清空视频帧列表"""
        self.video_frames = []
