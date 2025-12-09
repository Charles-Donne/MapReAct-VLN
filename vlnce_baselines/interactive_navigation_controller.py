"""
Interactive Navigation Controller
实时键盘控制导航系统：建图、检测、可视化
"""
import numpy as np
import cv2
import torch
from typing import Dict, Any
from torchvision import transforms
from habitat import Config
from habitat.core.simulator import Observations
from habitat_baselines.common.environments import get_env_class

from vlnce_baselines.detection import GroundedSAM
from vlnce_baselines.mapping import Semantic_Mapping, SemanticMapper, SemanticProcessor
from vlnce_baselines.visualization import MapVisualizer
from vlnce_baselines.config_system import ConfigHelper, create_category_config
from vlnce_baselines.common.env_utils import construct_envs
from vlnce_baselines.common.utils import get_device


class InteractiveNavigationController:
    """实时键盘控制导航器"""
    
    def __init__(self, config: Config):
        print("[Init] 配置MAP参数...")
        self.config = ConfigHelper.setup_navigation_config(config)
        self.device = get_device(self.config.TORCH_GPU_ID)
        torch.cuda.set_device(self.device)
        
        self.map_args = self.config.MAP
        self.resolution = self.config.MAP.MAP_RESOLUTION
        self.width = self.config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.WIDTH
        self.height = self.config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.HEIGHT
        self.map_shape = (self.config.MAP.MAP_SIZE_CM // self.resolution,
                         self.config.MAP.MAP_SIZE_CM // self.resolution)
        
        print("[Init] 初始化Habitat环境...")
        self.envs = construct_envs(
            self.config, 
            get_env_class(self.config.ENV_NAME),
            auto_reset_done=False,
            episodes_allowed=self.config.TASK_CONFIG.DATASET.EPISODES_ALLOWED,
        )
        print(f"[Init] 环境初始化完成，episodes: {self.envs.number_of_episodes}")
        
        print("[Init] 初始化GroundedSAM...")
        self.segment_module = GroundedSAM(self.config, self.device)
        
        print("[Init] 初始化Semantic Mapping...")
        mapping_module = Semantic_Mapping(self.config.MAP).to(self.device)
        mapping_module.eval()
        
        print("[Init] 初始化Semantic Mapper...")
        self.mapper = SemanticMapper(mapping_module, self.map_shape, self.resolution)
        
        print("[Init] 初始化Map Visualizer...")
        self.visualizer = MapVisualizer(self.config.RESULTS_DIR, self.resolution, self.map_shape)
        
        self.category_config = create_category_config()
        self.mapping_classes = self.category_config.mapping_classes
        self.landmark_classes = self.category_config.landmark_classes
        self.detection_classes = self.category_config.detection_classes
        self.classes = []
        
        from vlnce_baselines.config_system.constants import landmark_min_area_threshold, landmark_min_total_pixels
        self.landmark_min_area_threshold = landmark_min_area_threshold
        self.landmark_min_total_pixels = landmark_min_total_pixels
        
        self.current_episode_id = None
        self.current_step = 0
        print("[Init] 完成\n")
    
    @property
    def detected_classes(self):
        """便捷访问detected_classes（代理到category_config）"""
        return self.category_config._detected_classes
    
    def reset_episode(self, episode_id: int = None):
        print(f"\n{'='*60}")
        print(f"[Reset] Episode {episode_id if episode_id else 0}")
        print(f"{'='*60}")
        
        self.envs.reset()
        self.current_step = 0
        self.current_episode_id = episode_id if episode_id is not None else 0
        
        self.category_config.reset_detected()
        self.classes = self.category_config.detection_classes
        self.mapper.reset()
        self.mapper.init_map_and_pose(num_detected_classes=0)
        
        current_episodes = self.envs.current_episodes()
        self.current_instruction = current_episodes[0].instruction.instruction_text
        
        print(f"\n📍 Episode {self.current_episode_id}")
        print(f"📝 指令: {self.current_instruction}")
        print(f"{'='*60}\n")
    
    def look_around(self) -> None:
        """360度环视建图(12步×30°)，步数0-11"""
        print("\n" + "="*60)
        print("🔄 环视扫描 (360°)")
        print("="*60)
        
        from habitat.sims.habitat_simulator.actions import HabitatSimActions
        
        for step in range(12):
            actions = [{"action": HabitatSimActions.TURN_LEFT}]
            outputs = self.envs.step(actions)
            obs, _, dones, _ = [list(x) for x in zip(*outputs)]
            
            if dones[0]:
                print("⚠️ Episode提前结束")
                self.current_step = step + 1
                return
            
            prev_class_count = len(self.detected_classes)
            batch_obs = self._batch_obs(obs, save_object_detection=True, step=step)
            poses = torch.from_numpy(np.array([item['sensor_pose'] for item in obs])).float().to(self.device)
            
            map_state = self.mapper.update_map(
                batch_obs, poses, step,
                list(self.detected_classes), self.current_episode_id
            )
            
            new_classes = len(self.detected_classes) - prev_class_count
            if new_classes > 0:
                print(f"\r  [{step+1}/12] +{new_classes}类", end="", flush=True)
            
            rgb_bgr = cv2.cvtColor(obs[0]['rgb'], cv2.COLOR_RGB2BGR)
            _, landmarks = self.visualizer.save_step_visualization(
                step=step,
                episode_id=self.current_episode_id,
                rgb=rgb_bgr,
                full_map=map_state['full_map'],
                trajectory_points=map_state['trajectory_points'],
                detected_classes=list(self.detected_classes),
                current_pose=map_state['full_pose'],
                floor=map_state['floor'],
                hfov=self.config.MAP.HFOV,
                detections=self.latest_detections_full if hasattr(self, 'latest_detections_full') else None,
                labels=self.latest_labels_full if hasattr(self, 'latest_labels_full') else None,
                landmark_classes=self.landmark_classes,
                mapping_classes=self.mapping_classes,
                landmark_config={
                    'min_total_pixels': self.landmark_min_total_pixels,
                    'min_area_threshold': self.landmark_min_area_threshold
                }
            )
        
        self.current_step = 12
        print()
        print("="*60)
        print(f"✅ 完成 | {len(self.detected_classes)}类 | {len(self.mapper.trajectory_points)}点")
        
        landmarks_found = [cls for cls in self.detected_classes if cls in self.landmark_classes]
        if landmarks_found:
            print(f"📍 Landmark: {', '.join(landmarks_found)}")
        print("="*60 + "\n")
    
    def step(self, action: int, save_vis: bool = True) -> Dict[str, Any]:
        """执行一步动作，更新地图并保存可视化"""
        print(f"\n[步骤{self.current_step}] {self._action_name(action)}", end="")
        
        outputs = self.envs.step([action])
        obs, rewards, dones, infos = [list(x) for x in zip(*outputs)]
        
        if dones[0]:
            print(" → Episode结束")
            self.current_step += 1
            return {
                'obs': obs[0],
                'reward': rewards[0],
                'done': dones[0],
                'info': infos[0],
                'detected_classes': list(self.detected_classes)
            }
        
        prev_class_count = len(self.detected_classes)
        batch_obs = self._batch_obs(obs, save_object_detection=True)
        poses = torch.from_numpy(
            np.array([item['sensor_pose'] for item in obs])
        ).float().to(self.device)
        
        map_state = self.mapper.update_map(
            batch_obs, poses, self.current_step,
            list(self.detected_classes), self.current_episode_id
        )
        
        new_classes = len(self.detected_classes) - prev_class_count
        print(f" +{new_classes}类" if new_classes > 0 else "")
        
        if save_vis:
            rgb_bgr = cv2.cvtColor(obs[0]['rgb'], cv2.COLOR_RGB2BGR)
            _, landmarks = self.visualizer.save_step_visualization(
                step=self.current_step,
                episode_id=self.current_episode_id,
                rgb=rgb_bgr,
                full_map=map_state['full_map'],
                trajectory_points=map_state['trajectory_points'],
                detected_classes=list(self.detected_classes),
                current_pose=map_state['full_pose'],
                floor=map_state['floor'],
                hfov=self.config.MAP.HFOV,
                detections=self.latest_detections_full if hasattr(self, 'latest_detections_full') else None,
                labels=self.latest_labels_full if hasattr(self, 'latest_labels_full') else None,
                landmark_classes=self.landmark_classes,
                mapping_classes=self.mapping_classes,
                landmark_config={
                    'min_total_pixels': self.landmark_min_total_pixels,
                    'min_area_threshold': self.landmark_min_area_threshold
                }
            )
        
        self.current_step += 1
        return {
            'obs': obs[0],
            'reward': rewards[0],
            'done': dones[0],
            'info': infos[0],
            'detected_classes': list(self.detected_classes)
        }
    
    def get_state(self) -> Dict[str, Any]:
        """获取当前状态"""
        map_state = self.mapper.get_map_state()
        
        return {
            'step': self.current_step,
            'episode_id': self.current_episode_id,
            'full_map': map_state['full_map'],
            'trajectory_points': map_state['trajectory_points'],
            'floor': map_state['floor'],
            'detected_classes': list(self.detected_classes),
            'current_pose': map_state['full_pose']
        }
    
    def finish_episode(self, success: bool = False, stop_action: bool = False) -> None:
        """Episode结束总结"""
        print(f"\n{'='*60}")
        print(f"🏁 EPISODE 完成")
        print(f"{'='*60}")
        print(f"Episode: {self.current_episode_id}")
        print(f"📝 指令: {self.current_instruction if hasattr(self, 'current_instruction') else 'N/A'}")
        print(f"📊 步数: {self.current_step} | 类别: {len(self.detected_classes)}")
        if self.detected_classes:
            print(f"   {', '.join(list(self.detected_classes))}")
        status = "主动停止" if stop_action else "自动结束"
        print(f"✅ {status} | 成功: {'是' if success else '否'}")
        print(f"{'='*60}\n")
    
    def _concat_obs(self, obs: Observations) -> np.ndarray:
        """合并RGB和Depth"""
        rgb = obs['rgb'].astype(np.uint8)
        depth = obs['depth']
        state = np.concatenate((rgb, depth), axis=2).transpose(2, 0, 1)
        return state
    
    def _get_sem_pred(self, rgb: np.ndarray, save_object_detection: bool = False, step: int = None) -> np.ndarray:
        """语义分割：GroundedSAM检测 + Winner-Takes-All"""
        masks_all, labels_all, annotated_images, current_detections = \
            self.segment_module.segment(rgb, classes=self.classes)
        self.mapper.mapping_module.rgb_vis = annotated_images
        
        self.latest_detections_full = current_detections
        self.latest_labels_full = labels_all.copy()
        self.latest_rgb_original = rgb.copy()
        
        all_masks = []
        all_labels = []
        all_confidences = []
        
        for i, label in enumerate(labels_all):
            parts = label.split()
            label_name = parts[0]
            confidence = float(parts[-1]) if len(parts) > 1 else 0.5
            
            all_masks.append(masks_all[i])
            all_labels.append(label_name)
            all_confidences.append(confidence)
            self.detected_classes.add(label_name)
        
        if len(all_masks) == 0:
            # 当没有检测到任何语义类别时，返回一个全0的语义通道（H, W, 1）
            # 避免将形状为 (B, 0, H, W) 的张量传入下游的池化操作，导致错误。
            return np.zeros((self.height, self.width, 1), dtype=np.float32)
        
        all_masks = np.array(all_masks)
        masks_processed = self._process_masks_with_labels(all_masks, all_labels, all_confidences)
        
        current_classes = list(dict.fromkeys(all_labels))
        global_classes = list(self.detected_classes)
        global_masks = np.zeros((len(global_classes), self.height, self.width), dtype=np.float32)
        
        for i, cls_name in enumerate(current_classes):
            if cls_name in global_classes:
                global_idx = global_classes.index(cls_name)
                if i < masks_processed.shape[0]:
                    global_masks[global_idx] = masks_processed[i]
        
        return global_masks.transpose(1, 2, 0)
    
    def _process_masks_with_labels(self, masks: np.ndarray, labels: list, confidences: list = None) -> np.ndarray:
        """Winner-Takes-All掩码处理"""
        return SemanticProcessor.apply_winner_takes_all(
            masks, labels, confidences, self.height, self.width
        )
    
    def _preprocess_depth(self, depth: np.ndarray, min_depth: float, max_depth: float) -> np.ndarray:
        """预处理深度图"""
        depth = depth[:, :, 0] * 1
        for i in range(depth.shape[1]):
            depth[:, i][depth[:, i] == 0.] = depth[:, i].max()
        mask2 = depth > 0.99
        depth[mask2] = 0.
        mask1 = depth == 0
        depth[mask1] = 100.0
        depth = min_depth * 100.0 + depth * max_depth * 100.0
        return depth
    
    def _preprocess_state(self, state: np.ndarray, save_object_detection: bool = False, step: int = None) -> np.ndarray:
        """预处理状态：RGB+Depth+Semantic"""
        state = state.transpose(1, 2, 0)
        rgb = state[:, :, :3].astype(np.uint8)
        rgb = rgb[:,:,::-1]
        depth = state[:, :, 3:4]
        
        min_depth = self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH
        max_depth = self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH
        env_frame_width = self.config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.WIDTH
        
        sem_seg_pred = self._get_sem_pred(rgb, save_object_detection, step)
        depth = self._preprocess_depth(depth, min_depth, max_depth)
        
        ds = env_frame_width // self.map_args.FRAME_WIDTH
        if ds != 1:
            trans = transforms.Resize((self.map_args.FRAME_HEIGHT, self.map_args.FRAME_WIDTH))
            rgb_tensor = torch.from_numpy(rgb.astype(np.uint8)).permute(2,0,1)
            rgb = np.asarray(trans(rgb_tensor).permute(1,2,0))
            depth = depth[ds//2::ds, ds//2::ds]
            sem_seg_pred = sem_seg_pred[ds//2::ds, ds//2::ds]
        
        depth = np.expand_dims(depth, axis=2)
        state = np.concatenate((rgb, depth, sem_seg_pred), axis=2).transpose(2, 0, 1)
        return state
    
    def _preprocess_obs(self, obs: np.ndarray, save_object_detection: bool = False, step: int = None) -> np.ndarray:
        """预处理观察"""
        concated_obs = self._concat_obs(obs)
        return self._preprocess_state(concated_obs, save_object_detection, step)
    
    def _batch_obs(self, n_obs: list, save_object_detection: bool = False, step: int = None) -> torch.Tensor:
        """批处理观察"""
        n_states = [self._preprocess_obs(obs, save_object_detection, step) for obs in n_obs]
        max_channels = max([len(state) for state in n_states])
        batch = np.stack([np.pad(state,
                [(0, max_channels - state.shape[0]),
                 (0, 0),
                 (0, 0)],
                mode='constant')
         for state in n_states], axis=0)
        return torch.from_numpy(batch).float().to(self.device)
    
    def toggle_trajectory(self):
        status = self.mapper.toggle_trajectory()
        print(f"[轨迹] {status}")
    
    def clear_trajectory(self):
        self.mapper.clear_trajectory()
        print("[轨迹] 已清空")
    
    def get_keyboard_action(self) -> int:
        """获取键盘输入：w=前进 a=左转 d=右转 t=切换轨迹 c=清空轨迹"""
        a = input("action: ")
        if a == 'w':
            return 1
        elif a == 'a':
            return 2
        elif a == 'd':
            return 3
        elif a == 't':
            self.toggle_trajectory()
            return self.get_keyboard_action()
        elif a == 'c':
            self.clear_trajectory()
            return self.get_keyboard_action()
        else:
            return 0
    
    @staticmethod
    def _action_name(action: int) -> str:
        names = {0: 'STOP', 1: 'FORWARD', 2: 'LEFT', 3: 'RIGHT'}
        return names.get(action, f'UNKNOWN({action})')
    
    def close(self):
        print("\n[Close] 关闭环境...")
        self.envs.close()
        print("[Close] 完成！")
