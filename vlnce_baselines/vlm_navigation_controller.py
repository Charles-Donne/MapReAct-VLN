"""
VLM Navigation Controller
=========================
基于VLM的自动导航控制器

继承InteractiveNavigationController的核心功能：
- 语义建图（GroundedSAM + Semantic Mapping）
- 可视化（MapVisualizer）
- 12步×30°环视建图

新增VLM功能：
- LLM高层规划（生成子任务）
- VLM低层动作执行（基于RGB+地图决策）
- 4方向观察收集（前/右/后/左）
- RGB+俯视图拼接可视化（使用环境提供的top_down_map_vlnce）
- 结果保存供后续测评
"""
import os
import cv2
import json
import numpy as np
import torch
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime

from habitat import Config
from habitat.sims.habitat_simulator.actions import HabitatSimActions

from vlnce_baselines.interactive_navigation_controller import InteractiveNavigationController
from vlnce_baselines.vlm import LLMPlanner, ActionExecutor, ObservationCollector


class VLMNavigationController(InteractiveNavigationController):
    """
    VLM导航控制器
    
    继承自InteractiveNavigationController，添加VLM规划和执行功能
    
    工作流程：
    1. 初始环视建图（12步×30°）→ 收集4方向图像
    2. LLM规划 → 生成初始子任务
    3. VLM执行 → 循环执行动作直到子任务完成
    4. 验证环视建图（12步×30°）→ 更新地图和4方向图像
    5. 验证重规划 → 检查完成状态，生成下一子任务
    6. 重复3-5直到导航完成
    
    注意：每次验证重规划前都会执行360°环视，以更新语义地图和当前位置的4方向观察
    """
    
    # 4方向配置（从环视中提取）
    # 环视是逆时针TURN_LEFT，12步×30°=360°
    DIRECTION_STEPS = [0, 3, 6, 9]  # 对应12步中的祰0,3,6,9步
    DIRECTION_NAMES = [
        "Front (0°)",      # 步骤0: 初始朝向
        "Left (90°)",      # 步骤3: 左转90°
        "Back (180°)",     # 步骤6: 后方
        "Right (270°)"     # 步骤9: 右方（或左转270°）
    ]
    
    # 动作映射（与interactive_navigation一致）
    ACTION_MAPPING = {
        "STOP": 0,
        "MOVE_FORWARD": 1, 
        "TURN_LEFT": 2,
        "TURN_RIGHT": 3
    }
    
    def __init__(self, config: Config,
                 llm_config_path: str = "vlnce_baselines/vlm/llm_config.yaml",
                 vlm_config_path: str = "vlnce_baselines/vlm/vlm_config.yaml"):
        """
        初始化VLM导航控制器
        
        Args:
            config: Habitat配置
            llm_config_path: LLM配置文件路径
            vlm_config_path: VLM配置文件路径
        """
        # 调用父类初始化（初始化环境、检测、建图、可视化）
        super().__init__(config)
        
        # 初始化VLM模块
        print("\n[Init] 初始化VLM模块...")
        
        # 获取动作参数
        self.turn_angle = config.TASK_CONFIG.SIMULATOR.TURN_ANGLE  # 30°
        self.move_distance = config.TASK_CONFIG.SIMULATOR.FORWARD_STEP_SIZE  # 0.25m
        
        # 动作空间描述
        self.action_space = f"MOVE_FORWARD ({self.move_distance}m), TURN_LEFT ({self.turn_angle}°), TURN_RIGHT ({self.turn_angle}°), STOP"
        
        # 初始化LLM规划器
        try:
            self.planner = LLMPlanner(llm_config_path, self.action_space)
        except Exception as e:
            print(f"⚠️  LLM Planner初始化失败: {e}")
            self.planner = None
        
        # 初始化VLM执行器
        try:
            self.action_executor = ActionExecutor(vlm_config_path, self.turn_angle, self.move_distance)
        except Exception as e:
            print(f"⚠️  Action Executor初始化失败: {e}")
            self.action_executor = None
        
        # VLM状态
        self.current_subtask = None
        self.subtask_count = 0
        self.progress_summary = ""
        self.subtask_history = []
        self.current_subtask_file = None  # 当前子任务文件路径
        
        # 空间记忆系统（Waypoint Memory）
        self.waypoint_memory = []  # 路径点列表 [{id, position, area_name, description, step, detected_objects}]
        self.waypoint_counter = 0   # 路径点计数器
        
        # 观察缓存
        self.latest_obs = None  # 缓存最新的观察
        self.latest_info = None  # 缓存最新的info（包含top_down_map_vlnce）
        
        # 观察缓存（环视时收集的4方向图像）
        self.direction_images = {}  # {direction_name: image_path}
        self.latest_map_image = None
        
        # ObservationCollector（用于RGB+俯视图拼接和GIF生成）
        self.obs_collector = None
        
        print("[Init] VLM模块初始化完成\n")
    
    def reset_episode(self, episode_id: int = None):
        """重置Episode，包括VLM状态"""
        # 调用父类重置
        super().reset_episode(episode_id)
        
        # 重置VLM状态
        self.current_subtask = None
        self.subtask_count = 0
        self.progress_summary = ""
        self.subtask_history = []
        self.current_subtask_file = None
        self.direction_images = {}
        self.latest_map_image = None
        
        # 重置空间记忆
        self.waypoint_memory = []
        self.waypoint_counter = 0
        
        # 创建VLM专用目录
        self.vlm_dir = os.path.join(
            self.config.RESULTS_DIR, 
            f'episode_{self.current_episode_id}',
            'vlm'
        )
        os.makedirs(self.vlm_dir, exist_ok=True)
        os.makedirs(os.path.join(self.vlm_dir, 'observations'), exist_ok=True)
        os.makedirs(os.path.join(self.vlm_dir, 'subtasks'), exist_ok=True)
        
        # 初始化ObservationCollector（用于RGB+俯视图拼接）
        self.obs_collector = ObservationCollector(os.path.join(self.vlm_dir, 'observations'))
        self.obs_collector.setup_maps_dir(self.vlm_dir)
    
    def look_around_and_collect(self) -> Tuple[List[str], List[str]]:
        """
        360°环视建图 + 收集4方向观察图像
        
        执行12步×30°逆时针旋转（TURN_LEFT）建图，同时收集4方向图像：
        - 步骤0: 前 (0°)
        - 步骤3: 左 (90°)
        - 步骤6: 后 (180°)
        - 步骤9: 右 (270°)
        
        Returns:
            (image_paths, direction_names) - 4方向图像路径和名称
        """
        print("\n" + "="*60)
        print("🔄 环视扫描 + 收集4方向观察 (360°)")
        print("="*60)
        
        collected_images = []
        collected_directions = []
        
        # 执行12次旋转 (step-0 到 step-11)
        from habitat.sims.habitat_simulator.actions import HabitatSimActions
        for step in range(12):
            # 执行旋转
            actions = [{"action": HabitatSimActions.TURN_LEFT}]
            outputs = self.envs.step(actions)
            obs, _, dones, _ = [list(x) for x in zip(*outputs)]
            
            if dones[0]:
                print("⚠️ Episode提前结束")
                break
            
            # 更新检测和建图
            prev_class_count = len(self.detected_classes)
            batch_obs = self._batch_obs(obs, save_object_detection=True, step=step)
            poses = torch.from_numpy(np.array([item['sensor_pose'] for item in obs])).float().to(self.device)
            
            map_state = self.mapper.update_map(
                batch_obs, poses, step,
                list(self.detected_classes), self.current_episode_id
            )
            
            new_classes = len(self.detected_classes) - prev_class_count
            
            # 保存可视化
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
            
            # 收集4方向图像（步骤0,3,6,9 对应 前,右,后,左）
            if step in self.DIRECTION_STEPS:
                direction_idx = self.DIRECTION_STEPS.index(step)
                direction_name = self.DIRECTION_NAMES[direction_idx]
                
                # 保存方向观察图像
                img_path = os.path.join(
                    self.vlm_dir, 'observations',
                    f'lookaround_dir{direction_idx}_{direction_name.split()[0].lower()}.jpg'
                )
                cv2.imwrite(img_path, rgb_bgr)
                
                collected_images.append(img_path)
                collected_directions.append(direction_name)
                self.direction_images[direction_name] = img_path
                
                print(f"  📷 [{step+1}/12] 收集 {direction_name}" + (f" +{new_classes}类" if new_classes > 0 else ""))
            else:
                if new_classes > 0:
                    print(f"  [{step+1}/12] +{new_classes}类")
            
            # 缓存最后一步的观察
            self.latest_obs = obs[0]
        
        # 完成12次旋转，保存了 step-0 到 step-11（共12张）
        # 设置 current_step = 12，表示下次动作将保存为 step-12
        self.current_step = 12
        
        # 获取最新地图路径（global_map/中的图像由父类save_step_visualization保存）
        self._get_current_map_path()
        
        print("="*60)
        print(f"✅ 完成 | {len(self.detected_classes)}类 | {len(collected_images)}方向图像")
        print(f"   保存文件: step-0 到 step-12 (共13张)")
        print("="*60 + "\n")
        
        return collected_images, collected_directions
    
    def _get_current_map_path(self) -> str:
        """
        获取当前语义地图路径（使用global_map/目录中的图像，避免重复保存）
        
        Returns:
            global_map目录中上一步保存的地图路径
        """
        # 返回上一步保存的地图（当前步的地图要等step()执行后才会保存）
        episode_dir = os.path.join(
            self.config.RESULTS_DIR, 
            f'episode_{self.current_episode_id}'
        )
        last_step = self.current_step - 1
        map_path = os.path.join(episode_dir, 'global_map', f'step-{last_step}.png')
        self.latest_map_image = map_path
        return map_path

    def get_4_direction_images_from_cache(self, phase: str = "initial") -> Tuple[List[str], List[str]]:
        """
        从缓存中获取4方向图像
        
        注意：此方法仅在generate_initial_subtask中使用，因为初始规划前已经执行过环视
        验证重规划时会重新执行look_around_and_collect()来更新地图和4方向图像
        
        Args:
            phase: 阶段名称（用于文件命名）
            
        Returns:
            (image_paths, direction_names) - 4方向图像路径和名称
        """
        print(f"\n📷 获取4方向观察图像 ({phase})...")
        
        collected_images = []
        collected_directions = []
        
        # 使用环视时已经保存的图像
        for direction_name in self.DIRECTION_NAMES:
            if direction_name in self.direction_images:
                collected_images.append(self.direction_images[direction_name])
                collected_directions.append(direction_name)
                print(f"  ✅ {direction_name}")
            else:
                print(f"  ⚠️ {direction_name} 图像未找到")
        
        return collected_images, collected_directions
    
    def generate_initial_subtask(self) -> Optional[Dict]:
        """
        生成初始子任务
        
        使用环视收集的4方向图像 + 全局地图 + 局部地图调用LLM生成子任务
        """
        if not self.planner:
            print("✗ LLM Planner未初始化")
            return None
        
        print(f"\n{'*'*60}")
        print("🤖 生成初始子任务")
        print(f"{'*'*60}")
        
        # 使用环视时已经收集的图像（避免重复旋转）
        image_paths, direction_names = self.get_4_direction_images_from_cache("initial")
        
        # 获取地图路径
        # 注意：环视后 current_step=12，最后保存的地图是 step-11
        episode_dir = os.path.join(
            self.config.RESULTS_DIR, 
            f'episode_{self.current_episode_id}'
        )
        last_saved_step = self.current_step - 1  # 11
        global_map = os.path.join(episode_dir, 'global_map', f'step-{last_saved_step}.png')
        local_map = os.path.join(episode_dir, 'local_map', f'step-{last_saved_step}.png')
        
        # 验证地图文件存在
        if not os.path.exists(global_map):
            print(f"✗ Global map not found: {global_map}")
            return None
        
        # 创建带waypoint标注的地图副本（不覆盖原始地图）
        global_map_for_llm = global_map
        if len(self.waypoint_memory) > 0:
            global_map_img = cv2.imread(global_map)
            global_map_img = self.visualize_waypoints_on_map(global_map_img)
            # 保存为单独的可视化版本
            global_map_for_llm = os.path.join(episode_dir, 'global_map', f'step-{self.current_step}_with_waypoints.png')
            cv2.imwrite(global_map_for_llm, global_map_img)
        
        # 调用LLM生成初始子任务（第一次规划，无detected_landmarks和waypoint_summary）
        response = self.planner.generate_initial_subtask(
            instruction=self.current_instruction,
            observation_images=image_paths,
            direction_names=direction_names,
            global_map_image=global_map_for_llm,  # 使用带waypoint标注的版本
            local_map_image=local_map if os.path.exists(local_map) else None
        )
        
        if not response:
            print("✗ LLM未返回有效响应")
            return None
        
        # 保存子任务
        self.current_subtask = response
        self.subtask_count += 1
        self.progress_summary = ""
        
        # 记录当前位置信息（用于后续验证参考）
        self.current_position_info = {
            'waypoint': response.get('waypoint', 'Unknown'),
            'observation': response.get('current_observation', ''),
            'step': self.current_step
        }
        
        # 创建路径点记录（空间记忆）
        waypoint_desc = response.get('waypoint', 'Unknown location')
        # 不传position参数，让add_waypoint()从mapper.curr_loc获取正确的地图像素坐标
        self.add_waypoint(waypoint_desc)
        
        # 记录并动态更新目标landmark
        subtask_landmark = response.get('subtask_landmark', None)
        if subtask_landmark:
            # 验证：只要在mapping_classes中（能被GroundedSAM检测）就可以作为landmark
            if subtask_landmark in self.mapping_classes:
                # 动态更新landmark_classes（只标注当前子任务的目标）
                self.landmark_classes = [subtask_landmark]
                self.target_landmark = subtask_landmark
                print(f"  🎯 目标Landmark已设定: {self.target_landmark}")
                print(f"  📍 已动态更新landmark_classes: {self.landmark_classes}")
            else:
                print(f"  ⚠️  警告: '{subtask_landmark}' 不在mapping_classes中，GroundedSAM无法检测")
                print(f"  💡 可检测类别: {', '.join(self.mapping_classes)}")
                self.target_landmark = None
                self.landmark_classes = []  # 重置为空
        else:
            print(f"  ℹ️  未指定subtask_landmark，不标注landmark")
            self.target_landmark = None
            self.landmark_classes = []  # 重置为空
        
        self._save_subtask(response, "initial")
        
        # 打印子任务信息
        self._print_subtask_info(response, is_initial=True)
        
        return response
    
    def verify_and_replan(self) -> Tuple[bool, Optional[Dict]]:
        """
        验证当前子任务并重新规划
        
        流程：
        1. 执行360°环视建图（更新语义地图）
        2. 收集当前位置的4方向图像
        3. 调用LLM验证子任务完成状态
        4. 如未完成，生成新子任务
        
        Returns:
            (is_completed, new_subtask)
        """
        if not self.planner or not self.current_subtask:
            return False, None
        
        # 获取环视时收集的4方向图像
        phase = f"verify_{self.subtask_count}"
        image_paths, direction_names = self.get_4_direction_images_from_cache(phase)
        
        if not image_paths:
            print("✗ 环视建图失败")
            return False, None
        
        # 获取地图路径
        # 验证时刚完成 360° 扫描，current_step 已递增，需要使用上一步的地图
        episode_dir = os.path.join(
            self.config.RESULTS_DIR, 
            f'episode_{self.current_episode_id}'
        )
        last_saved_step = self.current_step - 1
        global_map = os.path.join(episode_dir, 'global_map', f'step-{last_saved_step}.png')
        local_map = os.path.join(episode_dir, 'local_map', f'step-{last_saved_step}.png')
        
        # 验证地图文件存在
        if not os.path.exists(global_map):
            print(f"✗ Global map not found: {global_map}")
            return False, None
        
        # 创建带waypoint标注的地图副本（不覆盖原始地图）
        global_map_for_llm = global_map
        if len(self.waypoint_memory) > 0:
            global_map_img = cv2.imread(global_map)
            global_map_img = self.visualize_waypoints_on_map(global_map_img)
            # 保存为单独的可视化版本
            global_map_for_llm = os.path.join(episode_dir, 'global_map', f'step-{last_saved_step}_with_waypoints.png')
            cv2.imwrite(global_map_for_llm, global_map_img)
        
        # 获取已检测到的landmark类别
        detected_landmarks = list(self.detected_classes) if hasattr(self, 'detected_classes') else []
        
        # 获取路径点历史记录
        waypoint_summary = self.get_waypoint_summary()
        
        # 调用LLM验证（全局地图必需，局部地图可选，传递实际检测到的类别）
        response, is_completed = self.planner.verify_and_replan(
            instruction=self.current_instruction,
            current_subtask=self.current_subtask,
            observation_images=image_paths,
            direction_names=direction_names,
            global_map_image=global_map_for_llm,  # 使用带waypoint标注的版本
            local_map_image=local_map if os.path.exists(local_map) else None,
            detected_landmarks=detected_landmarks,
            waypoint_summary=waypoint_summary
        )
        
        print(f"  🏷️  Detected landmarks: {detected_landmarks if detected_landmarks else 'None'}")
        
        if not response:
            print("✗ LLM验证未返回有效响应")
            return False, None
        
        if is_completed:
            print(f"\n✅ 子任务 #{self.subtask_count} 完成!")
            
            # 检查是否是最终子任务
            if response.get('is_final_subtask', False):
                print("🎯 到达最终目的地!")
                return True, response
            
            # 清空上一个子任务的轨迹（每个子任务独立显示）
            print("  🧹 清空上一子任务轨迹...")
            self.mapper.clear_trajectory()
            
            # 更新到新子任务
            self.subtask_count += 1
            self.current_subtask = response
            self.progress_summary = ""
            
            # 更新当前位置信息（用于后续参考）
            self.current_position_info = {
                'waypoint': response.get('waypoint', 'Unknown'),
                'observation': response.get('current_observation', ''),
                'step': self.current_step
            }
            
            # 创建路径点记录（空间记忆）
            waypoint_desc = response.get('waypoint', 'Unknown location')
            # 不传position参数，让add_waypoint()从mapper.curr_loc获取正确的地图像素坐标
            self.add_waypoint(waypoint_desc)
            
            # 动态更新目标landmark（自动替换上一个子任务的landmark）
            subtask_landmark = response.get('subtask_landmark', None)
            if subtask_landmark:
                # 验证：只要在mapping_classes中（能被GroundedSAM检测）就可以作为landmark
                if subtask_landmark in self.mapping_classes:
                    # 动态更新landmark_classes（只标注当前子任务的目标）
                    # 注意：这里更新会自动替换掉上一个子任务的landmark标注
                    self.landmark_classes = [subtask_landmark]
                    self.target_landmark = subtask_landmark
                    print(f"  🎯 新目标Landmark: {self.target_landmark}")
                    print(f"  📍 已更新landmark_classes: {self.landmark_classes} (替换上一子任务)")
                else:
                    print(f"  ⚠️  警告: '{subtask_landmark}' 不在mapping_classes中，GroundedSAM无法检测")
                    self.target_landmark = None
                    self.landmark_classes = []  # 重置为空
            else:
                print(f"  ℹ️  未指定新landmark，不标注landmark")
                self.target_landmark = None
                self.landmark_classes = []  # 重置为空
            
            self._save_subtask(response, f"subtask_{self.subtask_count}")
            self._print_subtask_info(response)
        else:
            print(f"\n🔄 子任务 #{self.subtask_count} 未完成，继续...")
            
            # 即使未完成也更新位置观察（用于记录轨迹）
            if 'current_observation' in response:
                self.current_position_info = {
                    'waypoint': response.get('waypoint', getattr(self, 'current_position_info', {}).get('waypoint', 'Unknown')),
                    'observation': response.get('current_observation', ''),
                    'step': self.current_step
                }
            self.current_subtask = response
            self._save_subtask(response, f"subtask_{self.subtask_count}_refined")
        
        return is_completed, response
    
    def execute_action_with_vlm(self) -> Tuple[Optional[int], Optional[str], bool]:
        """
        使用VLM决策并执行动作
        
        Returns:
            (action_id, action_name, should_stop)
        """
        if not self.action_executor or not self.current_subtask:
            return None, None, True
        
        # 获取当前观察：使用缓存的观察或通过旋转获取
        if self.latest_obs is not None:
            obs = self.latest_obs
        else:
            # 如果没有缓存，执行一次右转再左转回来获取观察
            actions = [{"action": HabitatSimActions.TURN_RIGHT}]
            outputs = self.envs.step(actions)
            obs, _, dones, _ = [list(x) for x in zip(*outputs)]
            if dones[0]:
                print("⚠️ Episode结束")
                return None, None, True
            
            actions = [{"action": HabitatSimActions.TURN_LEFT}]
            outputs = self.envs.step(actions)
            obs, _, dones, _ = [list(x) for x in zip(*outputs)]
            if dones[0]:
                print("⚠️ Episode结束")
                return None, None, True
            obs = obs[0]
        
        # 使用上一步保存的图像（当前步图像要等step()执行后才会保存）
        episode_dir = os.path.join(
            self.config.RESULTS_DIR, 
            f'episode_{self.current_episode_id}'
        )
        last_step = self.current_step - 1  # 上一步已保存的文件
        fp_image = os.path.join(episode_dir, 'rgb', f'step-{last_step}.png')
        
        # 如果rgb/中的图像还不存在，用当前观察创建临时文件
        if not os.path.exists(fp_image):
            rgb_bgr = cv2.cvtColor(obs['rgb'], cv2.COLOR_RGB2BGR)
            temp_image = os.path.join(
                self.vlm_dir, 'observations',
                f'step{last_step}_first_person.jpg'
            )
            cv2.imwrite(temp_image, rgb_bgr)
            fp_image = temp_image
        
        # 获取当前地图路径和检测图像
        self._get_current_map_path()
        
        # 获取detection图像路径（如果存在）
        detection_image = os.path.join(episode_dir, 'detection', f'step-{last_step}.png')
        if not os.path.exists(detection_image):
            detection_image = None
        
        # 获取局部地图路径
        local_map = os.path.join(episode_dir, 'local_map', f'step-{last_step}.png')
        if not os.path.exists(local_map):
            local_map = None
        
        # 获取已检测的landmark类别
        detected_landmarks = ', '.join(self.detected_classes) if hasattr(self, 'detected_classes') and self.detected_classes else None
        
        # 调用VLM决策
        action_id, action_name, updated_progress, response = self.action_executor.decide_action(
            subtask_destination=self.current_subtask.get('subtask_destination', ''),
            subtask_instruction=self.current_subtask.get('subtask_instruction', ''),
            first_person_image=fp_image,
            action_mapping=self.ACTION_MAPPING,
            progress_summary=self.progress_summary,
            detection_image=detection_image,
            local_map_image=local_map,
            detected_landmarks=detected_landmarks
        )
        
        if action_id is None:
            print("✗ VLM决策失败")
            return None, None, True
        
        # 更新进度
        self.progress_summary = updated_progress
        
        # 检查是否停止
        should_stop = (action_name == "STOP")
        
        return action_id, action_name, should_stop
    
    def step_with_vlm(self, action: int, action_name: str = "", save_vis: bool = True) -> Dict[str, Any]:
        """
        执行VLM决策的动作（调用父类step方法）并缓存观察
        
        Args:
            action: 动作ID
            action_name: 动作名称（用于可视化）
            save_vis: 是否保存可视化
            
        Returns:
            步骤结果字典
        """
        result = self.step(action, save_vis)
        # 缓存最新观察和info用于下次VLM决策和可视化
        self.latest_obs = result.get('obs', None)
        self.latest_info = result.get('info', None)
        
        # 保存RGB+俯视图拼接可视化（使用环境提供的top_down_map_vlnce）
        if save_vis and self.obs_collector and self.latest_obs is not None:
            subtask_text = None
            if self.current_subtask:
                subtask_text = self.current_subtask.get('subtask_instruction', '')
            
            distance = 0.0
            if self.latest_info:
                distance = self.latest_info.get('distance_to_goal', 0.0)
            
            self.obs_collector.save_step_visualization(
                observations=self.latest_obs,
                info=self.latest_info or {},
                step=self.current_step,
                instruction=self.current_instruction,
                current_subtask=subtask_text,
                distance=distance,
                action=action_name
            )
        
        return result
    
    def run_vlm_navigation(self, max_steps: int = 500, 
                          max_subtask_steps: int = 50,
                          verify_interval: int = 10) -> Dict[str, Any]:
        """
        运行完整的VLM导航流程
        
        Args:
            max_steps: 最大总步数
            max_subtask_steps: 每个子任务最大步数
            verify_interval: 验证间隔步数
            
        Returns:
            导航结果字典
        """
        print("\n" + "="*60)
        print("🚀 启动VLM自动导航")
        print("="*60)
        print(f"📝 指令: {self.current_instruction}")
        print(f"⚙️  最大步数: {max_steps} | 子任务步数: {max_subtask_steps} | 验证间隔: {verify_interval}")
        print("="*60 + "\n")
        
        # 1. 环视建图 + 收集观察
        self.look_around_and_collect()
        
        # 2. 生成初始子任务
        subtask = self.generate_initial_subtask()
        if not subtask:
            print("✗ 无法生成初始子任务")
            return {
                'success': False,
                'total_steps': self.current_step,
                'subtask_count': 0,
                'detected_classes': list(self.detected_classes) if hasattr(self, 'detected_classes') else [],
                'gif_path': None,
                'result_file': None,
                'reason': 'initial_subtask_failed'
            }
        
        # 3. 主导航循环
        total_steps = self.current_step
        subtask_steps = 0
        navigation_complete = False
        
        while total_steps < max_steps:
            # VLM决策动作
            action_id, action_name, should_stop = self.execute_action_with_vlm()
            
            if action_id is None:
                print("✗ VLM决策失败，尝试手动输入...")
                action_id = self.get_keyboard_action()
                action_name = self._action_name(action_id)
                should_stop = (action_id == 0)
            
            # 如果VLM决定停止 → 验证子任务
            if should_stop:
                is_completed, new_subtask = self.verify_and_replan()
                
                if is_completed and new_subtask and new_subtask.get('is_final_subtask', False):
                    print("\n🎯 导航完成！")
                    navigation_complete = True
                    break
                
                subtask_steps = 0
                continue
            
            # 执行动作（传入action_name用于可视化）
            result = self.step_with_vlm(action_id, action_name=action_name, save_vis=True)
            total_steps = self.current_step
            subtask_steps += 1
            
            print(f"[Step {total_steps}] {action_name} | 子任务步数: {subtask_steps}")
            
            if result['done']:
                print("\n✅ Episode自动完成")
                navigation_complete = True
                break
            
            # 定期验证
            if subtask_steps >= verify_interval:
                is_completed, _ = self.verify_and_replan()
                if is_completed:
                    subtask_steps = 0
            
            # 子任务超时
            if subtask_steps >= max_subtask_steps:
                print(f"\n⚠️ 子任务超时 ({max_subtask_steps}步)，重新规划...")
                _, _ = self.verify_and_replan()
                subtask_steps = 0
        
        # 4. 保存GIF动画
        gif_path = None
        if self.obs_collector:
            gif_path = self.obs_collector.save_gif(fps=2)
        
        # 5. 保存结果（供后续测评）
        final_result = self._save_navigation_result(navigation_complete, total_steps)
        
        return {
            'success': navigation_complete,
            'total_steps': total_steps,
            'subtask_count': self.subtask_count,
            'detected_classes': list(self.detected_classes),
            'gif_path': gif_path,
            'result_file': final_result
        }
    
    def _save_subtask(self, subtask: Dict, name: str):
        """保存子任务到文件"""
        filepath = os.path.join(self.vlm_dir, 'subtasks', f'{name}.json')
        
        data = {
            'episode_id': self.current_episode_id,
            'instruction': self.current_instruction,
            'subtask_id': self.subtask_count,
            'step': self.current_step,
            'timestamp': datetime.now().isoformat(),
            'subtask': subtask
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        self.subtask_history.append(data)
        self.current_subtask_file = filepath
    
    def _save_navigation_result(self, success: bool, total_steps: int) -> str:
        """
        保存导航结果（供后续测评）
        
        Args:
            success: 是否成功
            total_steps: 总步数
            
        Returns:
            结果文件路径
        """
        # 尝试获取评测指标（如果有的话）
        metrics = {}
        if self.latest_info:
            metrics = {
                'distance_to_goal': self.latest_info.get('distance_to_goal', -1),
                'success': self.latest_info.get('success', success),
                'spl': self.latest_info.get('spl', 0.0),
                'path_length': self.latest_info.get('path_length', 0.0),
                'oracle_success': self.latest_info.get('oracle_success', False)
            }
        
        result = {
            'episode_id': self.current_episode_id,
            'instruction': self.current_instruction,
            'success': success,
            'total_steps': total_steps,
            'subtask_count': self.subtask_count,
            'detected_classes': list(self.detected_classes),
            'subtask_history': self.subtask_history,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        # 保存到vlm目录
        filepath = os.path.join(self.vlm_dir, 'result.json')
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*60}")
        print(f"📁 结果已保存: {filepath}")
        print(f"   Steps: {total_steps} | Subtasks: {self.subtask_count}")
        if metrics:
            print(f"   Success: {metrics.get('success', success)} | SPL: {metrics.get('spl', 0.0):.4f}")
        print(f"{'='*60}")
        
        return filepath
    
    def record_action(self, action_name: str, action_id: int, vlm_response: Dict = None):
        """
        记录动作到当前子任务文件（与llm_vlm_control兼容的格式）
        
        Args:
            action_name: 动作名称
            action_id: 动作ID
            vlm_response: VLM响应字典（可选）
        """
        if not self.current_subtask_file or not os.path.exists(self.current_subtask_file):
            return
        
        action_data = {
            "step": self.current_step,
            "action_name": action_name,
            "action_id": action_id,
        }
        
        if self.latest_info:
            action_data["distance_to_goal"] = self.latest_info.get("distance_to_goal", -1)
        
        if vlm_response:
            action_data["vlm_response"] = {
                k: vlm_response.get(k, "") 
                for k in ['observation', 'reasoning', 'action', 'progress_summary']
            }
        
        # 读取并更新文件
        try:
            with open(self.current_subtask_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            if "actions" not in data:
                data["actions"] = []
            data["actions"].append(action_data)
            
            with open(self.current_subtask_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ 记录动作失败: {e}")
    
    def _print_subtask_info(self, response: Dict, is_initial: bool = False):
        """打印子任务信息"""
        title = "初始子任务" if is_initial else f"子任务 #{self.subtask_count}"
        print(f"\n✅ ===== {title} =====")
        print(f"🌍 全局指令: {self.current_instruction}")
        print(f"📍 Waypoint: {response.get('waypoint', 'N/A')}")
        print(f"👁️  环境观察: {response.get('current_observation', 'N/A')}")
        print(f"🎯 目的地: {response.get('subtask_destination', 'N/A')}")
        print(f"🏷️  目标Landmark: {response.get('subtask_destination_landmark', 'N/A')}")
        print(f"📋 子任务指令: {response.get('subtask_instruction', 'N/A')}")
        print(f"💡 规划提示: {response.get('planning_hints', 'N/A')}")
        print(f"✓ 完成条件: {response.get('completion_criteria', 'N/A')}")
        print(f"🏁 是否最终: {response.get('is_final_subtask', False)}")
        print(f"✅ {'='*50}\n")
    def add_waypoint(self, waypoint_description: str, position: np.ndarray = None) -> int:
        """
        添加路径点到空间记忆
        
        Args:
            waypoint_description: 路径点描述（格式: "<Area Type> - <Key Landmarks>"）
            position: 当前位置地图像素坐标（如果为None则从mapper获取）
            
        Returns:
            waypoint_id: 新添加的路径点ID
        """
        self.waypoint_counter += 1
        waypoint_id = self.waypoint_counter
        
        # 获取当前位置（地图像素坐标，用于可视化）
        if position is None:
            # 从mapper获取当前地图像素坐标 [x, y, theta]
            if hasattr(self.mapper, 'curr_loc'):
                position = self.mapper.curr_loc.copy()  # 这是地图像素坐标
            else:
                position = np.array([0, 0, 0])  # 默认值
        
        # 创建路径点记录
        waypoint = {
            "id": waypoint_id,
            "waypoint": waypoint_description,
            "position": position.tolist() if isinstance(position, np.ndarray) else position,
            "step": self.current_step,
            "detected_objects": list(self.detected_classes) if hasattr(self, 'detected_classes') else [],
            "subtask_id": self.subtask_count
        }
        
        self.waypoint_memory.append(waypoint)
        
        print(f"  📍 Waypoint #{waypoint_id} 已记录: {waypoint_description}")
        
        # 保存到文件
        self._save_waypoint_memory()
        
        return waypoint_id
    
    def get_waypoint_summary(self) -> str:
        """
        获取路径点摘要（用于LLM提示词）
        
        Returns:
            路径点摘要字符串
        """
        if not self.waypoint_memory:
            return "No waypoints recorded yet."
        
        summary_lines = ["Recorded Waypoints:"]
        for wp in self.waypoint_memory:
            # 格式: "1. Living Room - near sofa and TV (Step 12)"
            summary_lines.append(
                f"{wp['id']}. {wp['waypoint']} (Step {wp['step']})"
            )
        
        return "\n".join(summary_lines)
    
    def _save_waypoint_memory(self):
        """保存路径点记忆到JSON文件"""
        waypoint_file = os.path.join(self.vlm_dir, 'waypoint_memory.json')
        
        data = {
            "episode_id": self.current_episode_id,
            "instruction": self.current_instruction,
            "waypoints": self.waypoint_memory,
            "total_waypoints": len(self.waypoint_memory),
            "last_updated_step": self.current_step
        }
        
        try:
            with open(waypoint_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️  保存waypoint记忆失败: {e}")
    
    def visualize_waypoints_on_map(self, map_image: np.ndarray) -> np.ndarray:
        """
        在地图上可视化路径点
        
        Args:
            map_image: 输入地图图像（BGR格式）
            
        Returns:
            标注了路径点的地图图像
        """
        if not self.waypoint_memory:
            return map_image
        
        # 复制地图避免修改原图
        annotated_map = map_image.copy()
        
        for wp in self.waypoint_memory:
            try:
                # 获取位置坐标（地图像素坐标）
                pos = wp["position"]
                if isinstance(pos, list):
                    x, y = int(pos[0]), int(pos[1])
                else:
                    continue
                
                # 绘制圆形标记（深红色）
                cv2.circle(annotated_map, (x, y), 15, (0, 0, 139), -1)  # 深红色填充圆 (BGR: 0,0,139)
                cv2.circle(annotated_map, (x, y), 15, (255, 255, 255), 2)  # 白色边框
                
                # 绘制ID数字（白色）
                font = cv2.FONT_HERSHEY_SIMPLEX
                text = str(wp["id"])
                text_size = cv2.getTextSize(text, font, 0.6, 2)[0]
                text_x = x - text_size[0] // 2
                text_y = y + text_size[1] // 2
                cv2.putText(annotated_map, text, (text_x, text_y), 
                           font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                
            except Exception as e:
                print(f"⚠️  标注waypoint {wp['id']} 失败: {e}")
                continue
        
        return annotated_map
