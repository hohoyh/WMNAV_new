import gzip
import json
import logging
import math
import os
import random
import requests
import traceback
import habitat_sim

import pandas as pd
import numpy as np

from PIL import Image
from simWrapper import PolarAction, SimWrapper
from WMNav_agent import *
from custom_agent import *
from utils import *

class Env:
    """
    Base class for creating an environment for embodied navigation tasks.
    This class defines the setup, logging, running, and evaluation of episodes.
    """

    task = 'Not defined'

    def __init__(self, cfg: dict):
        """
        Initializes the environment with the provided configuration.

        Args:
            cfg (dict): Configuration dictionary containing environment, simulation, and agent settings.
        """
        self.cfg = cfg['env_cfg']
        self.sim_cfg = cfg['sim_cfg']
        if self.cfg['name'] == 'default':
            self.cfg['name'] = f'default_{random.randint(0, 1000)}'
        self._initialize_logging(cfg)
        self._initialize_agent(cfg)
        self.outer_run_name = self.task + '_' + self.cfg['name']
        self.inner_run_name = f'{self.cfg["instance"]}_of_{self.cfg["instances"]}'
        self.curr_run_name = "Not initialized"
        self.path_calculator = habitat_sim.MultiGoalShortestPath()
        self.simWrapper = None  # 修改self.simWrapper: SimWrapper = None
        self.num_episodes = 0
        self._initialize_experiment()

    def _initialize_agent(self, cfg: dict):
        """Initializes the agent for the environment."""
        PolarAction.default = PolarAction(cfg['agent_cfg']['default_action'], 0, 'default')
        cfg['agent_cfg']['sensor_cfg'] = cfg['sim_cfg']['sensor_cfg']
        agent_cls = globals()[cfg['agent_cls']]
        self.agent: Agent = agent_cls(cfg['agent_cfg'])
        self.agent_cls = cfg['agent_cls']

    def _initialize_logging(self, cfg: dict):
        """
        Initializes logging for the environment.

        Args:
            cfg (dict): Configuration dictionary containing logging settings.
        """
        self.log_file = os.path.join(os.environ.get("LOG_DIR"), f'{cfg["task"]}_{self.cfg["name"]}/{self.cfg["instance"]}_of_{self.cfg["instances"]}.txt')
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        if self.cfg['parallel']:
            logging.basicConfig(
                filename=self.log_file,
                level=logging.INFO,
                format='%(asctime)s %(levelname)s: %(message)s'
            )
        else:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s %(levelname)s: %(message)s'
            )

    def _initialize_experiment(self):
        """
        Abstract method for setting up the environment and initializing all required variables.
        Should be implemented in derived classes.
        """
        raise NotImplementedError

    def run_experiment(self):
        """
        Runs the experiment by iterating over episodes.
        """
        instance_size = math.ceil(self.num_episodes / self.cfg['instances'])  # 1000
        start_ndx = self.cfg['instance'] * instance_size
        end_ndx = self.num_episodes

        for episode_ndx in range(start_ndx, min(start_ndx + self.cfg['num_episodes'], end_ndx)):

            self.wandb_log_data = {
                'episode_ndx': episode_ndx,  # 0
                'instance': self.inner_run_name,  # 0_of_1
                'total_episodes': self.cfg['instances'] * self.cfg['num_episodes'],  # 1
                'task': self.task,  # ObjectNav
                'task_data': {},
                'spl': 0,
                'goal_reached': False
            }

            try:
                self._run_episode(episode_ndx)
            except Exception as e:
                log_exception(e)
                self.simWrapper.reset()


    def _run_episode(self, episode_ndx: int):
        """
        Runs a single episode.p

        Args:
            episode_ndx (int): The index of the episode to run.
        """
        obs = self._initialize_episode(episode_ndx)  # color_sensor(1080, 1920, 4) depth_sensor(1080, 1920) agent_state[position rotation sensor_states]

        logging.info(f'\n===================STARTING RUN: {self.curr_run_name} ===================\n')
        for _ in range(self.cfg['max_steps']):
            try:
                agent_action = self._step_env(obs)  # 根据单张RGB图片、深度图和agent以及相机位姿确定agent的下一步动作，保存运行结果
                if agent_action is None:
                    break
                obs = self.simWrapper.step(agent_action)  # 执行操作，更新agent的状态和观察

            except Exception as e:
                log_exception(e)

            finally:
                self.step += 1
        self._post_episode()

    def _initialize_episode(self, episode_ndx: int):
        """
        Initializes the episode. This method should be implemented in derived classes.

        Args:
            episode_ndx (int): The index of the episode to initialize.
        """
        self.step = 0
        self.init_pos = None
        self.df = pd.DataFrame({})
        self.agent_distance_traveled = 0
        self.prev_agent_position = None

    def _step_env(self, obs: dict):
        """
        Takes a step in the environment. This method should be implemented in derived classes.

        Args:
            obs (dict): The current observation. Contains agent state and sensor observations.

        Returns:
            PolarAction: The next action to be taken by the agent.
        """
        logging.info(f'Step {self.step}')
        agent_state = obs['agent_state']
        if self.prev_agent_position is not None:
            self.agent_distance_traveled += np.linalg.norm(agent_state.position - self.prev_agent_position)
        self.prev_agent_position = agent_state.position

        return None

    def _post_episode(self):
        """
        Called after the episode is complete, saves the dataframe log, and resets the environment.
        Sends a request to the aggregator server if parallel is set to True.
        """
        self.df.to_pickle(os.path.join(os.environ.get("LOG_DIR"), f'{self.outer_run_name}/{self.inner_run_name}/{self.curr_run_name}/df_results.pkl'))
        self.simWrapper.reset()
        self.agent.reset()
        if self.cfg['parallel']:
            try:
                self.wandb_log_data['spend'] = self.agent.get_spend()
                self.wandb_log_data['default_rate'] = len(self.df[self.df['success'] == 0]) / len(self.df)
                response = requests.post(f'http://localhost:{self.cfg["port"]}/log', json=self.wandb_log_data)
                if response.status_code != 200:
                    logging.error(f"Failed to send metrics: {response.text}")
            except Exception as e:
                tb = traceback.extract_tb(e.__traceback__)
                for frame in tb:
                    logging.error(f"Frame {frame.filename} line {frame.lineno}")
                logging.error(e)

        logging.info(f"Success: {self.wandb_log_data['goal_reached']}")
        logging.info('\n===================RUN COMPLETE===================\n')
        if self.cfg['log_freq'] == 1:
            create_gif(
                os.path.join(os.environ.get("LOG_DIR"), f'{self.outer_run_name}/{self.inner_run_name}/{self.curr_run_name}'), self.agent.cfg['sensor_cfg']['img_height'], self.agent.cfg['sensor_cfg']['img_width'], agent_cls=self.agent_cls
            )
            create_gif_voxel(
                os.path.join(os.environ.get("LOG_DIR"), f'{self.outer_run_name}/{self.inner_run_name}/{self.curr_run_name}'),
                1800, 1800
            )

    def _log(self, images: dict, step_metadata: dict, logging_data: dict):
        """
        Appends the step metadata to the dataframe, and saves the images and general metadata to disk.

        Args:
            images (dict): Images generated during the step.
            step_metadata (dict): Metadata for the current step.
            logging_data (dict): General logging data.
        """
        self.df = pd.concat([self.df, pd.DataFrame([step_metadata])], ignore_index=True)

        if self.step % self.cfg['log_freq'] == 0 or step_metadata['success'] == 0:
            path = os.path.join(os.environ.get("LOG_DIR"), f'{self.outer_run_name}/{self.inner_run_name}/{self.curr_run_name}/step{self.step}')
            if not step_metadata['success']:
                path += '_ERROR'
            os.makedirs(path, exist_ok=True)
            for name, im in images.items():
                if im is not None:
                    im = Image.fromarray(im[:, :, 0:3], mode='RGB')
                    im.save(f'{path}/{name}.png')
            with open(f'{path}/details.txt', 'w') as file:
                if step_metadata['success']:
                    for k, v in logging_data.items():
                        file.write(f'{k}\n{v}\n\n')

    def _calculate_metrics(self, agent_state: habitat_sim.AgentState, agent_action: PolarAction, geodesic_path: float, max_steps: int):
        metrics = {}
        self.path_calculator.requested_start = agent_state.position
        
        # 尝试计算距离 (单目标时有用，多目标时仅作参考)
        try:
            metrics['distance_to_goal'] = self.simWrapper.get_path(self.path_calculator)
        except Exception:
            metrics['distance_to_goal'] = 100.0 # 默认大值

        metrics['spl'] = 0.0
        metrics['goal_reached'] = False
        metrics['done'] = False
        metrics['finish_status'] = 'running'

        # ✅ 1. 获取多目标标志 (从 current_episode 获取更稳健)
        is_multi = self.current_episode.get('is_multi_object', False)
        num_found = 0 # 初始化变量，防止作用域错误

        # ✅ 2. 多目标状态检测
        if is_multi:
            # 获取状态列表引用
            found_targets = self.current_episode['found_targets']
            objects = self.current_episode['objects']
            
            # 检查是否有新目标被发现
            for i, goal_category in enumerate(objects):
                if not found_targets[i]:
                    # 检查是否可见 (基于距离阈值)
                    if self._check_goal_visible(agent_state.position, goal_category):
                        found_targets[i] = True
                        logging.info(f"🎉 Target Found: {goal_category}")
                        # =======================================================
                        # 🟢【新增】找到目标后，立即重置 Agent 的探索记忆
                        # =======================================================
                        logging.info(f"Resetting exploration state for next target...")
                        # 重置方向锁定状态
                        self.agent._locked_idx = None
                        self.agent._locked_steps = 0

                        # # 1. 重置好奇心地图 (所有方向恢复最高分10，鼓励重新探索)
                        # if hasattr(self.agent, 'cvalue_map') and self.agent.cvalue_map is not None:
                        #     self.agent.cvalue_map.fill(10)
                        
                        # # 2. 重置已探索地图 (让 Agent 认为周围环境是新的，可以再次经过)
                        # if hasattr(self.agent, 'explored_map') and self.agent.explored_map is not None:
                        #     self.agent.explored_map.fill(0)
                        
                        # 3. 清空 VLM 锁定的局部目标点
                        if hasattr(self.agent, 'goal_position'):
                            self.agent.goal_position = []
                        
                        # 4. 重置转身冷却 (允许 Agent 立即掉头去别处)
                        if hasattr(self.agent, 'turned'):
                            self.agent.turned = -100 
                        # 🟢【新增】5. 清除防卡死历史，防止刚找到目标就误触 God Mode 乱跳
                        if hasattr(self.agent, 'position_history'):
                            self.agent.position_history = []
                        if hasattr(self.agent, 'stuck_counter'):
                            self.agent.stuck_counter = 0
                        # =======================================================
            
            # 更新统计
            self.current_episode['found_targets'] = found_targets
            num_found = sum(found_targets)
            total_objects = len(found_targets)
            
            metrics['partial_success_rate'] = num_found / total_objects
            metrics['goal_reached'] = all(found_targets)
            
        else:
            # 单目标逻辑 (保持 pass，具体判定在下面 done 的时候做)
            # 🔴 核心修复：单目标也要进行动态距离检测！
            found_targets = self.current_episode['found_targets']
            goal_category = self.current_episode['object']
            
            if not found_targets[0]:
                if self._check_goal_visible(agent_state.position, goal_category):
                    found_targets[0] = True
                    logging.info(f"🎉 Single Target Found: {goal_category}")
            
            self.current_episode['found_targets'] = found_targets
            num_found = sum(found_targets)
        # ✅ 3. 结束判定与 SPL 计算
        if agent_action is PolarAction.stop or self.step + 1 == max_steps:
            metrics['done'] = True

            if is_multi:
                # --- 多目标 SPL 计算 ---
                total_objects = self.current_episode['num_objects']
                if num_found > 0:
                    success_rate = num_found / total_objects
                    # SPL = 成功率 * (最短路径 / max(最短路径, 实际路径))
                    path_ratio = geodesic_path / max(geodesic_path, self.agent_distance_traveled)
                    metrics['spl'] = success_rate * path_ratio
                    
                    metrics['finish_status'] = 'success' if metrics['goal_reached'] else 'partial'
                else:
                    metrics['finish_status'] = 'fp' if agent_action is PolarAction.stop else 'max_steps'
            else:
                # --- 单目标 SPL 计算 ---
                # 单目标判断标准：距离 < 阈值
                if metrics['distance_to_goal'] < self.cfg['success_threshold']:
                    metrics['finish_status'] = 'success'
                    metrics['goal_reached'] = True
                    metrics['spl'] = geodesic_path / max(geodesic_path, self.agent_distance_traveled)
                else:
                    metrics['finish_status'] = 'fp' if agent_action is PolarAction.stop else 'max_steps'

            # 更新 WandB 日志
            self.wandb_log_data.update({
                'spl': metrics['spl'],
                'goal_reached': metrics['goal_reached']
            })

        return metrics
    
    def _calculate_multi_spl(self, found_objects, geodesic_path, actual_distance):
        """
        ✅ 新增：计算多目标 SPL
        
        多目标 SPL = (最短路径总长 / 实际路径长度) × 成功率
        
        Args:
            found_objects: 找到的目标列表
            geodesic_path: 所有目标的最短路径总长
            actual_distance: 实际行走距离
        
        Returns:
            spl: Success weighted by Path Length
        """
        num_found = len(found_objects)
        num_total = self.current_episode['num_objects']
        
        if num_found == 0:
            return 0.0
        
        # 成功率
        success_rate = num_found / num_total
        
        # 路径效率
        path_efficiency = geodesic_path / max(geodesic_path, actual_distance)
        
        # 多目标 SPL
        multi_spl = success_rate * path_efficiency
        
        return multi_spl

    def _check_goal_visible(self, agent_position, goal_category, threshold=1.0):

        """兼容单目标(List)和多目标(Dict)，支持模糊名称匹配"""

        # 关键词映射：数据集里对象名可能是 armchair/office chair 等变体
        category_keywords = {
            'chair':      ['chair', 'armchair', 'stool'],
            'toilet':     ['toilet'],
            'bed':        ['bed'],
            'sofa':       ['sofa', 'couch', 'loveseat'],
            'plant':      ['plant'],
            'tv_monitor': ['tv', 'monitor', 'television', 'screen'],
        }
        keywords = category_keywords.get(goal_category, [goal_category.replace('_', ' ')])

        if self.current_episode.get('is_multi_object', False):
            obj_positions_dict = self.current_episode.get('object_positions', {})
            # 模糊匹配：key包含keyword就算
            goal_positions = []
            for key, positions in obj_positions_dict.items():
                key_low = key.lower().replace('_', ' ')
                if any(kw in key_low for kw in keywords):
                    goal_positions.extend(positions)
            if not goal_positions:
                goal_positions = obj_positions_dict.get(goal_category, [])
        else:
            goal_positions = self.current_episode.get('object_positions', [])

        # 动态阈值
        dynamic_threshold = self.cfg['success_threshold']
        if goal_category in ['bed', 'sofa', 'couch']:
            dynamic_threshold = max(dynamic_threshold, 2.0)
        elif goal_category in ['chair', 'plant', 'toilet', 'tv_monitor', 'tv screen']:
            dynamic_threshold = max(dynamic_threshold, 1.5)

        if not goal_positions:
            logging.warning(f"[CHECK] No positions found for '{goal_category}', available: {list(self.current_episode.get('object_positions', {}).keys())}")
            return False

        min_dist = min(np.linalg.norm(np.array(agent_position) - np.array(gp)) for gp in goal_positions)
        logging.info(f"[CHECK] '{goal_category}' min_distance={min_dist:.2f}m threshold={dynamic_threshold}m")

        for goal_pos in goal_positions:
            if np.linalg.norm(np.array(agent_position) - np.array(goal_pos)) < dynamic_threshold:
                return True
        return False

class WMNavEnv(Env):

    task = 'ObjectNav'

    def _initialize_experiment(self):
        """
        Initializes the experiment by setting up the dataset split, scene configuration, and goals.
        """
        self.all_episodes = []

        # ✅ 唯一的 dataset 判断逻辑 (删除了冗余的旧代码)
        if self.cfg['dataset'] == 'multi_hm3d_v0.2':
            # 多目标配置
            scene_config_path = 'hm3d_v0.2/hm3d_annotated_basis.scene_dataset_config.json'
            objnav_path = 'multi_objectnav_hm3d'
            self.is_multi_object = True
        # ==========================================
        # 🔴 必须确保有这段 OneMap 官方多目标数据集配置
        # ==========================================
        elif self.cfg['dataset'] == 'objectnav_hm3d_multi_v2':
            scene_config_path = 'hm3d_v0.2/hm3d_annotated_basis.scene_dataset_config.json'
            objnav_path = 'objectnav_hm3d_multi_v2' 
            self.is_multi_object = True
        # ==========================================
        elif self.cfg['dataset'] == 'hm3d_v0.2':
            # 单目标 v0.2 配置
            scene_config_path = 'hm3d_v0.2/hm3d_annotated_basis.scene_dataset_config.json'
            objnav_path = 'objectnav_hm3d_v2'
            self.is_multi_object = False
        elif self.cfg['dataset'] == 'hm3d_v0.1':
            # 单目标 v0.1 配置
            scene_config_path = 'hm3d_v0.1/hm3d_annotated_basis.scene_dataset_config.json'
            objnav_path = 'objectnav_hm3d_v1'
            self.is_multi_object = False
        elif self.cfg['dataset'] == 'mp3d':
            # MP3D 配置
            scene_config_path = 'mp3d/mp3d_annotated_basis.scene_dataset_config.json'
            objnav_path = 'objectnav_mp3d'
            self.is_multi_object = False
        else:
            raise ValueError(f"Unknown dataset type: {self.cfg['dataset']}")

        self.sim_cfg['scene_config'] = os.path.join(os.environ.get("DATASET_ROOT"), scene_config_path)
        self.goals = {}

        # ✅ 加载数据
        if self.is_multi_object:
            # 多目标：加载 val.json.gz（episode 列表）
            val_file = os.path.join(os.environ.get("DATASET_ROOT"), objnav_path, f'{self.cfg["split"]}/val.json.gz')
            logging.info(f"Loading multi-object dataset from: {val_file}")
            
            with gzip.open(val_file, 'rt') as gz:
                js = json.load(gz)
                self.all_episodes += js['episodes']
            
            # ✅ 同时加载 goals_by_category（用于获取物体位置和观察点）
            # 多目标数据集的 episode 文件中没有位置信息，需要从单目标的 goals_by_category 获取
            content_dir = os.path.join(os.environ.get("DATASET_ROOT"), 'objectnav_hm3d_v2', f'{self.cfg["split"]}/content')
            logging.info(f"Loading goals_by_category from: {content_dir}")
            
            for f in sorted(os.listdir(content_dir)):
                with gzip.open(os.path.join(content_dir, f), 'rt') as gz:
                    js = json.load(gz)
                    hsh = f.split('.')[0]
                    self.goals[hsh] = js['goals_by_category']
        else:
            # 单目标：加载 content 目录下的所有文件
            content_dir = os.path.join(os.environ.get("DATASET_ROOT"), objnav_path, f'{self.cfg["split"]}/content')
            logging.info(f"Loading single-object dataset from: {content_dir}")
            
            for f in sorted(os.listdir(content_dir)):
                with gzip.open(os.path.join(content_dir, f), 'rt') as gz:
                    js = json.load(gz)
                    hsh = f.split('.')[0]
                    self.goals[hsh] = js['goals_by_category']
                    self.all_episodes += js['episodes']
        
        self.num_episodes = len(self.all_episodes)
        logging.info(f"Loaded {self.num_episodes} episodes. Mode: {'Multi-Object' if self.is_multi_object else 'Single-Object'}")

    def _initialize_episode(self, episode_ndx: int):
            """
            Initializes the episode for the BASE task.
            """
            super()._initialize_episode(episode_ndx)
            episode = self.all_episodes[episode_ndx]
            
            # ✅ Step 1: 初始化场景
            if 'hm3d' in self.cfg['dataset']:
                f = episode['scene_id'].split('/')[1:]
                self.sim_cfg['scene_id'] = f[1][2:5]
                self.sim_cfg['scene_path'] = os.path.join(
                    os.environ.get("DATASET_ROOT"), 
                    'hm3d_v0.1' if self.cfg['dataset'] == 'hm3d_v0.1' else 'hm3d_v0.2', 
                    f'{self.cfg["split"]}/{f[1]}/{f[2]}'
                )
                self.simWrapper = SimWrapper(self.sim_cfg)
                scene_id_key = f[-1]
                
            elif 'mp3d' in self.cfg['dataset']:
                self.sim_cfg['scene_id'] = episode['scene_id'].split('/')[1]
                self.sim_cfg['scene_path'] = os.path.join(os.environ.get("DATASET_ROOT"), f'{episode["scene_id"]}')
                self.simWrapper = SimWrapper(self.sim_cfg)
                scene_id_key = episode["scene_id"].split("/")[2]
            else:
                raise ValueError('Dataset type must be hm3d_v0.1, hm3d_v0.2, or mp3d')
            
            # 🔴 核心修复 2：把初始位置的设置提前！防止 dummy_target 报错
            self.init_pos = np.array(episode['start_position'])
            self.simWrapper.set_state(pos=self.init_pos, quat=episode['start_rotation'])
            self.curr_run_name = f"{episode_ndx}_{self.simWrapper.scene_id}"

            # ✅ Step 2: 根据是否多目标分别处理
            if self.is_multi_object:
                # === 多目标模式 ===
                # 🔴 核心修复 1：智能解析目标列表，防止出现 [['a', 'b']] 的套娃
                raw_categories = episode.get('object_categories', episode.get('object_category'))
                if isinstance(raw_categories, str):
                    object_categories = [raw_categories]
                elif isinstance(raw_categories, list):
                    object_categories = raw_categories
                else:
                    object_categories = []
                    
                object_categories = ['tv screen' if cat == 'tv_monitor' else cat for cat in object_categories]
                
                # ✅ 从 goals_by_category 获取真实位置和观察点
                goals = self.goals[f[1][6:]]  # 使用场景哈希值
                
                all_view_positions = []
                object_positions_dict = {}
                
                for obj_cat in object_categories:
                    # 翻译回真名，去底层查坐标
                    raw_cat = 'tv_monitor' if obj_cat == 'tv screen' else obj_cat
                    obj_key = f'{scene_id_key}_{raw_cat}'
                    display_cat = 'tv screen' if obj_cat == 'tv_monitor' else obj_cat
                    
                    if obj_key in goals:
                        all_objects = goals[obj_key]
                        object_positions_dict[display_cat] = [obj['position'] for obj in all_objects]
                        
                        # 收集观察点
                        for obj in all_objects:
                            for vp in obj.get('view_points', []):
                                all_view_positions.append(vp['agent_state']['position'])
                    else:
                        logging.warning(f"Object category '{raw_cat}' not found in goals_by_category")
                
                print(f"[DEBUG] Multi-Object Episode: {object_categories}")
                print(f"[DEBUG] Found {len(all_view_positions)} view positions")
                
                self.current_episode = {
                    'objects': object_categories,
                    'num_objects': len(object_categories),
                    'shortest_path': episode['info']['geodesic_distance'],
                    'object_positions': object_positions_dict,
                    'view_positions': all_view_positions,
                    'is_multi_object': True,
                    'found_targets': [False] * len(object_categories)
                }
                
                # ✅ 设置路径规划器的目标点
                if all_view_positions:
                    self.path_calculator.requested_ends = np.array(all_view_positions, dtype=np.float32)
                else:
                    dummy_target = self.init_pos + np.array([100.0, 0.0, 100.0])
                    self.path_calculator.requested_ends = np.array([dummy_target], dtype=np.float32)
                
            else:
                # === 单目标模式 ===
                goals = self.goals[f[1][6:]]
                raw_cat = 'tv_monitor' if episode["object_category"] == 'tv screen' else episode["object_category"]
                obj_key = f'{scene_id_key}_{raw_cat}'
                
                if obj_key in goals:
                    all_objects = goals[obj_key]
                    view_positions = []
                    for obj in all_objects:
                        for vp in obj['view_points']:
                            view_positions.append(vp['agent_state']['position'])
                    
                    self.path_calculator.requested_ends = np.array(view_positions, dtype=np.float32)
                    
                    display_cat = 'tv screen' if episode['object_category'] == 'tv_monitor' else episode['object_category']
                    print(f"[DEBUG] Single-Object Episode: {display_cat}")
                    
                    self.current_episode = {
                        'object': display_cat,
                        'shortest_path': episode['info']['geodesic_distance'],
                        'object_positions': [obj['position'] for obj in all_objects],
                        'view_positions': view_positions,
                        'is_multi_object': False,
                        'found_targets': [False]
                    }
                else:
                    logging.error(f"KeyError: {obj_key} not found in single object dataset!")

            obs = self.simWrapper.step(PolarAction.null)
            self.previous_subtask = '{}'
            
            # 重置 Agent
            self.agent.reset()
            
            return obs

    def _step_env(self, obs: dict):
            """
            Takes a step in the environment.
            """
            episode_images = [(obs['color_sensor'].copy())[:, :, :3]]
            color_origin = episode_images[0]
            loop_action_clockwise = PolarAction(0, -0.167 * np.pi)
            loop_action_counterclock = PolarAction(0, 0.167 * np.pi)

            # 1. 采集全景图
            for _ in range(11):
                obs = self.simWrapper.step(loop_action_clockwise)
                if _ % 2 == 0:
                    self.agent.navigability(obs, _+1)
                episode_images.append((obs['color_sensor'].copy())[:, :, :3])
            nav_map = self.agent.generate_voxel(obs['agent_state'])
            # ==========================================
            # 🔴 核心修复 1：补齐最后 30 度，让 Agent 物理朝向完美归零！
            # ==========================================
            obs = self.simWrapper.step(loop_action_clockwise)
            
            # ✅ 2. 准备数据
            is_multi = self.current_episode.get('is_multi_object', False)
            
            if is_multi:
                goals_list = self.current_episode['objects']
                # 使用 found_targets (布尔列表)
                found_list = self.current_episode['found_targets']
                num_objects = self.current_episode['num_objects']
                
                try:
                    # 找到第一个 False 的索引
                    idx = found_list.index(False)
                    current_goal = goals_list[idx]
                except ValueError:
                    current_goal = goals_list[-1]
            else:
                current_goal = self.current_episode['object']
                goals_list = [current_goal]
                # found_list = [False]
                found_list = self.current_episode['found_targets']
                num_objects = 1
            
            # ✅ 3. 强制类型转换：防止 'int' is not iterable 报错
            goals_list_str = [str(g) for g in goals_list]

          
            # ✅ 在第737行后添加：构建 goal_num 字典（只构建一次）
            goal_num = {
                'total': num_objects,
                'found': sum(found_list),
                'remaining': [g for i, g in enumerate(goals_list_str) if not found_list[i]]
            } if num_objects > 1 else None

            # ✅ 4. 调用 Agent (make_curiosity_value) - 第738-742行修改为：
            panoramic_image, explorable_value, reason = self.agent.make_curiosity_value(
                episode_images[-12:], 
                goals_list_str, 
                goal_num  # ✅ 使用 goal_num
            )

            # ✅ 5. 调用 Agent (update_curiosity_value)
            # 核心修复：接收 3 个返回值 (使用 _ 接收 active_target)
            goal_rotate, goal_reason, _ = self.agent.update_curiosity_value(
                explorable_value, 
                reason, 
                goals_list_str, 
                found_list
            )
            # # =================================================================
            # # 🔴 核心修复：物理级记忆抹除 (彻底根治反复进出同一个房间)
            # # =================================================================
            # # 1. 抹除视野内已探索的区域
            # mask_explored = np.all(self.agent.explored_map == self.agent.explored_color, axis=-1)
            # self.agent.cvalue_map[mask_explored] = 0.0

            # # 2. 抹除脚下绝对死角 (半径 2.5 米)，防止原地打转
            # agent_coords = self.agent._global_to_grid(obs['agent_state'].position)
            # Y, X = np.ogrid[:self.agent.cvalue_map.shape[0], :self.agent.cvalue_map.shape[1]]
            # dist_from_center = np.sqrt((X - agent_coords[0])**2 + (Y - agent_coords[1])**2)
            # mask_feet = dist_from_center <= int(2.5 * self.agent.scale)
            # self.agent.cvalue_map[mask_feet] = 0.0
            # # =================================================================
            
            direction_image = episode_images[-12:][goal_rotate]
            
            ## ✅ 6. Make Plan - 第756行修改为：
            goal_flag, subtask = self.agent.make_plan(
                direction_image, 
                self.previous_subtask, 
                goal_reason, 
                current_goal, 
                goal_num  # ✅ 使用 goal_num（不是 num_objects）
            )

            # 7. 转向目标
            steps_clockwise = goal_rotate
            steps_counter = 12 - goal_rotate
            for j in range(min(steps_clockwise, steps_counter)):
                if steps_clockwise <= steps_counter:
                    obs = self.simWrapper.step(loop_action_clockwise)
                else:
                    obs = self.simWrapper.step(loop_action_counterclock)
            logging.info(f"Turning to angle {goal_rotate*30} deg (steps={min(steps_clockwise, steps_counter)}), goal_flag={goal_flag}")

            # 8. 绘制地图
            try:
                cvalue_map = self.agent.draw_cvalue_map(obs['agent_state'])
            except TypeError:
                # 如果 draw_cvalue_map 还没更新支持 target_name 参数，就用旧的调用方式
                cvalue_map = self.agent.draw_cvalue_map(obs['agent_state'])

            super()._step_env(obs)

            # 9. 更新 obs
            if is_multi:
                obs['goals'] = goals_list
                obs['found_objects'] = found_list
                obs['num_objects'] = num_objects
                obs['current_target'] = current_goal
                # ✅✅✅ 必须补上这一行！让 Agent 知道当前专注的目标是什么
                obs['goal'] = current_goal
            else:
                obs['goal'] = current_goal
                # 🔴 核心修复：把单目标的完成状态也传递给 Agent 大脑！
                obs['found_objects'] = found_list

            obs['subtask'] = subtask
            obs['goal_flag'] = goal_flag
            agent_state = obs['agent_state']
            self.agent_distance_traveled += np.linalg.norm(agent_state.position - self.prev_agent_position)
            self.prev_agent_position = agent_state.position
            
            # 10. Step Agent
            agent_action, metadata = self.agent.step(obs, goal_num=goal_num)
            step_metadata = metadata['step_metadata']
            
            metadata['logging_data']['EVALUATOR_RESPONSE'] = str({'goal_rotate':goal_rotate*30, 'explorable_value': explorable_value, 'reason': reason})
            metadata['logging_data']['PLANNING_RESPONSE'] = str({'goal_flag': goal_flag, 'subtask': subtask})
            logging_data = metadata['logging_data']

            images = metadata['images']
            if metadata['step'] is not None:
                step_text = f"step {metadata['step']}"
                color_origin = np.ascontiguousarray(color_origin)
                color_origin = cv2.putText(color_origin, step_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            scale_factor = color_origin.shape[0] / 1080
            padding = 20
            text_size = 2.0 * scale_factor
            text_thickness = 2
            color_origin = np.ascontiguousarray(color_origin)
            if is_multi:
                y_offset = padding
                for i, obj_name in enumerate(goals_list):
                    is_found = found_list[i]
                    status_text = "[DONE]" if is_found else "[  ]"
                    color = (0, 220, 0) if is_found else (0, 0, 255)
                    display_text = f"{obj_name} {status_text}"
                    (text_width, text_height), _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, text_size, text_thickness)
                    text_position = (color_origin.shape[1] - text_width - padding, int(y_offset + text_height))
                    cv2.putText(color_origin, display_text, text_position, cv2.FONT_HERSHEY_SIMPLEX, text_size, color, text_thickness, cv2.LINE_AA)
                    y_offset += text_height + int(15 * scale_factor)
            else:
                display_goal = current_goal
                if display_goal is not None:
                    text_size_single = 2.5 * scale_factor
                    (text_width, text_height), _ = cv2.getTextSize(f"goal:{display_goal}", cv2.FONT_HERSHEY_SIMPLEX, text_size_single, text_thickness)
                    text_position = (color_origin.shape[1] - text_width - padding, padding + text_height)
                    cv2.putText(color_origin, f"goal:{display_goal}", text_position, cv2.FONT_HERSHEY_SIMPLEX, text_size_single, (255, 0, 0), text_thickness, cv2.LINE_AA)

            planner_images = {'panoramic': panoramic_image,
                            'color_origin': color_origin,
                            'nav_map': nav_map,
                            'cvalue_map': cvalue_map}
            images.update(planner_images)

            metrics = self._calculate_metrics(agent_state, agent_action, self.current_episode['shortest_path'], self.cfg['max_steps'])
            step_metadata.update(metrics)

            self._log(images, step_metadata, logging_data)

            if metrics['done']:
                agent_action = None

            return agent_action
    def _post_episode(self):
        """
        Called after the episode is complete, saves the dataframe log, and resets the environment.
        Sends a request to the aggregator server if parallel is set to True.
        """
        self.df.to_pickle(os.path.join(os.environ.get("LOG_DIR"), f'{self.outer_run_name}/{self.inner_run_name}/{self.curr_run_name}/df_results.pkl'))
        self.simWrapper.reset()
        self.agent.reset()
        if self.cfg['parallel']:
            try:
                self.wandb_log_data['spend'] = self.agent.get_spend()
                self.wandb_log_data['default_rate'] = len(self.df[self.df['success'] == 0]) / len(self.df)
                response = requests.post(f'http://localhost:{self.cfg["port"]}/log', json=self.wandb_log_data)
                if response.status_code != 200:
                    logging.error(f"Failed to send metrics: {response.text}")
            except Exception as e:
                tb = traceback.extract_tb(e.__traceback__)
                for frame in tb:
                    logging.error(f"Frame {frame.filename} line {frame.lineno}")
                logging.error(e)

        logging.info(f"Success: {self.wandb_log_data['goal_reached']}")
        logging.info('\n===================RUN COMPLETE===================\n')
        if self.cfg['log_freq'] == 1:
            create_gif(
                os.path.join(os.environ.get("LOG_DIR"), f'{self.outer_run_name}/{self.inner_run_name}/{self.curr_run_name}'), self.agent.cfg['sensor_cfg']['img_height'], self.agent.cfg['sensor_cfg']['img_width'], agent_cls=self.agent_cls
            )
            create_gif_nav(
                    os.path.join(os.environ.get("LOG_DIR"), f'{self.outer_run_name}/{self.inner_run_name}/{self.curr_run_name}'),
                    1800, 1800
            )
            create_gif_cvalue(
                os.path.join(os.environ.get("LOG_DIR"), f'{self.outer_run_name}/{self.inner_run_name}/{self.curr_run_name}'),
                1800, 1800
            )