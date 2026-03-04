import logging
import math
import random
import habitat_sim
import numpy as np
import cv2
import ast
import concurrent.futures

from simWrapper import PolarAction
from utils import *
from api import *

class Agent:
    def __init__(self, cfg: dict):
        pass

    def step(self, obs: dict):
        """Primary agent loop to map observations to the agent's action and returns metadata."""
        raise NotImplementedError

    def get_spend(self):
        """Returns the dollar amount spent by the agent on API calls."""
        return 0

    def reset(self):
        """To be called after each episode."""
        pass

class RandomAgent(Agent):
    """Example implementation of a random agent."""
    
    def step(self, obs):
        rotate = random.uniform(-0.2, 0.2)
        forward = random.uniform(0, 1)

        agent_action = PolarAction(forward, rotate)
        metadata = {
            'step_metadata': {'success': 1}, # indicating the VLM succesfully selected an action
            'logging_data': {}, # to be logged in the txt file
            'images': {'color_sensor': obs['color_sensor']} # to be visualized in the GIF
        }
        return agent_action, metadata

class VLMNavAgent(Agent):
    """
    Primary class for the VLMNav agent. Four primary components: navigability, action proposer, projection, and prompting. Runs seperate threads for stopping and preprocessing. This class steps by taking in an observation and returning a PolarAction, along with metadata for logging and visulization.
    """
    explored_color = GREY
    unexplored_color = GREEN
    map_size = 5000
    explore_threshold = 3
    voxel_ray_size = 60
    e_i_scaling = 0.8

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.fov = cfg['sensor_cfg']['fov']
        self.resolution = (
            cfg['sensor_cfg']['img_height'],
            cfg['sensor_cfg']['img_width']
        )

        self.focal_length = calculate_focal_length(self.fov, self.resolution[1])
        self.scale = cfg['map_scale']
        self._initialize_vlms(cfg['vlm_cfg'])       

        assert cfg['navigability_mode'] in ['none', 'depth_estimate', 'segmentation', 'depth_sensor']
        self.depth_estimator = DepthEstimator() if cfg['navigability_mode'] == 'depth_estimate' else None
        self.segmentor = Segmentor() if cfg['navigability_mode'] == 'segmentation' else None
        self.reset()

    # def step(self, obs: dict, goal_num: dict=None):
    #     agent_state: habitat_sim.AgentState = obs['agent_state']

    #     # 检测是否卡住
    #     current_pos = np.array([agent_state.position[0], agent_state.position[2]])
    #     self.position_history.append(current_pos)
        
    #     if len(self.position_history) > 10:
    #         self.position_history.pop(0)
        
    #     if len(self.position_history) >= 10:
    #         distances = [np.linalg.norm(self.position_history[i] - self.position_history[i-1]) 
    #                     for i in range(1, len(self.position_history))]
    #         avg_movement = np.mean(distances)
            
    #         if avg_movement < 0.5:
    #             self.stuck_counter += 1
    #             if self.stuck_counter >= 3:
    #                 logging.warning(f"⚠️ Agent stuck! Avg movement: {avg_movement:.2f}m")
                    
    #                 self.turned = self.step_ndx - self.cfg['turn_around_cooldown']
                    
    #                 # 🔴 核心修复 2：物理级反卡死（彻底抹除当前区域的好奇心）
    #                 if hasattr(self, 'cvalue_map') and self.cvalue_map is not None:
    #                     agent_coords = self._global_to_grid(agent_state.position)
    #                     x, y = agent_coords
    #                     radius = int(3.0 * self.scale) # 半径设为3米范围
                        
    #                     # 使用 numpy 生成圆圈掩码，将自身周围 3 米范围内的好奇心分数强制归零
    #                     Y, X = np.ogrid[:self.cvalue_map.shape[0], :self.cvalue_map.shape[1]]
    #                     dist_from_center = np.sqrt((X - x)**2 + (Y - y)**2)
    #                     mask = dist_from_center <= radius
                        
    #                     # 强制清零所有目标通道的当前区域分数，强迫大模型走向全新的区域
    #                     self.cvalue_map[mask] = 0.0
    #                     logging.info("🛑 Dead End Registered: Wiped local curiosity map around agent.")
                    
    #                 self.position_history = [current_pos] 
    #                 self.stuck_counter = 0
    #         else:
    #             self.stuck_counter = 0

    #     if self.step_ndx == 0:
    #         self.init_pos = agent_state.position

    #     agent_action, metadata = self._choose_action(obs, goal_num)
    #     metadata['step_metadata'].update(self.cfg)

    #     if metadata['step_metadata']['action_number'] == 0:
    #         self.turned = self.step_ndx

    #     chosen_action_image = obs['color_sensor'].copy()
    #     self._project_onto_image(
    #         metadata['a_final'], chosen_action_image, agent_state,
    #         agent_state.sensor_states['color_sensor'], 
    #         chosen_action=metadata['step_metadata']['action_number'],
    #         step=self.step_ndx,
    #         goal=obs['goal']
    #     )
    #     metadata['images']['color_sensor_chosen'] = chosen_action_image

    #     metadata['images']['voxel_map_chosen'] = self._generate_voxel(
    #         metadata['a_final'],
    #         agent_state=agent_state,
    #         chosen_action=metadata['step_metadata']['action_number'],
    #         step=self.step_ndx
    #     )
    #     self.step_ndx += 1
    #     return agent_action, metadata
    def step(self, obs: dict, goal_num: dict=None):
        agent_state: habitat_sim.AgentState = obs['agent_state']

        if self.step_ndx == 0:
            self.init_pos = agent_state.position

        # --- 1. 正常让大模型做决策 ---
        agent_action, metadata = self._choose_action(obs, goal_num)
        metadata['step_metadata'].update(self.cfg)

        if metadata['step_metadata']['action_number'] == 0:
            self.turned = self.step_ndx

        # --- 2. 🔴 核心反卡死：上帝之手 (God Mode) ---
        current_pos = np.array([agent_state.position[0], agent_state.position[2]])
        self.position_history.append(current_pos)
        
        if len(self.position_history) > 20:
            self.position_history.pop(0)
            
        if len(self.position_history) >= 5:
            # 计算移动距离
            recent_path = self.position_history[-5:]
            distances = [np.linalg.norm(recent_path[i] - recent_path[i-1]) for i in range(1, len(recent_path))]
            avg_movement = np.mean(distances)
            
            # 计算历史重合度
            history_pool = np.array(self.position_history[:-2]) 
            if len(history_pool) > 0:
                dists_to_curr = np.linalg.norm(history_pool - current_pos, axis=1)
                nearby_count = np.sum(dists_to_curr < 1.0)
            else:
                nearby_count = 0
            
            # 🚨 触发条件
            is_stuck = (avg_movement < 0.3) or (nearby_count > 6)
            
            if is_stuck:
                self.stuck_counter += 1
                if self.stuck_counter >= 2: 
                    logging.warning(f"⚠️ Agent Stuck/Looping! Move: {avg_movement:.2f}m, Revisits: {nearby_count}")
                    logging.warning("🚀 GOD MODE: Hijacking VLM and clearing local memory!")
                    
                    self.stuck_counter = 0
                    self.position_history = [] 
                    
                    # 🔴 核心修复 1：物理拉黑该区域，防止“橡皮筋效应”弹回来
                    if hasattr(self, 'cvalue_map') and self.cvalue_map is not None:
                        agent_coords = self._global_to_grid(agent_state.position)
                        x, y = agent_coords
                        radius = int(3.0 * self.scale) # 拉黑半径 3 米
                        Y, X = np.ogrid[:self.cvalue_map.shape[0], :self.cvalue_map.shape[1]]
                        dist_from_center = np.sqrt((X - x)**2 + (Y - y)**2)
                        mask = dist_from_center <= radius
                        self.cvalue_map[mask] = 0.0 # 好奇心全部清零
                        logging.info("🛑 Wiped local curiosity map to prevent snapping back.")

                    # 🔴 核心修复 2：强行选一条尽量远的路逃跑
                    a_final = metadata['a_final']
                    # 优先选择距离大于 1.0 米的路线（直接跨出房间门）
                    valid_indices = [i + 1 for i, (r, theta) in enumerate(a_final) if r > 1.0]
                    if not valid_indices:
                         valid_indices = [i + 1 for i, (r, theta) in enumerate(a_final) if r > 0.5]
                    if not valid_indices:
                         valid_indices = list(range(1, len(a_final) + 1))
                    
                    if valid_indices:
                        forced_action_number = random.choice(valid_indices)
                        metadata['step_metadata']['action_number'] = forced_action_number
                        agent_action = self._action_number_to_polar(forced_action_number, list(a_final))
                        self.turned = self.step_ndx - self.cfg['turn_around_cooldown']
            else:
                self.stuck_counter = max(0, self.stuck_counter - 0.5)

        # --- 3. 画面渲染 (保持不变) ---
        chosen_action_image = obs['color_sensor'].copy()
        self._project_onto_image(
            metadata['a_final'], chosen_action_image, agent_state,
            agent_state.sensor_states['color_sensor'], 
            chosen_action=metadata['step_metadata']['action_number'],
            step=self.step_ndx,
            goal=obs['goal']
        )
        metadata['images']['color_sensor_chosen'] = chosen_action_image

        metadata['images']['voxel_map_chosen'] = self._generate_voxel(
            metadata['a_final'],
            agent_state=agent_state,
            chosen_action=metadata['step_metadata']['action_number'],
            step=self.step_ndx
        )
        self.step_ndx += 1
        return agent_action, metadata
    def get_spend(self):
        return self.actionVLM.get_spend() + self.stoppingVLM.get_spend()

    def reset(self):
        self.voxel_map = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        self.explored_map = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        self.stopping_calls = [-2]
        self.step_ndx = 0
        self.init_pos = None
        self.turned = -self.cfg['turn_around_cooldown']
        # 第 114 行后添加
        self.position_history = []  # 记录最近的位置
        self.stuck_counter = 0      # 卡住计数器
        self.actionVLM.reset()

    def _construct_prompt(self, **kwargs):
        raise NotImplementedError
    
    def _choose_action(self, obs):
        raise NotImplementedError

    def _initialize_vlms(self, cfg: dict):
        vlm_cls = globals()[cfg['model_cls']]
        system_instruction = (
            "You are an embodied robotic assistant, with an RGB image sensor. You observe the image and instructions "
            "given to you and output a textual response, which is converted into actions that physically move you "
            "within the environment. You cannot move through closed doors. "
        )
        self.actionVLM: VLM = vlm_cls(**cfg['model_kwargs'], system_instruction=system_instruction)
        self.stoppingVLM: VLM = vlm_cls(**cfg['model_kwargs'])

    def _run_threads(self, obs: dict, stopping_images: list[np.array], goal):
        """Concurrently runs the stopping thread to determine if the agent should stop, and the preprocessing thread to calculate potential actions."""
        with concurrent.futures.ThreadPoolExecutor() as executor:
            preprocessing_thread = executor.submit(self._preprocessing_module, obs)
            stopping_thread = executor.submit(self._stopping_module, stopping_images, goal)

            a_final, images = preprocessing_thread.result()
            called_stop, stopping_response = stopping_thread.result()
        
        if called_stop:
            logging.info('Model called stop')
            self.stopping_calls.append(self.step_ndx)
            # If the model calls stop, turn off navigability and explore bias tricks
            if self.cfg['navigability_mode'] != 'none':
                new_image = obs['color_sensor'].copy()
                a_final = self._project_onto_image(
                    self._get_default_arrows(), new_image, obs['agent_state'],
                    obs['agent_state'].sensor_states['color_sensor'],
                    step=self.step_ndx,
                    goal=obs['goal']
                )
                images['color_sensor'] = new_image

        step_metadata = {
            'action_number': -10,
            'success': 1,
            'model': self.actionVLM.name,
            'agent_location': obs['agent_state'].position,
            'called_stopping': called_stop
        }
        return a_final, images, step_metadata, stopping_response

    def _preprocessing_module(self, obs: dict):
        """Excutes the navigability, action_proposer and projection submodules."""
        agent_state = obs['agent_state']
        images = {'color_sensor': obs['color_sensor'].copy()}

        if self.cfg['navigability_mode'] == 'none':
            a_final = [
                # Actions for the w/o nav baseline
                (self.cfg['max_action_dist'], -0.36 * np.pi),
                (self.cfg['max_action_dist'], -0.28 * np.pi),
                (self.cfg['max_action_dist'], 0),
                (self.cfg['max_action_dist'], 0.28 * np.pi),
                (self.cfg['max_action_dist'], 0.36 * np.pi)
            ]
        else:
            a_initial = self._navigability(obs)
            a_final = self._action_proposer(a_initial, agent_state)


        a_final_projected = self._projection(a_final, images, agent_state, obs['goal'])
        images['voxel_map'] = self._generate_voxel(a_final_projected, agent_state=agent_state, step=self.step_ndx)
        return a_final_projected, images

    def _stopping_module(self, stopping_images: list[np.array], goal):
        """Determines if the agent should stop."""
        stopping_prompt = self._construct_prompt(goal, 'stopping')
        stopping_response = self.stoppingVLM.call(stopping_images, stopping_prompt)
        dct = self._eval_response(stopping_response)
        if 'done' in dct and int(dct['done']) == 1:
            return True, stopping_response
        
        return False, stopping_response

    def _navigability(self, obs: dict):
        """Generates the set of navigability actions and updates the voxel map accordingly."""
        agent_state: habitat_sim.AgentState = obs['agent_state']
        sensor_state = agent_state.sensor_states['color_sensor']
        rgb_image = obs['color_sensor']
        depth_image = obs[f'depth_sensor']
        if self.cfg['navigability_mode'] == 'depth_estimate':
            depth_image = self.depth_estimator.call(rgb_image)
        if self.cfg['navigability_mode'] == 'segmentation':
            depth_image = None

        navigability_mask = self._get_navigability_mask(
            rgb_image, depth_image, agent_state, sensor_state
        )

        sensor_range =  np.deg2rad(self.fov / 2) * 1.5

        all_thetas = np.linspace(-sensor_range, sensor_range, self.cfg['num_theta'])
        start = agent_frame_to_image_coords(
            [0, 0, 0], agent_state, sensor_state,
            resolution=self.resolution, focal_length=self.focal_length
        )  # agent的原点在图像坐标系的位置（像素）

        a_initial = []
        for theta_i in all_thetas:
            r_i, theta_i = self._get_radial_distance(start, theta_i, navigability_mask, agent_state, sensor_state, depth_image)
            if r_i is not None:
                self._update_voxel(
                    r_i, theta_i, agent_state,
                    clip_dist=self.cfg['max_action_dist'], clip_frac=self.e_i_scaling
                )
                a_initial.append((r_i, theta_i))

        return a_initial

    def _action_proposer(self, a_initial: list, agent_state: habitat_sim.AgentState):
        """Refines the initial set of actions, ensuring spacing and adding a bias towards exploration."""
        min_angle = self.fov/self.cfg['spacing_ratio']
        explore_bias = self.cfg['explore_bias']
        clip_frac = self.cfg['clip_frac']
        clip_mag = self.cfg['max_action_dist']

        explore = explore_bias > 0
        unique = {}
        for mag, theta in a_initial:
            if theta in unique:
                unique[theta].append(mag)
            else:
                unique[theta] = [mag]
        arrowData = []

        topdown_map = self.voxel_map.copy()
        mask = np.all(self.explored_map == self.explored_color, axis=-1)
        topdown_map[mask] = self.explored_color
        for theta, mags in unique.items():
            # Reference the map to classify actions as explored or unexplored
            mag = min(mags)
            cart = [self.e_i_scaling*mag*np.sin(theta), 0, -self.e_i_scaling*mag*np.cos(theta)]
            global_coords = local_to_global(agent_state.position, agent_state.rotation, cart)
            grid_coords = self._global_to_grid(global_coords)
            score = (sum(np.all((topdown_map[grid_coords[1]-2:grid_coords[1]+2, grid_coords[0]] == self.explored_color), axis=-1)) + 
                    sum(np.all(topdown_map[grid_coords[1], grid_coords[0]-2:grid_coords[0]+2] == self.explored_color, axis=-1)))
            arrowData.append([clip_frac*mag, theta, score<3])

        arrowData.sort(key=lambda x: x[1])
        thetas = set()
        out = []
        filter_thresh = 0.75
        filtered = list(filter(lambda x: x[0] > filter_thresh, arrowData))

        filtered.sort(key=lambda x: x[1])
        if filtered == []:
            return []
        if explore:
            # Add unexplored actions with spacing, starting with the longest one
            f = list(filter(lambda x: x[2], filtered))
            if len(f) > 0:
                longest = max(f, key=lambda x: x[0])
                longest_theta = longest[1]
                smallest_theta = longest[1]
                longest_ndx = f.index(longest)
            
                out.append([min(longest[0], clip_mag), longest[1], longest[2]])
                thetas.add(longest[1])
                for i in range(longest_ndx+1, len(f)):
                    if f[i][1] - longest_theta > (min_angle*0.9):
                        out.append([min(f[i][0], clip_mag), f[i][1], f[i][2]])
                        thetas.add(f[i][1])
                        longest_theta = f[i][1]
                for i in range(longest_ndx-1, -1, -1):
                    if smallest_theta - f[i][1] > (min_angle*0.9):
                        
                        out.append([min(f[i][0], clip_mag), f[i][1], f[i][2]])
                        thetas.add(f[i][1])
                        smallest_theta = f[i][1]

                for r_i, theta_i, e_i in filtered:
                    if theta_i not in thetas and min([abs(theta_i - t) for t in thetas]) > min_angle*explore_bias:
                        out.append((min(r_i, clip_mag), theta_i, e_i))
                        thetas.add(theta)

        if len(out) == 0:
            # if no explored actions or no explore bias
            longest = max(filtered, key=lambda x: x[0])
            longest_theta = longest[1]
            smallest_theta = longest[1]
            longest_ndx = filtered.index(longest)
            out.append([min(longest[0], clip_mag), longest[1], longest[2]])
            
            for i in range(longest_ndx+1, len(filtered)):
                if filtered[i][1] - longest_theta > min_angle:
                    out.append([min(filtered[i][0], clip_mag), filtered[i][1], filtered[i][2]])
                    longest_theta = filtered[i][1]
            for i in range(longest_ndx-1, -1, -1):
                if smallest_theta - filtered[i][1] > min_angle:
                    out.append([min(filtered[i][0], clip_mag), filtered[i][1], filtered[i][2]])
                    smallest_theta = filtered[i][1]


        if (out == [] or max(out, key=lambda x: x[0])[0] < self.cfg['min_action_dist']) and (self.step_ndx - self.turned) < self.cfg['turn_around_cooldown']:
            return self._get_default_arrows()
        
        out.sort(key=lambda x: x[1])
        return [(mag, theta) for mag, theta, _ in out]

    def _projection(self, a_final: list, images: dict, agent_state: habitat_sim.AgentState, goal: str, candidate_flag: bool=False):
        """
        Projection component of VLMnav. Projects the arrows onto the image, annotating them with action numbers.
        Note actions that are too close together or too close to the boundaries of the image will not get projected.
        """
        a_final_projected = self._project_onto_image(
            a_final, images['color_sensor'], agent_state,
            agent_state.sensor_states['color_sensor'],
            step=self.step_ndx,
            goal=goal,
            candidate_flag=candidate_flag
        )

        if not a_final_projected and (self.step_ndx - self.turned < self.cfg['turn_around_cooldown']) and not candidate_flag:
            logging.info('No actions projected and cannot turn around')
            a_final = self._get_default_arrows()
            a_final_projected = self._project_onto_image(
                a_final, images['color_sensor'], agent_state,
                agent_state.sensor_states['color_sensor'],
                step=self.step_ndx,
                goal=goal
            )

        return a_final_projected

    def _prompting(self, goal, a_final: list, images: dict, step_metadata: dict, subtask: str, goal_num: int):
        """添加 goal_num 参数"""
        prompt_type = 'action'
        action_prompt = self._construct_prompt(goal, prompt_type, subtask, num_actions=len(a_final), goal_num=goal_num)

        prompt_images = [images['color_sensor']]
        if 'goal_image' in images:
            prompt_images.append(images['goal_image'])

        response = self.ActionVLM.call_chat(prompt_images, action_prompt)

        logging_data = {}
        try:
            response_dict = self._eval_response(response)
            step_metadata['action_number'] = int(response_dict['action'])
        except (IndexError, KeyError, TypeError, ValueError) as e:
            logging.error(f'Error parsing response {e}')
            step_metadata['success'] = 0
        finally:
            logging_data['ACTION_NUMBER'] = step_metadata.get('action_number')
            logging_data['ACTION_PROMPT'] = action_prompt
            logging_data['ACTION_RESPONSE'] = response

        return step_metadata, logging_data, response

    def _get_navigability_mask(self, rgb_image: np.array, depth_image: np.array, agent_state: habitat_sim.AgentState, sensor_state: habitat_sim.SixDOFPose):
        """
        Get the navigability mask for the current state, according to the configured navigability mode.
        """
        if self.cfg['navigability_mode'] == 'segmentation':
            navigability_mask = self.segmentor.get_navigability_mask(rgb_image)
        else:
            thresh = 1 if self.cfg['navigability_mode'] == 'depth_estimate' else self.cfg['navigability_height_threshold']
            height_map = depth_to_height(depth_image, self.fov, sensor_state.position, sensor_state.rotation)
            navigability_mask = abs(height_map - (agent_state.position[1] - 0.04)) < thresh

        return navigability_mask

    def _get_default_arrows(self):
        """
        Get the action options for when the agent calls stop the first time, or when no navigable actions are found.
        """
        angle = np.deg2rad(self.fov / 2) * 0.7
        
        default_actions = [
            (self.cfg['stopping_action_dist'], -angle),
            (self.cfg['stopping_action_dist'], -angle / 4),
            (self.cfg['stopping_action_dist'], angle / 4),
            (self.cfg['stopping_action_dist'], angle)
        ]
        
        default_actions.sort(key=lambda x: x[1])
        return default_actions

    def _get_radial_distance(self, start_pxl: tuple, theta_i: float, navigability_mask: np.ndarray, 
                             agent_state: habitat_sim.AgentState, sensor_state: habitat_sim.SixDOFPose, 
                             depth_image: np.ndarray):
        """
        Calculates the distance r_i that the agent can move in the direction theta_i, according to the navigability mask.
        """
        agent_point = [2 * np.sin(theta_i), 0, -2 * np.cos(theta_i)]
        end_pxl = agent_frame_to_image_coords(
            agent_point, agent_state, sensor_state, 
            resolution=self.resolution, focal_length=self.focal_length
        )
        if end_pxl is None or end_pxl[1] >= self.resolution[0]:
            return None, None

        H, W = navigability_mask.shape

        # Find intersections of the theoretical line with the image boundaries
        intersections = find_intersections(start_pxl[0], start_pxl[1], end_pxl[0], end_pxl[1], W, H)
        if intersections is None:
            return None, None

        (x1, y1), (x2, y2) = intersections
        num_points = max(abs(x2 - x1), abs(y2 - y1)) + 1
        if num_points < 5:
            return None, None
        x_coords = np.linspace(x1, x2, num_points)
        y_coords = np.linspace(y1, y2, num_points)

        out = (int(x_coords[-1]), int(y_coords[-1]))
        if not navigability_mask[int(y_coords[0]), int(x_coords[0])]:
            return 0, theta_i

        for i in range(num_points - 4):
            # Trace pixels until they are not navigable
            y = int(y_coords[i])
            x = int(x_coords[i])
            if sum([navigability_mask[int(y_coords[j]), int(x_coords[j])] for j in range(i, i + 4)]) <= 2:
                out = (x, y)
                break

        if i < 5:
            return 0, theta_i

        if self.cfg['navigability_mode'] == 'segmentation':
            #Simple estimation of distance based on number of pixels
            r_i = 0.0794 * np.exp(0.006590 * i) + 0.616

        else:
            #use depth to get distance
            out = (np.clip(out[0], 0, W - 1), np.clip(out[1], 0, H - 1))
            camera_coords = unproject_2d(
                *out, depth_image[out[1], out[0]], resolution=self.resolution, focal_length=self.focal_length
            )
            local_coords = global_to_local(
                agent_state.position, agent_state.rotation,
                local_to_global(sensor_state.position, sensor_state.rotation, camera_coords)
            )
            r_i = np.linalg.norm([local_coords[0], local_coords[2]])

        return r_i, theta_i

    def _can_project(self, r_i: float, theta_i: float, agent_state: habitat_sim.AgentState, sensor_state: habitat_sim.SixDOFPose):
        """
        Checks whether the specified polar action can be projected onto the image, i.e., not too close to the boundaries of the image.
        """
        agent_point = [r_i * np.sin(theta_i), 0, -r_i * np.cos(theta_i)]
        end_px = agent_frame_to_image_coords(
            agent_point, agent_state, sensor_state, 
            resolution=self.resolution, focal_length=self.focal_length
        )
        if end_px is None:
            return None

        if (
            self.cfg['image_edge_threshold'] * self.resolution[1] <= end_px[0] <= (1 - self.cfg['image_edge_threshold']) * self.resolution[1] and
            self.cfg['image_edge_threshold'] * self.resolution[0] <= end_px[1] <= (1 - self.cfg['image_edge_threshold']) * self.resolution[0]
        ):
            return end_px
        return None

    def _project_onto_image(self, a_final: list, rgb_image: np.ndarray, agent_state: habitat_sim.AgentState, sensor_state: habitat_sim.SixDOFPose, chosen_action: int=None, step: int=None, goal: str='', candidate_flag: bool=False):
        """
        Projects a set of actions onto a single image. Keeps track of action-to-number mapping.
        """
        # 🔴 核心修复 1：强制放大 3 倍比例尺。这样即使图片被压缩到 512，模型依然能清晰读出数字
        scale_factor = (rgb_image.shape[0] / 1080) * 1.0 
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_color = BLACK
        circle_color = WHITE
        projected = {}
        if chosen_action == -1:
            put_text_on_image(
                rgb_image, 'TERMINATING EPISODE', text_color=GREEN, text_size=4 * scale_factor,
                location='center', text_thickness=math.ceil(3 * scale_factor), highlight=False
            )
            return projected

        start_px = agent_frame_to_image_coords(
            [0, 0, 0], agent_state, sensor_state, 
            resolution=self.resolution, focal_length=self.focal_length
        )
        for _, (r_i, theta_i) in enumerate(a_final):
            text_size = 2.4 * scale_factor
            text_thickness = math.ceil(3 * scale_factor)

            end_px = self._can_project(r_i, theta_i, agent_state, sensor_state)
            if end_px is not None:
                action_name = len(projected) + 1
                projected[(r_i, theta_i)] = action_name

                cv2.arrowedLine(rgb_image, tuple(start_px), tuple(end_px), RED, math.ceil(5 * scale_factor), tipLength=0.0)
                text = str(action_name)
                (text_width, text_height), _ = cv2.getTextSize(text, font, text_size, text_thickness)
                circle_center = (end_px[0], end_px[1])
                circle_radius = max(text_width, text_height) // 2 + math.ceil(15 * scale_factor)

                if chosen_action is not None and action_name == chosen_action:
                    cv2.circle(rgb_image, circle_center, circle_radius, GREEN, -1)
                else:
                    cv2.circle(rgb_image, circle_center, circle_radius, circle_color, -1)
                cv2.circle(rgb_image, circle_center, circle_radius, RED, math.ceil(2 * scale_factor))
                text_position = (circle_center[0] - text_width // 2, circle_center[1] + text_height // 2)
                cv2.putText(rgb_image, text, text_position, font, text_size, text_color, text_thickness)

        if not candidate_flag and ((self.step_ndx - self.turned) >= self.cfg['turn_around_cooldown'] or self.step_ndx == self.turned or (chosen_action == 0)):
            text = '0'
            text_size = 3.1 * scale_factor
            text_thickness = math.ceil(3 * scale_factor)
            (text_width, text_height), _ = cv2.getTextSize(text, font, text_size, text_thickness)
            circle_center = (math.ceil(0.1 * rgb_image.shape[1]), math.ceil(rgb_image.shape[0] / 2)) # 稍微往右移一点防止越界
            circle_radius = max(text_width, text_height) // 2 + math.ceil(15 * scale_factor)

            if chosen_action is not None and chosen_action == 0:
                cv2.circle(rgb_image, circle_center, circle_radius, GREEN, -1)
            else:
                cv2.circle(rgb_image, circle_center, circle_radius, circle_color, -1)
            cv2.circle(rgb_image, circle_center, circle_radius, RED, math.ceil(2 * scale_factor))
            text_position = (circle_center[0] - text_width // 2, circle_center[1] + text_height // 2)
            cv2.putText(rgb_image, text, text_position, font, text_size, text_color, text_thickness)
            cv2.putText(rgb_image, 'TURN', (text_position[0] - 10, text_position[1] + math.ceil(60 * scale_factor)), font, text_size * 0.75, RED, text_thickness)

        if step is not None:
            step_text = f'step {step}'
            cv2.putText(rgb_image, step_text, (10, 50), font, 1.5, (255, 255, 255), 2, cv2.LINE_AA)

        if goal is not None:
            padding = 20
            text_size = 2.5 * scale_factor
            text_thickness = 2
            (text_width, text_height), _ = cv2.getTextSize(f"goal:{goal}", font, text_size, text_thickness)
            text_position = (rgb_image.shape[1] - text_width - padding, padding + text_height)
            cv2.putText(rgb_image, f"goal:{goal}", text_position, font, text_size, (255, 0, 0), text_thickness, cv2.LINE_AA)

        return projected

    def _update_voxel(self, r: float, theta: float, agent_state: habitat_sim.AgentState, clip_dist: float, clip_frac: float):
        """Update the voxel map to mark actions as explored or unexplored"""
        agent_coords = self._global_to_grid(agent_state.position)

        # Mark unexplored regions
        unclipped = max(r - 0.5, 0)
        local_coords = np.array([unclipped * np.sin(theta), 0, -unclipped * np.cos(theta)])
        global_coords = local_to_global(agent_state.position, agent_state.rotation, local_coords)
        point = self._global_to_grid(global_coords)
        cv2.line(self.voxel_map, agent_coords, point, self.unexplored_color, self.voxel_ray_size)

        # Mark explored regions
        clipped = min(clip_frac * r, clip_dist)
        local_coords = np.array([clipped * np.sin(theta), 0, -clipped * np.cos(theta)])
        global_coords = local_to_global(agent_state.position, agent_state.rotation, local_coords)
        point = self._global_to_grid(global_coords)
        cv2.line(self.explored_map, agent_coords, point, self.explored_color, self.voxel_ray_size)

    def _global_to_grid(self, position: np.ndarray, rotation=None):
        """Convert global coordinates to grid coordinates in the agent's voxel map"""
        dx = position[0] - self.init_pos[0]
        dz = position[2] - self.init_pos[2]
        resolution = self.voxel_map.shape
        x = int(resolution[1] // 2 + dx * self.scale)
        y = int(resolution[0] // 2 + dz * self.scale)

        if rotation is not None:
            original_coords = np.array([x, y, 1])
            new_coords = np.dot(rotation, original_coords)
            new_x, new_y = new_coords[0], new_coords[1]
            return (int(new_x), int(new_y))

        return (x, y)

    def _generate_voxel(self, a_final: dict, zoom: int=9, agent_state: habitat_sim.AgentState=None, chosen_action: int=None, step: int=None):
        """For visualization purposes, add the agent's position and actions onto the voxel map"""
        agent_coords = self._global_to_grid(agent_state.position)
        right = (agent_state.position[0] + zoom, 0, agent_state.position[2])
        right_coords = self._global_to_grid(right)
        delta = abs(agent_coords[0] - right_coords[0])

        topdown_map = self.voxel_map.copy()
        mask = np.all(self.explored_map == self.explored_color, axis=-1)
        topdown_map[mask] = self.explored_color

        text_size = 1.25
        text_thickness = 1
        rotation_matrix = None
        agent_coords = self._global_to_grid(agent_state.position, rotation=rotation_matrix)
        x, y = agent_coords
        font = cv2.FONT_HERSHEY_SIMPLEX

        if self.step_ndx - self.turned >= self.cfg['turn_around_cooldown']:
            a_final[(0.75, np.pi)] = 0

        for (r, theta), action in a_final.items():
            local_pt = np.array([r * np.sin(theta), 0, -r * np.cos(theta)])
            global_pt = local_to_global(agent_state.position, agent_state.rotation, local_pt)
            act_coords = self._global_to_grid(global_pt, rotation=rotation_matrix)

            # Draw action arrows and labels
            cv2.arrowedLine(topdown_map, tuple(agent_coords), tuple(act_coords), RED, 5, tipLength=0.05)
            text = str(action)
            (text_width, text_height), _ = cv2.getTextSize(text, font, text_size, text_thickness)
            circle_center = (act_coords[0], act_coords[1])
            circle_radius = max(text_width, text_height) // 2 + 15

            if chosen_action is not None and action == chosen_action:
                cv2.circle(topdown_map, circle_center, circle_radius, GREEN, -1)
            else:
                cv2.circle(topdown_map, circle_center, circle_radius, WHITE, -1)

            text_position = (circle_center[0] - text_width // 2, circle_center[1] + text_height // 2)
            cv2.circle(topdown_map, circle_center, circle_radius, RED, 1)
            cv2.putText(topdown_map, text, text_position, font, text_size, RED, text_thickness + 1)

        # Draw agent's current position
        cv2.circle(topdown_map, agent_coords, radius=15, color=RED, thickness=-1)


        # Zoom the map
        max_x, max_y = topdown_map.shape[1], topdown_map.shape[0]
        x1 = max(0, x - delta)
        x2 = min(max_x, x + delta)
        y1 = max(0, y - delta)
        y2 = min(max_y, y + delta)

        zoomed_map = topdown_map[y1:y2, x1:x2]

        if step is not None:
            step_text = f'step {step}'
            cv2.putText(zoomed_map, step_text, (30, 90), font, 3, (255, 255, 255), 2, cv2.LINE_AA)

        return zoomed_map

    def _action_number_to_polar(self, action_number: int, a_final: list):
        """Converts the chosen action number to its PolarAction instance"""
        try:
            action_number = int(action_number)
            if action_number <= len(a_final) and action_number > 0:
                r, theta = a_final[action_number - 1]
                return PolarAction(r, -theta)
            if action_number == 0:
                return PolarAction(0, np.pi)
        except ValueError:
            pass

        logging.info("Bad action number: " + str(action_number))
        return PolarAction.default

    # def _eval_response(self, response: str):
    #     """Converts the VLM response string into a dictionary, if possible"""
    #     try:
    #         eval_resp = ast.literal_eval(response[response.rindex('{'):response.rindex('}') + 1])
    #         if isinstance(eval_resp, dict):
    #             return eval_resp
    #         else:
    #             raise ValueError
    #     except (ValueError, SyntaxError):
    #         logging.error(f'Error parsing response {response}')
    #         return {}
    def _eval_response(self, response: str):
        """Converts the VLM response string into a dictionary, if possible"""
        import re
        result = re.sub(r"(?<=[a-zA-Z])'(?=[a-zA-Z])", "\\'", response)
        try:
            eval_resp = ast.literal_eval(result[result.index('{') + 1:result.rindex('}')]) # {{}}
            if isinstance(eval_resp, dict):
                return eval_resp
        except:
            try:
                eval_resp = ast.literal_eval(result[result.rindex('{'):result.rindex('}') + 1]) # {}
                if isinstance(eval_resp, dict):
                    return eval_resp
            except:
                try:
                    eval_resp = ast.literal_eval(result[result.index('{'):result.rindex('}')+1]) # {{}, {}}
                    if isinstance(eval_resp, dict):
                        return eval_resp
                except:
                    logging.error(f'Error parsing response {response}')
                    return {}

class WMNavAgent(VLMNavAgent):
    # 🟢 修改后 (正确做法：设为 None，延迟初始化)
    def reset(self):
        self.voxel_map = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        self.explored_map = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        
        # 修改：设为 None。后续代码会在 _update_curiosity_value 中根据目标数量动态初始化
        self.cvalue_map = None 
        
        self.goal_position = []
        self.goal_mask = None
        self.panoramic_mask = {}
        self.effective_mask = {}
        self.stopping_calls = [-2]
        self.step_ndx = 0
        self.init_pos = None
        self.turned = -self.cfg['turn_around_cooldown']
           # ✅ 添加这两行
        self.position_history = []  # 位置历史
        self.stuck_counter = 0      # 卡住计数器
        # 添加多目标跟踪
        self.found_goal_positions = []
       
        self.ActionVLM.reset()
        self.PlanVLM.reset()
        self.PredictVLM.reset()
        self.GoalVLM.reset()

    def _initialize_curiosity_map(self, num_goals):
        """
        ✅ 新增：动态初始化多通道 Curiosity Map
        
        Args:
            num_goals: 目标数量
        """
        if self.cvalue_map is None:
            self.num_goals = num_goals
            # 多通道：(H, W, num_goals)
            self.cvalue_map = 10 * np.ones(
                (self.map_size, self.map_size, num_goals), 
                dtype=np.float16
            )
            logging.info(f"Initialized Curiosity Map with {num_goals} channels")

    # def _initialize_vlms(self, cfg: dict):
    #         import api
    #         vlm_cls = globals()[cfg['model_cls']]
            
    #         action_system_instruction = (
    #             "You are an embodied robotic assistant, with an RGB image sensor. You observe the image and instructions "
    #             "given to you and output a textual response, which is converted into actions that physically move you "
    #             "within the environment. You cannot move through closed doors. "
    #         )
            
    #         # 🔴 温和版打分人设，防止触发 RLHF 幻觉
    #         predict_system_instruction = (
    #             "You are a spatial scoring AI. You evaluate panoramic images and output a JSON dictionary of scores. No other text."
    #         )

    #         self.ActionVLM: VLM = vlm_cls(**cfg['model_kwargs'], system_instruction=action_system_instruction)
    #         self.PlanVLM: VLM = vlm_cls(**cfg['model_kwargs'], system_instruction=action_system_instruction)
    #         self.GoalVLM: VLM = vlm_cls(**cfg['model_kwargs'], system_instruction=action_system_instruction)
            
    #         self.PredictVLM: VLM = vlm_cls(**cfg['model_kwargs'], system_instruction=predict_system_instruction)
    def _initialize_vlms(self, cfg: dict):
            import api  # 确保导入了 api.py
            
            action_system_instruction = (
                "You are an embodied robotic assistant, with an RGB image sensor. You observe the image and instructions "
                "given to you and output a textual response, which is converted into actions that physically move you "
                "within the environment. You cannot move through closed doors. "
            )
            
            predict_system_instruction = (
                "You are a spatial scoring AI. You evaluate panoramic images and output a JSON dictionary of scores. No other text."
            )

            # ==========================================
            # 🧠 混合架构 (Cloud-Edge Hybrid) 部署
            # ==========================================
            
            # 1. 局部小脑 (动作、规划、目标定位)：交给本地免费的 Qwen-3B
            # 速度极快，没有网络延迟，负责高频的简单决策
            # 改为本地实际运行的 3B 模型
            self.ActionVLM = api.QwenVLM(model="Qwen2.5-VL-3B-Instruct", system_instruction=action_system_instruction)
            self.PlanVLM = api.QwenVLM(model="Qwen2.5-VL-3B-Instruct", system_instruction=action_system_instruction)
            self.GoalVLM = api.QwenVLM(model="Qwen2.5-VL-3B-Instruct", system_instruction=action_system_instruction)
            # 2. 全局大脑 (全景打分)：交给云端的 GPT-4o
            # 负责极具挑战的 6宫格拼接图空间推理，彻底解决算错分和崩溃问题
            self.PredictVLM = api.GeminiVLM(model="gpt-4o", system_instruction=predict_system_instruction)

    def _update_panoramic_voxel(self, r: float, theta: float, agent_state: habitat_sim.AgentState, clip_dist: float, clip_frac: float):
        """Update the voxel map to mark actions as explored"""
        agent_coords = self._global_to_grid(agent_state.position)

        # Mark circle explored regions
        clipped = min(clip_frac * r, clip_dist)
        local_coords = np.array([clipped * np.sin(theta), 0, -clipped * np.cos(theta)])
        global_coords = local_to_global(agent_state.position, agent_state.rotation, local_coords)
        point = self._global_to_grid(global_coords)

        radius = int(np.linalg.norm(np.array(agent_coords) - np.array(point)))
        cv2.circle(self.explored_map, agent_coords, radius, self.explored_color, -1)  # -1表示实心圆

    # def _stopping_module(self, obs, goal_num, threshold_dist=0.8):
    #     """检查是否找到所有目标"""
    #     if not self.goal_position:
    #         return False
        
    #     # ✅ 解析 goal_num：兼容字典和整数
    #     if isinstance(goal_num, dict):
    #         total_goals = goal_num.get('total', 1)
    #     else:
    #         total_goals = goal_num if goal_num else 1
        
    #     # 计算当前目标的平均位置
    #     arr = np.array(self.goal_position)
    #     avg_goal_position = np.mean(arr, axis=0)
    #     agent_state = obs['agent_state']
    #     current_position = np.array([agent_state.position[0], agent_state.position[2]])
    #     goal_position_2d = np.array([avg_goal_position[0], avg_goal_position[2]])
    #     dist = np.linalg.norm(current_position - goal_position_2d)

    #     if dist < threshold_dist:
    #         # 检查是否是新目标（避免重复计数）
    #         is_new = True
    #         for found_pos in self.found_goal_positions:
    #             found_pos_2d = np.array([found_pos[0], found_pos[2]])
    #             if np.linalg.norm(goal_position_2d - found_pos_2d) < threshold_dist:
    #                 is_new = False
    #                 break
            
    #         if is_new:
    #             # 记录新找到的目标
    #             self.found_goal_positions.append(avg_goal_position)
    #             logging.info(f"Found goal {len(self.found_goal_positions)}/{total_goals}")
                
    #             # 清空当前目标信息，准备寻找下一个
    #             self.goal_position = []
    #             self.goal_mask = None

    #             # ✅ 重置好奇心地图（让 agent 重新探索）
    #             if hasattr(self, 'cvalue_map') and self.cvalue_map is not None:
    #                 # 只重置未找到目标的通道
    #                 for i in range(len(self.found_goal_positions), self.cvalue_map.shape[2]):
    #                     self.cvalue_map[:, :, i] = 10  # 重置为最大好奇心值
    #                 logging.info(f"🔄 Reset curiosity map for remaining targets")
                
    #             # ✅ 重置 explored_map（让 agent 可以重新访问已探索区域）
    #             if hasattr(self, 'explored_map'):
    #                 self.explored_map.fill(0)
    #                 logging.info(f"🔄 Reset explored map")
            
    #         # 检查是否找到所有目标
    #         if len(self.found_goal_positions) >= total_goals:
    #             logging.info(f"All {total_goals} goals found!")
    #             return True
        
    #     return False

    def _stopping_module(self, obs, goal_num, threshold_dist=0.8):
            """
            🔴 彻底切除 Agent 自身的假进度统计
            完全信任 Env 传来的真实探索进度 (obs['found_objects'])
            """
            found_list = obs.get('found_objects', [])
            
            if not found_list:
                return False
                
            # 只有当 Env 确认所有目标都找到时，Agent 才真正停止
            if all(found_list):
                logging.info("🎯 Agent confirmed all goals found based on Ground Truth!")
                return True
            
            return False
    def _run_threads(self, obs: dict, stopping_images: list[np.array], goal, goal_num):
        """添加 goal_num 参数"""
        called_stop = False
        with concurrent.futures.ThreadPoolExecutor() as executor:
            preprocessing_thread = executor.submit(self._preprocessing_module, obs)
            stopping_thread = executor.submit(self._stopping_module, obs, goal_num)

            a_final, images, a_goal, candidate_images = preprocessing_thread.result()
            called_stop = stopping_thread.result()

        if called_stop:
            logging.info('Model called stop')
            self.stopping_calls.append(self.step_ndx)
            if self.cfg['navigability_mode'] != 'none':
                new_image = obs['color_sensor'].copy()
                a_final = self._project_onto_image(
                    self._get_default_arrows(), new_image, obs['agent_state'],
                    obs['agent_state'].sensor_states['color_sensor'],
                    step=self.step_ndx,
                    goal=obs['goal']
                )
                images['color_sensor'] = new_image

        step_metadata = {
            'action_number': -10,
            'success': 1,
            'model': self.ActionVLM.name,
            'agent_location': obs['agent_state'].position,
            'called_stopping': called_stop
        }
        return a_final, images, step_metadata, a_goal, candidate_images

    def _draw_direction_arrow(self, roomtrack_map, direction_vector, position, coords, angle_text='', arrow_length=1):
        # 箭头的终点
        arrow_end = np.array(
            [position[0] + direction_vector[0] * arrow_length, position[1],
             position[2] + direction_vector[2] * arrow_length])  # 假设 y 轴是高度轴，不变

        # 将世界坐标转换为网格坐标
        arrow_end_coords = self._global_to_grid(arrow_end)

        # 绘制箭头
        cv2.arrowedLine(roomtrack_map, (coords[0], coords[1]),
                        (arrow_end_coords[0], arrow_end_coords[1]), WHITE, 4, tipLength=0.1)

        # 绘制文本，表示角度（假设为 30°，你可以根据实际情况调整）
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = 1
        text_thickness = 2

        # 获取文本的宽度和高度，用来居中文本
        (text_width, text_height), _ = cv2.getTextSize(angle_text, font, text_size, text_thickness)

        # 设置文本的位置为箭头终点稍微偏移
        text_end_coords = self._global_to_grid(np.array(
            [position[0] + direction_vector[0] * arrow_length * 1.5, position[1],
             position[2] + direction_vector[2] * arrow_length * 1.5]))
        text_position = (text_end_coords[0] - text_width // 2, text_end_coords[1] + text_height // 2)

        # 绘制文本
        cv2.putText(roomtrack_map, angle_text, text_position, font, text_size, (255, 255, 255), text_thickness,
                    cv2.LINE_AA)

    def generate_voxel(self, agent_state: habitat_sim.AgentState=None, zoom: int=9):
        """For visualization purposes, add the agent's position and actions onto the voxel map"""
        agent_coords = self._global_to_grid(agent_state.position)
        right = (agent_state.position[0] + zoom, 0, agent_state.position[2])
        right_coords = self._global_to_grid(right)
        delta = abs(agent_coords[0] - right_coords[0])

        topdown_map = self.voxel_map.copy()
        mask = np.all(self.explored_map == self.explored_color, axis=-1)
        topdown_map[mask] = self.explored_color

        # direction vector
        direction_vector = habitat_sim.utils.quat_rotate_vector(agent_state.rotation, habitat_sim.geo.FRONT)

        self._draw_direction_arrow(topdown_map, direction_vector, agent_state.position, agent_coords,
                                   angle_text="0")
        theta_60 = -np.pi / 3
        theta_30 = -np.pi / 6
        y_axis = np.array([0, 1, 0])
        quat_60 = habitat_sim.utils.quat_from_angle_axis(theta_60, y_axis)
        quat_30 = habitat_sim.utils.quat_from_angle_axis(theta_30, y_axis)
        direction_30_vector = habitat_sim.utils.quat_rotate_vector(quat_30, direction_vector)
        self._draw_direction_arrow(topdown_map, direction_30_vector, agent_state.position, agent_coords,
                                   angle_text="30")
        direction_60_vector = direction_30_vector.copy()
        for i in range(5):
            direction_60_vector = habitat_sim.utils.quat_rotate_vector(quat_60, direction_60_vector)
            angle = (i + 1) * 60 + 30
            self._draw_direction_arrow(topdown_map, direction_60_vector, agent_state.position, agent_coords,
                                       angle_text=str(angle))

        text_size = 1.25
        text_thickness = 1
        x, y = agent_coords
        font = cv2.FONT_HERSHEY_SIMPLEX

        text = str(self.step_ndx)
        (text_width, text_height), _ = cv2.getTextSize(text, font, text_size, text_thickness)
        circle_center = (agent_coords[0], agent_coords[1])
        circle_radius = max(text_width, text_height) // 2 + 15

        cv2.circle(topdown_map, circle_center, circle_radius, WHITE, -1)

        text_position = (circle_center[0] - text_width // 2, circle_center[1] + text_height // 2)
        cv2.circle(topdown_map, circle_center, circle_radius, RED, 1)
        cv2.putText(topdown_map, text, text_position, font, text_size, RED, text_thickness + 1)


        # Zoom the map
        max_x, max_y = topdown_map.shape[1], topdown_map.shape[0]
        x1 = max(0, x - delta)
        x2 = min(max_x, x + delta)
        y1 = max(0, y - delta)
        y2 = min(max_y, y + delta)

        zoomed_map = topdown_map[y1:y2, x1:x2]

        if self.step_ndx is not None:
            step_text = f'step {self.step_ndx}'
            cv2.putText(zoomed_map, step_text, (30, 90), font, 3, (255, 255, 255), 2, cv2.LINE_AA)

        return zoomed_map

    def update_voxel(self, r: float, theta: float, agent_state: habitat_sim.AgentState, temp_map: np.ndarray, effective_dist: float=3):
        agent_coords = self._global_to_grid(agent_state.position)

        # Mark unexplored regions
        unclipped = max(r, 0)
        local_coords = np.array([unclipped * np.sin(theta), 0, -unclipped * np.cos(theta)])
        global_coords = local_to_global(agent_state.position, agent_state.rotation, local_coords)
        point = self._global_to_grid(global_coords)
        cv2.line(self.voxel_map, agent_coords, point, self.unexplored_color, 40)

        # Mark directional regions
        cv2.line(temp_map, agent_coords, point, WHITE, 40) # whole area
        unclipped = min(r, effective_dist)
        local_coords = np.array([unclipped * np.sin(theta), 0, -unclipped * np.cos(theta)])
        global_coords = local_to_global(agent_state.position, agent_state.rotation, local_coords)
        point = self._global_to_grid(global_coords)
        cv2.line(temp_map, agent_coords, point, GREEN, 40) # effective area

    def _goal_proposer(self, a_initial: list, agent_state: habitat_sim.AgentState):
        min_angle = self.fov / self.cfg['spacing_ratio']

        unique = {}
        for mag, theta in a_initial:
            if theta in unique:
                unique[theta].append(mag)
            else:
                unique[theta] = [mag]

        arrowData = []

        for theta, mags in unique.items():
            mag = min(mags)
            arrowData.append([mag, theta])

        arrowData.sort(key=lambda x: x[1])
        thetas = set()
        out = []
        filter_thresh = 0
        f = list(filter(lambda x: x[0] > filter_thresh, arrowData))

        f.sort(key=lambda x: x[1])
        if f == []:
            return []
        # Add unexplored actions with spacing, starting with the longest one
        if len(f) > 0:
            longest = max(f, key=lambda x: x[0])
            longest_theta = longest[1]
            smallest_theta = longest[1]
            longest_ndx = f.index(longest)

            out.append([longest[0], longest[1]])
            thetas.add(longest[1])
            for i in range(longest_ndx + 1, len(f)):
                if f[i][1] - longest_theta > (min_angle * 0.45):
                    out.append([f[i][0], f[i][1]])
                    thetas.add(f[i][1])
                    longest_theta = f[i][1]
            for i in range(longest_ndx - 1, -1, -1):
                if smallest_theta - f[i][1] > (min_angle * 0.45):
                    out.append([f[i][0], f[i][1]])
                    thetas.add(f[i][1])
                    smallest_theta = f[i][1]

        out.sort(key=lambda x: x[1])
        return [(mag, theta) for mag, theta in out]

    def _preprocessing_module(self, obs: dict):
        """Excutes the navigability, action_proposer and projection submodules."""
        agent_state = obs['agent_state']
        images = {'color_sensor': obs['color_sensor'].copy()}
        candidate_images = {'color_sensor': obs['color_sensor'].copy()}
        a_goal_projected = None

        if self.cfg['navigability_mode'] == 'none':
            a_final = [
                # Actions for the w/o nav baseline
                (self.cfg['max_action_dist'], -0.36 * np.pi),
                (self.cfg['max_action_dist'], -0.28 * np.pi),
                (self.cfg['max_action_dist'], 0),
                (self.cfg['max_action_dist'], 0.28 * np.pi),
                (self.cfg['max_action_dist'], 0.36 * np.pi)
            ]
        else:
            a_initial = self._navigability(obs)
            a_final = self._action_proposer(a_initial, agent_state)
            if obs['goal_flag']:
                a_goal = self._goal_proposer(a_initial, agent_state)

        a_final_projected = self._projection(a_final, images, agent_state, obs['goal'])
        if obs['goal_flag']:
            a_goal_projected = self._projection(a_goal, candidate_images, agent_state, obs['goal'], candidate_flag=True)
        images['voxel_map'] = self._generate_voxel(a_final_projected, agent_state=agent_state, step=self.step_ndx)
        return a_final_projected, images, a_goal_projected, candidate_images

    def navigability(self, obs: dict, direction_idx: int):
        """Generates the set of navigability actions and updates the voxel map accordingly."""
        agent_state: habitat_sim.AgentState = obs['agent_state']
        if self.step_ndx == 0:
            self.init_pos = agent_state.position
        sensor_state = agent_state.sensor_states['color_sensor']
        rgb_image = obs['color_sensor']
        depth_image = obs[f'depth_sensor']
        if self.cfg['navigability_mode'] == 'depth_estimate':
            depth_image = self.depth_estimator.call(rgb_image)
        if self.cfg['navigability_mode'] == 'segmentation':
            depth_image = None

        navigability_mask = self._get_navigability_mask(
            rgb_image, depth_image, agent_state, sensor_state
        )

        sensor_range = np.deg2rad(self.fov / 2) * 1.5

        all_thetas = np.linspace(-sensor_range, sensor_range, 120)
        start = agent_frame_to_image_coords(
            [0, 0, 0], agent_state, sensor_state,
            resolution=self.resolution, focal_length=self.focal_length
        )  # agent的原点在图像坐标系的位置（像素）

        temp_map = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        for theta_i in all_thetas:
            r_i, theta_i = self._get_radial_distance(start, theta_i, navigability_mask, agent_state, sensor_state,
                                                     depth_image)
            if r_i is not None:
                self.update_voxel(
                    r_i, theta_i, agent_state, temp_map
                )
        angle = str(int(direction_idx * 30))
        self.panoramic_mask[angle] = np.all(temp_map == WHITE, axis=-1) | np.all(temp_map == GREEN, axis=-1)
        self.effective_mask[angle] = np.all(temp_map == GREEN, axis=-1)

    def _update_voxel(self, r: float, theta: float, agent_state: habitat_sim.AgentState, clip_dist: float, clip_frac: float):
        """Update the voxel map to mark actions as explored or unexplored"""
        agent_coords = self._global_to_grid(agent_state.position)

        # Mark explored regions
        clipped = min(r, clip_dist)
        local_coords = np.array([clipped * np.sin(theta), 0, -clipped * np.cos(theta)])
        global_coords = local_to_global(agent_state.position, agent_state.rotation, local_coords)
        point = self._global_to_grid(global_coords)
        cv2.line(self.explored_map, agent_coords, point, self.explored_color, self.voxel_ray_size)

    def _navigability(self, obs: dict):
        """Generates the set of navigability actions and updates the voxel map accordingly."""
        agent_state: habitat_sim.AgentState = obs['agent_state']
        sensor_state = agent_state.sensor_states['color_sensor']
        rgb_image = obs['color_sensor']
        depth_image = obs[f'depth_sensor']
        if self.cfg['navigability_mode'] == 'depth_estimate':
            depth_image = self.depth_estimator.call(rgb_image)
        if self.cfg['navigability_mode'] == 'segmentation':
            depth_image = None

        navigability_mask = self._get_navigability_mask(
            rgb_image, depth_image, agent_state, sensor_state
        )

        sensor_range =  np.deg2rad(self.fov / 2) * 1.5

        all_thetas = np.linspace(-sensor_range, sensor_range, self.cfg['num_theta'])
        start = agent_frame_to_image_coords(
            [0, 0, 0], agent_state, sensor_state,
            resolution=self.resolution, focal_length=self.focal_length
        )  # agent的原点在图像坐标系的位置（像素）

        a_initial = []
        for theta_i in all_thetas:
            r_i, theta_i = self._get_radial_distance(start, theta_i, navigability_mask, agent_state, sensor_state, depth_image)
            if r_i is not None:
                self._update_voxel(
                    r_i, theta_i, agent_state,
                    clip_dist=self.cfg['max_action_dist'], clip_frac=self.e_i_scaling
                )
                a_initial.append((r_i, theta_i))

        # draw explored circle
        if self.cfg['panoramic_padding'] == True:
            r_max, theta_max = max(a_initial, key=lambda x: x[0])
            self._update_panoramic_voxel(r_max, theta_max, agent_state,
                    clip_dist=self.cfg['max_action_dist'], clip_frac=self.e_i_scaling)
        return a_initial

    def get_spend(self):
        return self.ActionVLM.get_spend()  + self.PlanVLM.get_spend() + self.PredictVLM.get_spend() + self.GoalVLM.get_spend()

    # 请确保这段代码是在 class WMNavAgent(VLMNavAgent): 内部
    
    def _prompting(self, goal, a_final: list, images: dict, step_metadata: dict, subtask: str, goal_num: int):
        """
        Prompting component of BASEV2. Constructs the textual prompt and calls the action model.
        ✅ 修复：WMNavAgent 子类必须同步更新，添加 goal_num 参数
        """
        prompt_type = 'action'
        
        # ✅ 关键修复：传递 goal_num 给 _construct_prompt
        action_prompt = self._construct_prompt(
            goal, 
            prompt_type, 
            subtask, 
            num_actions=len(a_final), 
            goal_num=goal_num
        )

        prompt_images = [images['color_sensor']]
        if 'goal_image' in images:
            prompt_images.append(images['goal_image'])

        response = self.ActionVLM.call_chat(prompt_images, action_prompt)

        logging_data = {}
        try:
            response_dict = self._eval_response(response)
            step_metadata['action_number'] = int(response_dict['action'])
        except (IndexError, KeyError, TypeError, ValueError) as e:
            logging.error(f'Error parsing response {e}')
            step_metadata['success'] = 0
        finally:
            logging_data['ACTION_NUMBER'] = step_metadata.get('action_number')
            logging_data['ACTION_PROMPT'] = action_prompt
            logging_data['ACTION_RESPONSE'] = response

        return step_metadata, logging_data, response
    
    # def _goal_module(self, goal_image: np.array, a_goal, goal, goal_num):
    #     location_prompt = self._construct_prompt(goal, 'goal', num_actions=len(a_goal), goal_num=goal_num)
    #     location_response = self.GoalVLM.call([goal_image], location_prompt)
    #     dct = self._eval_response(location_response)

    #     try:
    #         number = int(dct['Number'])
    #     except:
    #         number = None

    #     return number, location_response
    def _goal_module(self, goal_image: np.array, a_goal, goal, goal_num):
        location_prompt = self._construct_prompt(goal, 'goal', num_actions=len(a_goal), goal_num=goal_num)
        location_response = self.GoalVLM.call([goal_image], location_prompt)
        dct = self._eval_response(location_response)

        try:
            # 🔴 兼容提取 Number
            if 'Number' in dct:
                number = int(dct['Number'])
            else:
                number = 0
        except:
            number = None

        return number, location_response

    def _get_goal_position(self, action_goal, idx, agent_state):
        # 初始化默认值，避免 UnboundLocalError
        r = None
        theta = None
        
        for key, value in action_goal.items():
            if value == idx:
                r, theta = key
                break
        
        # 如果没找到匹配的 idx，返回 None
        if r is None or theta is None:
            print(f"Warning: idx {idx} not found in action_goal")
            return None, None
        
        agent_coords = self._global_to_grid(agent_state.position)

        local_goal = np.array([r * np.sin(theta), 0, -r * np.cos(theta)])
        global_goal = local_to_global(agent_state.position, agent_state.rotation, local_goal)
        point = self._global_to_grid(global_goal)

        # get top down radius
        radius = 1  # real radius (m)
        local_radius = np.array([0, 0, -radius])
        global_radius = local_to_global(agent_state.position, agent_state.rotation, local_radius)
        radius_point = self._global_to_grid(global_radius)
        top_down_radius = int(np.linalg.norm(np.array(agent_coords) - np.array(radius_point)))

        temp_map = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        cv2.circle(temp_map, point, top_down_radius, WHITE, -1)
        goal_mask = np.all(temp_map == WHITE, axis=-1)

        return global_goal, goal_mask

    def _choose_action(self, obs: dict, goal_num: dict=None):
        agent_state = obs['agent_state']
        goal = obs['goal']

        a_final, images, step_metadata, a_goal, candidate_images = self._run_threads(obs, [obs['color_sensor']], goal, goal_num)

        goal_image = candidate_images['color_sensor'].copy()
        # 🔴 核心修复：每步清空历史目标点，绝不保留上一帧的幻觉坐标
        self.goal_position = [] 
        self.goal_mask = None
        if a_goal is not None:
            goal_number, location_response = self._goal_module(goal_image, a_goal, goal, goal_num)
            images['goal_image'] = goal_image
            if goal_number is not None and goal_number != 0:
                goal_position, self.goal_mask = self._get_goal_position(a_goal, goal_number, agent_state)
                self.goal_position.append(goal_position)

        step_metadata['object'] = goal

        if step_metadata['called_stopping']:
            step_metadata['action_number'] = -1
            agent_action = PolarAction.stop
            logging_data = {}
        else:
            if a_goal is not None and goal_number is not None and goal_number != 0:
                logging_data = {}
                logging_data['ACTION_NUMBER'] = int(goal_number)
                step_metadata['action_number'] = goal_number
                a_final = a_goal
            else:
                step_metadata, logging_data, _ = self._prompting(goal, a_final, images, step_metadata, obs['subtask'], goal_num)
            agent_action = self._action_number_to_polar(step_metadata['action_number'], list(a_final))
        
        if a_goal is not None:
            logging_data['LOCATOR_RESPONSE'] = location_response
        
        metadata = {
            'step_metadata': step_metadata,
            'logging_data': logging_data,
            'a_final': a_final,
            'images': images,
            'step': self.step_ndx
        }
        return agent_action, metadata

    # def _eval_response(self, response: str):
    #     """Converts the VLM response string into a dictionary, if possible"""
    #     import re
    #     result = re.sub(r"(?<=[a-zA-Z])'(?=[a-zA-Z])", "\\'", response)
    #     try:
    #         eval_resp = ast.literal_eval(result[result.index('{') + 1:result.rindex('}')]) # {{}}
    #         if isinstance(eval_resp, dict):
    #             return eval_resp
    #     except:
    #         try:
    #             eval_resp = ast.literal_eval(result[result.rindex('{'):result.rindex('}') + 1]) # {}
    #             if isinstance(eval_resp, dict):
    #                 return eval_resp
    #         except:
    #             try:
    #                 eval_resp = ast.literal_eval(result[result.index('{'):result.rindex('}')+1]) # {{}, {}}
    #                 if isinstance(eval_resp, dict):
    #                     return eval_resp
    #             except:
    #                 logging.error(f'Error parsing response {response}')
    #                 return {}
    def _eval_response(self, response: str):
        """Converts the VLM response string into a dictionary, using robust JSON parsing."""
        import json
        import re
        import ast
        import logging
        
        # 1. 剥离可能存在的 Markdown 代码块 (防止 json.loads 崩溃)
        clean_text = response.replace("```json", "").replace("```", "").strip()
        
        try:
            # 2. 精准定位最外层的大括号
            start_idx = clean_text.find('{')
            end_idx = clean_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = clean_text[start_idx:end_idx]
                
                # 3. 优先使用 json.loads (最稳定，原生支持多层嵌套字典)
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    # 如果 json.loads 失败 (比如大模型用了单引号)，启用 ast 备用方案
                    # 处理英文缩写中的单引号，如 didn't -> didn\'t
                    result = re.sub(r"(?<=[a-zA-Z])'(?=[a-zA-Z])", "\\'", json_str)
                    
                    # 🔴 修复：直接解析完整的字典字符串，绝不切除外围的 { } 
                    eval_resp = ast.literal_eval(result)
                    if isinstance(eval_resp, dict):
                        return eval_resp
        except Exception as e:
            logging.error(f'Error parsing response: {response}\nError detail: {e}')
            
        return {}

    def _planning_module(self, planning_image: list[np.array], previous_subtask, goal_reason: str, goal, goal_num):
        planning_prompt = self._construct_prompt(goal, 'planning', previous_subtask, goal_reason, goal_num=goal_num)
        planning_response = self.PlanVLM.call([planning_image], planning_prompt)
        planning_response = planning_response.replace('false', 'False').replace('true', 'True')
        dct = self._eval_response(planning_response)
        return dct
    
    def _predicting_module(self, evaluator_image, goal, goal_num):
        evaluator_prompt = self._construct_prompt(goal, 'predicting', goal_num=goal_num)
        # evaluator_response = self.PredictVLM.call([evaluator_image], evaluator_prompt)
        # 🔴 必须改为 call_chat，这样 API 才会把 System Prompt (打分员人设) 发过去！
        evaluator_response = self.PredictVLM.call_chat([evaluator_image], evaluator_prompt)
        # 🔴 核心调试：打印大模型最原始的回复，看看是不是 API 崩溃返回了 {"action": 0}！
        print(f"\n==============================================")
        print(f"[DEBUG] RAW PREDICT VLM TEXT: {evaluator_response}")
        print(f"==============================================\n")
        dct = self._eval_response(evaluator_response)
        return dct

    @staticmethod
    def _concat_panoramic(images, angles):
        try:
            height, width = images[0].shape[0], images[0].shape[1]
        except:
            height, width = 480, 640
        background_image = np.zeros((2 * height + 3 * 10, 3 * width + 4 * 10, 3), np.uint8)
        copy_images = np.array(images, dtype=np.uint8)
        for i in range(len(copy_images)):
            if i % 2 == 0:
                continue
            copy_images[i] = cv2.putText(copy_images[i], "Angle %d" % angles[i], (100, 100), cv2.FONT_HERSHEY_SIMPLEX,
                                         2, (255, 0, 0), 6, cv2.LINE_AA)
            row = i // 6
            col = (i // 2) % 3
            background_image[10 * (row + 1) + row * height:10 * (row + 1) + row * height + height:,
            10 * (col + 1) + col * width:10 * (col + 1) + col * width + width, :] = copy_images[i]
        return background_image

    def make_curiosity_value(self, pano_images, goal, goal_num):
            angles = (np.arange(len(pano_images))) * 30
            inference_image = self._concat_panoramic(pano_images, angles)

            # 🔴 关键加速与防崩溃：将巨幅全景图长宽各缩小一半！
            import cv2
            h, w = inference_image.shape[:2]
            inference_image = cv2.resize(inference_image, (w // 2, h // 2))
            # 🔴 终极取证：把这一步的全景图存到本地，看看长什么样！
            # cv2.imwrite(f"debug_pano_step_{self.step_ndx}.jpg", inference_image)

            response = self._predicting_module(inference_image, goal, goal_num)

            explorable_value = {}
            reason = {}

            # try:
            #     # 兼容处理 VLM 返回的数据
            #     for angle, values in response.items():
            #         # 如果 values 是字典且包含 'Score' (单目标/通用评分模式)
            #         if isinstance(values, dict) and 'Score' in values:
            #             explorable_value[angle] = values['Score']
            #             reason[angle] = values['Explanation']
            #         # 如果 VLM 返回了多目标结构 (例如 {'chair': 5, 'table': 3})
            #         elif isinstance(values, dict): 
            #             explorable_value[angle] = values
            #             reason[angle] = values
            #         else:
            #             # 保底
            #             explorable_value[angle] = 0
            #             reason[angle] = ""
            # except Exception as e:
            #     logging.error(f"Error parsing curiosity response: {e}")
            #     explorable_value, reason = {}, {}
            try:
                # 🔴 核心修复：顺从大模型的格式，精准剥离出 'scores' 字典
                target_dict = response.get('scores', response) if isinstance(response, dict) else response

                for angle, values in target_dict.items():
                    angle = str(angle).strip()
                    
                    if not angle.isdigit():
                        continue
                        
                    if isinstance(values, (int, float)): 
                        explorable_value[angle] = int(values)
                        reason[angle] = ""
                    elif isinstance(values, str) and values.isdigit(): 
                        explorable_value[angle] = int(values)
                        reason[angle] = ""
                    elif isinstance(values, dict) and 'Score' in values:
                        explorable_value[angle] = values.get('Score', 0)
                        reason[angle] = values.get('Explanation', "")
                    else:
                        explorable_value[angle] = 0
                        reason[angle] = ""
            except Exception as e:
                logging.error(f"Error parsing curiosity response: {e}")
                explorable_value, reason = {}, {}

            return inference_image, explorable_value, reason

    @staticmethod
    def _merge_evalue(arr, num):
        return np.minimum(arr, num)

    def update_curiosity_value(self, explorable_value, reason, goals, found_objects):
        """
        更新多通道 Curiosity Map (增强健壮性版本)
        """
        import traceback  # 必须导入，用于查看报错具体在哪一行

        try:
            # --- 1. 数据清洗与强类型转换 ---
            # 调试日志：看看 goals 到底传进来了什么鬼东西
            # logging.info(f"DEBUG: update_curiosity_value input goals type: {type(goals)}, content: {goals}")

            final_goals = []
            
            # 情况 A: 字典 (处理 {'goal': ...} 或其他字典结构)
            if isinstance(goals, dict):
                if 'goal' in goals:
                    val = goals['goal']
                    if isinstance(val, list): final_goals = val
                    else: final_goals = [val]
                else:
                    # 如果字典里没有 'goal' 键，直接取所有的值，或者取键作为备选
                    # logging.warning(f"Warning: goals is dict but no 'goal' key. Keys: {goals.keys()}")
                    # 尝试把字典所有的 value 展平放入列表
                    for v in goals.values():
                        if isinstance(v, list): final_goals.extend(v)
                        else: final_goals.append(v)
            
            # 情况 B: 列表
            elif isinstance(goals, list):
                final_goals = goals
            
            # 情况 C: 单个值 (字符串/数字)
            else:
                final_goals = [goals]

            # 统一转成字符串列表，过滤掉 None
            goal_names = [str(g) for g in final_goals if g is not None]
            
            # 如果清洗后列表为空，给一个默认值防止崩溃
            if not goal_names:
                logging.warning("Warning: goal_names is empty after cleaning! Using default 'target'.")
                goal_names = ['target']

            # --- 2. 初始化地图 ---
            if self.cvalue_map is None:
                self._initialize_curiosity_map(len(goal_names))

            # --- 3. 遍历更新 ---
            for goal_idx, target_goal in enumerate(goal_names):
                # 防止 goal_idx 越界 (如果中途目标变多了)
                if goal_idx >= self.cvalue_map.shape[2]:
                    break

                # 判断当前目标是否已找到
                is_found = False
                if isinstance(found_objects, list) and len(found_objects) > goal_idx:
                    val = found_objects[goal_idx]
                    if isinstance(val, bool): is_found = val
                    elif isinstance(val, str) and val == target_goal: is_found = True

                if is_found:
                    self.cvalue_map[:, :, goal_idx] = 0
                    continue
                
                # 遍历全景图的 12 个角度
                for i in range(12):
                    if i % 2 == 0: continue
                    
                    angle = str(int(i * 30))
                    last_angle = str(int((i-2)*30)) if i != 1 else '330'
                    next_angle = str(int((i+2)*30)) if i != 11 else '30'
                    
                    if angle not in self.panoramic_mask or np.all(self.panoramic_mask[angle] == False):
                        continue
                    
                    # 获取分数 (兼容字典和单数值)
                    score = 0
                    if angle in explorable_value:
                        val = explorable_value[angle]
                        if isinstance(val, (int, float)):
                            score = val
                        elif isinstance(val, dict):
                            # 安全获取：先试目标名，再试 'Score'，最后默认 0
                            score = val.get(target_goal, val.get('Score', 0))
                    
                    # 更新地图 (Vectorized操作)
                    intersection1 = self.effective_mask[last_angle] & self.effective_mask[angle]
                    intersection2 = self.effective_mask[angle] & self.effective_mask[next_angle]
                    mask_minus_intersection = self.effective_mask[angle] & ~intersection1 & ~intersection2
                    
                    self.cvalue_map[mask_minus_intersection, goal_idx] = self._merge_evalue(
                        self.cvalue_map[mask_minus_intersection, goal_idx], score
                    )
                    
                    if np.all(intersection2 == False): continue
                    
                    next_score = 0
                    if next_angle in explorable_value:
                        val = explorable_value[next_angle]
                        if isinstance(val, (int, float)):
                            next_score = val
                        elif isinstance(val, dict):
                            next_score = val.get(target_goal, val.get('Score', 0))
                        
                    self.cvalue_map[intersection2, goal_idx] = self._merge_evalue(
                        self.cvalue_map[intersection2, goal_idx], (score + next_score) / 2
                    )
            
            # --- 4. 融合与决策 ---
            priorities = self._calculate_target_priorities(goal_names, found_objects)
            fused_map = self._fuse_curiosity_maps(priorities)
            
            final_score = {}
            # 确保优先级索引合法
            max_priority_idx = np.argmax(priorities)
            if max_priority_idx < len(goal_names):
                current_target = goal_names[max_priority_idx]
            else:
                current_target = "unknown"

            for i in range(12):
                if i % 2 == 0: continue
                angle = str(int(i * 30))
                
                if angle not in self.panoramic_mask or np.all(self.panoramic_mask[angle] == False):
                    final_score[i] = 0
                    if angle in explorable_value:
                        val = explorable_value[angle]
                        if isinstance(val, (int, float)): final_score[i] = val
                        elif isinstance(val, dict): final_score[i] = val.get(current_target, val.get('Score', 0))
                else:
                    final_score[i] = np.mean(fused_map[self.panoramic_mask[angle]])
            
            idx = max(final_score, key=final_score.get) if final_score else 0
            
            angle_str = str(int(idx * 30))
            final_reason = ""
            if angle_str in reason:
                r_val = reason[angle_str]
                if isinstance(r_val, str): final_reason = r_val
                elif isinstance(r_val, dict): final_reason = r_val.get(current_target, r_val.get('Explanation', ""))
            
            if not final_reason:
                final_reason = f"Exploring direction {idx*30} for {current_target}"

            return idx, final_reason, current_target

        except Exception as e:
            # 🔴 捕捉所有错误并打印详细堆栈，防止 Env 崩溃
            logging.error(f"CRITICAL ERROR in update_curiosity_value: {e}")
            logging.error(traceback.format_exc()) # 打印具体的出错行号
            
            # 灾难恢复：返回一个随机动作，保证程序不退
            idx = np.random.choice([30, 90, 150, 210, 270, 330])
            return idx, "Error recovery", "unknown"
    
    def _calculate_target_priorities(self, goals, found_objects):
        """
        ✅ 新增：计算每个目标的优先级
        
        优先级公式：
        Priority = 0.5 × Proximity + 0.3 × Certainty + 0.2 × Value
        
        Args:
            goals: 所有目标列表
            found_objects: 已找到的目标
        
        Returns:
            priorities: numpy array，形状 (num_goals,)
        """
        num_goals = len(goals)
        priorities = np.zeros(num_goals)
        
        for i, goal in enumerate(goals):
            if goal in found_objects:
                priorities[i] = 0  # 已找到的优先级为0
                continue
            
            # 1. Proximity: 基于地图中该目标的最大分数（越高越近）
            max_score = np.max(self.cvalue_map[:, :, i])
            proximity = max_score / 10.0  # 归一化到 [0, 1]
            
            # 2. Certainty: 基于该目标分数的方差（方差越小越确定）
            variance = np.var(self.cvalue_map[:, :, i])
            certainty = 1.0 / (1.0 + variance)  # 方差越小，确定性越高
            
            # 3. Value: 固定值（可以根据任务重要性调整）
            value = 1.0  # 所有目标价值相同
            
            # 加权求和
            priorities[i] = 0.5 * proximity + 0.3 * certainty + 0.2 * value
        
        return priorities
    
    def _fuse_curiosity_maps(self, priorities):
        """
        ✅ 新增：融合多通道 Curiosity Map
        
        Args:
            priorities: 目标优先级数组，形状 (num_goals,)
        
        Returns:
            fused_map: 融合后的地图，形状 (H, W)
        """
        # 归一化优先级
        priorities = priorities / (np.sum(priorities) + 1e-8)
        
        # 加权求和
        # cvalue_map: (H, W, K)
        # priorities: (K,)
        # 结果: (H, W)
        fused_map = np.sum(
            self.cvalue_map * priorities[np.newaxis, np.newaxis, :], 
            axis=2
        )
        
        return fused_map

    def draw_cvalue_map(self, agent_state: habitat_sim.AgentState=None, zoom: int=9):
        """
        绘制好奇心价值地图 (修复多通道显示问题)
        """
        agent_coords = self._global_to_grid(agent_state.position)
        right = (agent_state.position[0] + zoom, 0, agent_state.position[2])
        right_coords = self._global_to_grid(right)
        delta = abs(agent_coords[0] - right_coords[0])

        # ✅ 关键修复：处理多通道地图
        if self.cvalue_map.shape[2] != 3:
            # 如果不是 3 通道 (例如 2 个目标或 5 个目标)
            # 策略：取所有通道的最大值 (Max Projection)，表示“任何目标的最高价值”
            # 结果形状变为 (H, W)
            combined_map = np.max(self.cvalue_map, axis=2)
            
            # 归一化并转为 uint8
            combined_map = (combined_map / 10 * 255).astype(np.uint8)
            
            # 将单通道复制为 3 通道 (Grayscale -> RGB)
            # stack 后形状变为 (H, W, 3)
            cvalue_map_rgb = np.stack([combined_map] * 3, axis=2)
        else:
            # 如果本来就是 3 通道 (例如单目标兼容模式或者巧合)，直接用
            cvalue_map_rgb = (self.cvalue_map / 10 * 255).astype(np.uint8)

        x, y = agent_coords
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Zoom the map
        max_x, max_y = cvalue_map_rgb.shape[1], cvalue_map_rgb.shape[0]
        x1 = max(0, x - delta)
        x2 = min(max_x, x + delta)
        y1 = max(0, y - delta)
        y2 = min(max_y, y + delta)

        zoomed_map = cvalue_map_rgb[y1:y2, x1:x2]

        if self.step_ndx is not None:
            step_text = f'step {self.step_ndx}'
            # 确保在图片范围内绘制
            if zoomed_map.shape[0] > 100 and zoomed_map.shape[1] > 200:
                cv2.putText(zoomed_map, step_text, (30, 90), font, 3, (0, 0, 0), 2, cv2.LINE_AA)

        return zoomed_map
    def make_plan(self, pano_images, previous_subtask, goal_reason, goal, goal_num):
        response = self._planning_module(pano_images, previous_subtask, goal_reason, goal, goal_num)

        try:
            goal_flag, subtask = response['Flag'], response['Subtask']
        except:
            print("planning failed!")
            print('response:', response)
            goal_flag, subtask = False, '{}'

        return goal_flag, subtask


    """
    修正后的 _construct_prompt 函数 - 适配多目标导航

    主要修改点：
    1. ✅ 删除所有硬编码的"chair is NOT sofa which is NOT bed"干扰信息
    2. ✅ 兼容 goal 为列表或字符串的情况
    3. ✅ 添加多目标进度上下文（已找到X个，还需找Y个）
    4. ✅ 为所有 prompt 类型添加 goal_num 参数支持

    使用方法：
    将此函数替换 WMNav_agent.py 中的 WMNavAgent._construct_prompt() 方法
    """

    # def _construct_prompt(self, goal, prompt_type: str, subtask: str = '{}', reason: str = '{}', 
    #                     num_actions: int = 0, goal_num: dict = None):
    #     """
    #     构建不同类型的 prompt
        
    #     Args:
    #         goal: 目标物体（字符串或列表）
    #         prompt_type: prompt 类型 ('goal', 'predicting', 'planning', 'action')
    #         subtask: 当前子任务
    #         reason: 选择该方向的原因
    #         num_actions: 可选动作数量
    #         goal_num: 多目标模式下的目标数量字典 {'total': int, 'found': int, 'remaining': list}
        
    #     Returns:
    #         str: 构建好的 prompt
    #     """
        
    #     # ==================== 处理目标参数 ====================
    #     # 兼容单目标（字符串）和多目标（列表）
    #     if isinstance(goal, list):
    #         goal_str = " and ".join([g.upper() for g in goal])
    #         goal_lower = " and ".join(goal)
    #     else:
    #         goal_str = goal.upper()
    #         goal_lower = goal
        
    #     # ==================== 构建多目标进度上下文 ====================
    #     progress_context = ""
    #     if goal_num:
    #         # 兼容字典和整数两种格式
    #         if isinstance(goal_num, dict):
    #             total = goal_num.get('total', 1)
    #             found = goal_num.get('found', 0)
    #             remaining = goal_num.get('remaining', [])
    #         else:
    #             # 如果是整数，转换为字典格式
    #             total = goal_num
    #             found = 0
    #             remaining = []
            
    #         if total > 1:
    #             if found > 0 and remaining:
    #                 remaining_str = " and ".join([r.upper() for r in remaining])
    #                 progress_context = (
    #                     f"PROGRESS UPDATE: You have successfully found {found} out of {total} targets. "
    #                     f"You need to find the remaining {total - found} target(s): {remaining_str}. "
    #                 )
    #             else:
    #                 progress_context = f"TASK: Find all {total} targets: {goal_str}. "
        
    #     # ==================== Prompt 类型：goal (目标定位) ====================
    #     if prompt_type == 'goal':
    #         location_prompt = (
    #             f"{progress_context}"
    #             f"The agent has been tasked with navigating to a {goal_str}. "
    #             f"The agent has sent you an image taken from its current location. "
    #             f"There are {num_actions} red arrows superimposed onto your observation, which represent potential positions. "
    #             f"These are labeled with a number in a white circle, which represent the location you can move to. "
    #             f"First, tell me whether the {goal_str} is in the image. Make sure the object you see is ACTUALLY a {goal_lower}. "
    #             f"Return number 0 if there is no {goal_lower}, or if you are not sure. "
    #             f"Second, if there is {goal_lower} in the image, then determine which circle best represents the location of the {goal_lower} "
    #             f"(close enough to the target - if a person is standing in that position, they can easily touch the {goal_lower}), and give the number and a reason. "
    #             f"If none of the circles represent the position of the {goal_lower}, return number 0, and give a reason why you returned 0. "
    #             f"Format your answer in the json {{'Number': <The number you choose>}}"
    #         )
    #         return location_prompt
    #     # if prompt_type == 'goal':
    #     #     location_prompt = (
    #     #         f"{progress_context}"
    #     #         f"The agent has been tasked with navigating to a {goal_str}. "
    #     #         f"The agent has sent you an image taken from its current location. "
    #     #         f"There are {num_actions} red arrows superimposed onto your observation, which represent potential positions. "
    #     #         f"These are labeled with a number in a white circle, which represent the location you can move to. "
    #     #         f"First, CAREFULLY CHECK if the {goal_str} is clearly visible in the image. "
    #     #         f"Do not hallucinate. If the object is blurry, far away, or you are guessing, return number 0. "
    #     #         f"Only identify the object if you are 100% sure it is a {goal_lower}. "
    #     #         f"Second, ONLY if there is a {goal_lower}, determine which circle best represents its location "
    #     #         f"(close enough to touch). "
    #     #         f"If none of the circles represent the position of the {goal_lower}, return number 0. "
    #     #         # 🔴 关键修改：强制要求先输出 Explanation，再输出 Number
    #     #         f"Format your answer strictly in the json {{'Explanation': '<State clearly what you see and why you chose the number>', 'Number': <The number you choose, or 0 if not found>}}"
    #     #     )
    #     #     return location_prompt
        
    #     # ==================== Prompt 类型：predicting (探索价值预测) ====================
    #     if prompt_type == 'predicting':
    #         evaluator_prompt = (
    #             f"{progress_context}"
    #             f"The agent has been tasked with navigating to a {goal_str}. "
    #             f"The agent has sent you the panoramic image describing your surrounding environment, "
    #             f"each image contains a label indicating the relative rotation angle(30, 90, 150, 210, 270, 330) with red fonts. "
    #             f"Your job is to assign a score to each direction (ranging from 0 to 10), judging whether this direction is worth exploring. "
    #             f"The following criteria should be used: "
    #             f"(1) If there is no visible way to move to other areas and it is clear that the target is not in sight, assign a score of 0. "
    #             f"(2) If the {goal_lower} is found, assign a score of 10. "
    #             f"(3) If there is a way to move to another area, assign a score based on your estimate of the likelihood of finding a {goal_lower}, using your common sense. "
    #             f"Moving to another area means there is a turn in the corner, an open door, a hallway, etc. "
    #             f"Note you CANNOT GO THROUGH CLOSED DOORS. CLOSED DOORS and GOING UP OR DOWN STAIRS are not considered. "
    #             f"For each direction, provide an explanation for your assigned score. "
    #             f"Format your answer in the json {{'30': {{'Score': <The score(from 0 to 10) of angle 30>, 'Explanation': <An explanation for your assigned score.>}}, "
    #             f"'90': {{...}}, '150': {{...}}, '210': {{...}}, '270': {{...}}, '330': {{...}}}}. "
    #             f"Answer Example: {{'30': {{'Score': 0, 'Explanation': 'Dead end with a wall. No sign of the target or any other room.'}}, "
    #             f"'90': {{'Score': 2, 'Explanation': 'Dining area. It is possible there is a doorway leading to other rooms.'}}, "
    #             f"..., '330': {{'Score': 2, 'Explanation': 'Living room area. Similar to 270, there is a possibility of other rooms.'}}}}"
    #         )
    #         return evaluator_prompt
        
    #     # ==================== Prompt 类型：planning (路径规划) ====================
    #     if prompt_type == 'planning':
    #         # 有原因和子任务的情况
    #         if reason != '' and subtask != '{}':
    #             planning_prompt = (
    #                 f"{progress_context}"
    #                 f"The agent has been tasked with navigating to a {goal_str}. "
    #                 f"The agent has sent you the following elements: "
    #                 f"(1)<The observed image>: The image taken from its current location. "
    #                 f"(2){reason}. This explains why you should go in this direction. "
    #                 f"Your job is to describe next place to go. "
    #                 f"To help you plan your best next step, I can give you some human suggestions: "
    #                 f"(1) If the {goal_lower} appears in the image, directly choose the target as the next step in the plan. "
    #                 f"(2) If the {goal_lower} is not found and the previous subtask {subtask} has not completed, continue to complete the last subtask {subtask} that has not been completed. "
    #                 f"(3) If the {goal_lower} is not found and the previous subtask {subtask} has already been completed, "
    #                 f"identify a new subtask by describing where you are going next to be more likely to find clues to the {goal_lower} "
    #                 f"and think about whether the {goal_lower} is likely to occur in that direction. "
    #                 f"Note you need to pay special attention to open doors and hallways, as they can lead to other unseen rooms. "
    #                 f"Note GOING UP OR DOWN STAIRS is an option. "
    #                 f"Format your answer in the json {{'Subtask': <Where you are going next>, 'Flag': <Whether the target is in your view, True or False>}}. "
    #                 f"Answer Example: {{'Subtask': 'Go to the hallway', 'Flag': False}} or "
    #                 f"{{'Subtask': 'Go to the {goal_lower}', 'Flag': True}} or "
    #                 f"{{'Subtask': 'Go to the open door', 'Flag': True}}"
    #             )
    #         else:
    #             # 无原因和子任务的情况
    #             planning_prompt = (
    #                 f"{progress_context}"
    #                 f"The agent has been tasked with navigating to a {goal_str}. "
    #                 f"The agent has sent you an image taken from its current location. "
    #                 f"Your job is to describe next place to go. "
    #                 f"To help you plan your best next step, I can give you some human suggestions: "
    #                 f"(1) If the {goal_lower} appears in the image, directly choose the target as the next step in the plan. "
    #                 f"(2) If the {goal_lower} is not found, describe where you are going next to be more likely to find clues to the {goal_lower} "
    #                 f"and analyze the room type and think about whether the {goal_lower} is likely to occur in that direction. "
    #                 f"Note you need to pay special attention to open doors and hallways, as they can lead to other unseen rooms. "
    #                 f"Note GOING UP OR DOWN STAIRS is an option. "
    #                 f"Format your answer in the json {{'Subtask': <Where you are going next>, 'Flag': <Whether the target is in your view, True or False>}}. "
    #                 f"Answer Example: {{'Subtask': 'Go to the hallway', 'Flag': False}} or "
    #                 f"{{'Subtask': 'Go to the {goal_lower}', 'Flag': True}} or "
    #                 f"{{'Subtask': 'Go to the open door', 'Flag': True}}"
    #             )
    #         return planning_prompt
        
    #     # ==================== Prompt 类型：action (动作选择) ====================
    #     if prompt_type == 'action':
    #         # 有子任务的情况
    #         if subtask != '{}':
    #             action_prompt = (
    #                 f"{progress_context}"
    #                 f"TASK: {subtask}. Your final task is to NAVIGATE TO THE NEAREST {goal_str}, and get as close to it as possible. "
    #                 f"There are {num_actions - 1} red arrows superimposed onto your observation, which represent potential actions. "
    #                 f"These are labeled with a number in a white circle, which represent the location you would move to if you took that action. "
    #                 f"{'NOTE: choose action 0 if you want to TURN AROUND or DONT SEE ANY GOOD ACTIONS. ' if self.step_ndx - self.turned >= self.cfg['turn_around_cooldown'] else ''}"
    #                 f"In order to complete the subtask {subtask} and eventually the final task NAVIGATING TO THE NEAREST {goal_str}, "
    #                 f"explain which action achieves that best. "
    #                 f"Return your answer as {{'action': <action_key>}}. "
    #                 f"Note you CANNOT GO THROUGH CLOSED DOORS, and you DO NOT NEED TO GO UP OR DOWN STAIRS"
    #             )
    #         else:
    #             # 无子任务的情况
    #             action_prompt = (
    #                 f"{progress_context}"
    #                 f"TASK: NAVIGATE TO THE NEAREST {goal_str}, and get as close to it as possible. "
    #                 f"Use your prior knowledge about where items are typically located within a home. "
    #                 f"There are {num_actions - 1} red arrows superimposed onto your observation, which represent potential actions. "
    #                 f"These are labeled with a number in a white circle, which represent the location you would move to if you took that action. "
    #                 f"{'NOTE: choose action 0 if you want to TURN AROUND or DONT SEE ANY GOOD ACTIONS. ' if self.step_ndx - self.turned >= self.cfg['turn_around_cooldown'] else ''}"
    #                 f"First, tell me what you see in your sensor observation, and if you have any leads on finding the {goal_str}. "
    #                 f"Second, tell me which general direction you should go in. "
    #                 f"Lastly, explain which action achieves that best, and return it as {{'action': <action_key>}}. "
    #                 f"Note you CANNOT GO THROUGH CLOSED DOORS, and you DO NOT NEED TO GO UP OR DOWN STAIRS"
    #             )
    #         return action_prompt
        
    #     raise ValueError('Prompt type must be goal, predicting, planning, or action')
    def _construct_prompt(self, goal: str, prompt_type:str, subtask: str='{}', reason: str='{}', num_actions: int=0, goal_num: dict=None):
        
        # 统一处理目标字符串（兼容多目标列表或单目标）
        goal_str = str(goal).upper()
        
        progress_context = ""
        if goal_num is not None:
            progress_context = f"Progress: Found {goal_num['found']}/{goal_num['total']} targets. Remaining: {goal_num['remaining']}. "

        # ==================== 1. Goal 提示词 (融合：严格物体定义 + 强制理由输出) ====================
        if prompt_type == 'goal':
            location_prompt = (
                f"{progress_context}"
                f"The agent has been tasked with navigating to a {goal_str}. The agent has sent you an image taken from its current location. "
                f"There are {num_actions} red arrows superimposed onto your observation, which represent potential positions. " 
                f"These are labeled with a number in a white circle, which represent the location you can move to. "
                f"First, CAREFULLY check whether the {goal} is in the image, and make sure the object you see is ACTUALLY a {goal}. "
                f"Note a chair must have a backrest and a chair is not a stool. Note a chair is NOT sofa(couch) which is NOT a bed. "
                f"DO NOT hallucinate. Return number 0 if there is no {goal}, or if you are not sure. "
                f"Second, if there is {goal} in the image, determine which circle best represents the location of the {goal} "
                f"(close enough to the target. If a person is standing in that position, they can easily touch the {goal}). "
                f"If none of the circles represent the position of the {goal}, return number 0. "
                f"Format your answer strictly in the json {{'Explanation': '<State what you see and why>', 'Number': <The number you choose, or 0>}}"
            )
            return location_prompt

        # ==================== 2. Predicting 提示词 (融合：多目标独立打分机制) ====================
        # if prompt_type == 'predicting':
        #     # 动态构建需要独立打分的目标格式
        #     if goal_num and 'remaining' in goal_num and goal_num['remaining']:
        #         target_list = goal_num['remaining']
        #     else:
        #         target_list = [goal_str] if isinstance(goal_str, str) else goal_str
                
        #     # 生成类似 "'BED': <0-10>, 'TV': <0-10>" 的 JSON 格式字符串
        #     score_format = ", ".join([f"'{t}': <score 0-10>" for t in target_list])
            
        #     evaluator_prompt = (
        #         f"{progress_context}"
        #         f"The agent is searching for the following targets: {goal_str}. "
        #         f"You will see a panoramic image with red labels indicating relative rotation angles (30, 90, 150, 210, 270, 330). "
        #         f"Your job is to evaluate each direction and assign a SEPARATE exploration score (0 to 10) for EACH remaining target. "
        #         f"Please follow these step-by-step instructions for EACH TARGET INDEPENDENTLY:\n"
        #         f"(1) If a direction is a dead end (e.g., blank wall, corner) with no path forward, and the target is NOT visible, assign a score of 0 for that target.\n"
        #         f"(2) If a specific target is CLEARLY visible in a direction, assign a score of 10 for that target.\n"
        #         f"(3) If a direction has an open door or hallway leading to a new area, assign a score based on your common-sense estimate of finding THAT SPECIFIC TARGET there. "
        #         f"(e.g., A bathroom door gets a high score for 'TOILET' but a low score for 'BED').\n"
        #         f"Note you CANNOT GO THROUGH CLOSED DOORS. STAIRS are not considered.\n\n"
        #         f"Format your answer STRICTLY as a JSON object with this exact structure: \n"
        #         f"{{'30': {{{score_format}, 'Explanation': '<reason>'}}, '90': {{{score_format}, 'Explanation': '<reason>'}}, ...}}\n"
        #         f"Answer Example: {{'30': {{'BED': 0, 'TOILET': 0, 'Explanation': 'Dead end with a wall.'}}, "
        #         f"'90': {{'BED': 2, 'TOILET': 9, 'Explanation': 'Bathroom door visible, highly likely for toilet but not bed.'}}}}"
        #     )
        #     return evaluator_prompt


        # if prompt_type == 'predicting':
        #     if goal_num and 'remaining' in goal_num and goal_num['remaining']:
        #         target_list = goal_num['remaining']
        #     else:
        #         target_list = [goal_str] if isinstance(goal_str, str) else goal_str
                
        #     # 动态生成极其明确的 Answer Example，防止模型输出 action: 0
        #     score_format = ", ".join([f"'{t}': <0-10>" for t in target_list])
        #     example_scores_0 = ", ".join([f"'{t}': 0" for t in target_list])
        #     example_scores_8 = ", ".join([f"'{t}': 8" if i==0 else f"'{t}': 0" for i, t in enumerate(target_list)])
            
        #     evaluator_prompt = (
        #         f"{progress_context}"
        #         f"The agent is searching for target(s): {goal_str}. "
        #         f"You will see a panoramic image with red labels indicating rotation angles (30, 90, 150, 210, 270, 330). "
        #         f"Your job is to assign an exploration score (0 to 10) for EACH remaining target in EACH direction.\n"
        #         f"CRITICAL WARNING: DO NOT output the word 'action'! ONLY output the angles and scores.\n"
        #         f"Format your answer STRICTLY as a JSON object with this exact structure:\n"
        #         f"{{'30': {{{score_format}, 'Explanation': '<reason>'}}, '90': {{{score_format}, 'Explanation': '<reason>'}}, ...}}\n"
        #         f"Answer Example:\n"
        #         f"{{'30': {{{example_scores_0}, 'Explanation': 'Dead end.'}}, '90': {{{example_scores_8}, 'Explanation': 'Open hallway.'}}, ...}}"
        #     )
        #     return evaluator_prompt


        # # ==================== 2. Predicting 提示词 (完美融合原版逻辑与多目标格式) ====================
        # if prompt_type == 'predicting':
        #     # 获取当前剩余的所有目标
        #     if goal_num and 'remaining' in goal_num and goal_num['remaining']:
        #         target_list = goal_num['remaining']
        #     else:
        #         target_list = [goal_str] if isinstance(goal_str, str) else goal_str
                
        #     # 动态生成大写目标名和格式字符串，适配任意数量的目标
        #     targets_str = ", ".join([t.upper() for t in target_list])
        #     score_format = ", ".join([f"'{t}': <0-10>" for t in target_list])
            
        #     # 为 Answer Example 生成动态分数模板
        #     example_scores_0 = ", ".join([f"'{t}': 0" for t in target_list])
        #     example_scores_mixed = ", ".join([f"'{t}': 10" if i==0 else f"'{t}': 2" for i, t in enumerate(target_list)])
            
        #     evaluator_prompt = (
        #         f"{progress_context}"
        #         f"The agent has been tasked with navigating to the following target(s): {targets_str}. "
        #         f"The agent has sent you the panoramic image describing your surrounding environment, each image contains a label indicating the relative rotation angle (30, 90, 150, 210, 270, 330) with red fonts.\n"
        #         f"Your job is to assign a score to EACH direction for EACH target (ranging from 0 to 10), judging whether this direction is worth exploring.\n"
        #         f"To help you describe the layout of your surrounding, please follow my step-by-step instructions:\n"
        #         f"(1) If there is no visible way to move to other areas and it is clear that the target is not in sight, assign a score of 0. Note a chair must have a backrest and a chair is not a stool. Note a chair is NOT sofa(couch) which is NOT a bed.\n"
        #         f"(2) 🚨 CRITICAL: If any specific target is found or DIRECTLY VISIBLE in a direction, assign a score of 10 for THAT specific target immediately.\n"
        #         f"(3) If there is a way to move to another area, assign a score based on your estimate of the likelihood of finding each target, using your common sense. Moving to another area means there is a turn in the corner, an open door, a hallway, etc. Note you CANNOT GO THROUGH CLOSED DOORS. CLOSED DOORS and GOING UP OR DOWN STAIRS are not considered.\n"
        #         f"WARNING: DO NOT output the word 'action'! ONLY output the angles, the target scores, and explanations.\n"
        #         f"For each direction, provide an explanation for your assigned scores. Format your answer STRICTLY as a JSON object:\n"
        #         f"{{'30': {{{score_format}, 'Explanation': '<An explanation for your assigned score.>'}}, '90': {{{score_format}, 'Explanation': '<reason>'}}, ...}}\n"
        #         f"Answer Example:\n"
        #         f"{{'30': {{{example_scores_0}, 'Explanation': 'Dead end with a recliner. No sign of the targets or any other room.'}}, '90': {{{example_scores_mixed}, 'Explanation': 'The first target is clearly visible here. The hallway also leads to other areas where the second target might be.'}}, ...}}"
        #     )
        #     return evaluator_prompt

        # # ==================== 2. Predicting 提示词 (文字推理 + 扁平JSON版) ====================
        # if prompt_type == 'predicting':
        #     if goal_num and 'remaining' in goal_num and goal_num['remaining']:
        #         target_list = goal_num['remaining']
        #     else:
        #         target_list = [goal_str] if isinstance(goal_str, str) else goal_str
                
        #     targets_str = ", ".join([t.upper() for t in target_list])
            
        #     evaluator_prompt = (
        #         f"{progress_context}\n"
        #         f"You are evaluating a panoramic image to find: {targets_str}.\n"
        #         f"The image contains 6 panels labeled with angles: 30, 90, 150, 210, 270, 330.\n"
        #         f"For each angle, look at the ACTUAL panel and describe what you see, then score it 0-10.\n"
        #         f"10 = target or its typical room is clearly visible. 0 = blank wall or dead end.\n"
        #         f"Output ONLY valid JSON. No markdown. No 'action' field.\n"
        #         f"Format: {{\"30\": {{\"Explanation\": \"<what you actually see>\", \"Score\": <0-10>}}, "
        #         f"\"90\": {{\"Explanation\": \"<what you actually see>\", \"Score\": <0-10>}}, "
        #         f"\"150\": {{\"Explanation\": \"<what you actually see>\", \"Score\": <0-10>}}, "
        #         f"\"210\": {{\"Explanation\": \"<what you actually see>\", \"Score\": <0-10>}}, "
        #         f"\"270\": {{\"Explanation\": \"<what you actually see>\", \"Score\": <0-10>}}, "
        #         f"\"330\": {{\"Explanation\": \"<what you actually see>\", \"Score\": <0-10>}}}}"
        #     )
        #     return evaluator_prompt

# ==================== 2. Predicting 提示词 (终极完全体：思维链 + 顺从天性) ====================
        if prompt_type == 'predicting':
            if goal_num and 'remaining' in goal_num and goal_num['remaining']:
                target_list = goal_num['remaining']
            else:
                target_list = [goal_str] if isinstance(goal_str, str) else goal_str
                
            targets_str = ", ".join([t.upper() for t in target_list])
            
            evaluator_prompt = (
                f"{progress_context}\n"
                f"Task: Evaluate the panoramic image to find: {targets_str}.\n"
                f"The image shows 6 panels labeled with angles: 30, 90, 150, 210, 270, 330.\n"
                f"Step 1: For each angle, briefly describe what you actually see.\n"
                f"Step 2: Assign a score from 0 to 10 for each angle (10 = target clearly visible, 0 = dead end/blank wall).\n"
                f"OUTPUT FORMAT: You MUST output a JSON object containing a 'scores' dictionary AND an 'action' key.\n"
                f"Format EXACTLY like this (replace <description> and <score>):\n"
                f"{{\n"
                f"  \"scores\": {{\n"
                f"    \"30\": {{\"Explanation\": \"<description>\", \"Score\": <score>}},\n"
                f"    \"90\": {{\"Explanation\": \"<description>\", \"Score\": <score>}},\n"
                f"    \"150\": {{\"Explanation\": \"<description>\", \"Score\": <score>}},\n"
                f"    \"210\": {{\"Explanation\": \"<description>\", \"Score\": <score>}},\n"
                f"    \"270\": {{\"Explanation\": \"<description>\", \"Score\": <score>}},\n"
                f"    \"330\": {{\"Explanation\": \"<description>\", \"Score\": <score>}}\n"
                f"  }},\n"
                f"  \"action\": 0\n"
                f"}}"
            )
            return evaluator_prompt

        # ==================== 3. Planning 提示词 (融合：寻找开放空间/门 + 子任务继承) ====================
        # if prompt_type == 'planning':
        #     if reason != '' and subtask != '{}':
        #         planning_prompt = (
        #             f"{progress_context}"
        #             f"The agent has been tasked with navigating to a {goal_str}. The agent has sent you the following elements: "
        #             f"(1)<The observed image>: The image taken from its current location. "
        #             f"(2){reason}. This explains why you should go in this direction. "
        #             f"Your job is to describe next place to go. Human suggestions: "
        #             f"(1) If the {goal} appears in the image, directly choose the target as the next step. Note a chair must have a backrest and is not a stool/sofa. "
        #             f"(2) If the {goal} is not found and the previous subtask {subtask} has not completed, continue to complete the last subtask {subtask}. "
        #             f"(3) If the {goal} is not found and the previous subtask {subtask} has already been completed. Identify a new subtask by describing where you are going next to be more likely to find clues. "
        #             f"Note you need to pay special attention to open doors and hallways, as they can lead to other unseen rooms. Note you CANNOT GO THROUGH CLOSED DOORS, and GOING UP OR DOWN STAIRS is NOT considered. "
        #             f"Format your answer in the json {{'Subtask': '<Where you are going next>', 'Flag': <True or False>}}. "
        #         )
        #     else:
        #         planning_prompt = (
        #             f"{progress_context}"
        #             f"The agent has been tasked with navigating to a {goal_str}. The agent has sent you an image taken from its current location. "
        #             f"Your job is to describe next place to go. Human suggestions: "
        #             f"(1) If the {goal} appears in the image, directly choose the target as the next step. Note a chair must have a backrest and is not a stool/sofa. "
        #             f"(2) If the {goal} is not found, describe where you are going next to find clues. Analyze the room type and think about whether the {goal} is likely to occur in that direction. "
        #             f"Note you need to pay special attention to open doors and hallways. Note you CANNOT GO THROUGH CLOSED DOORS, and GOING UP OR DOWN STAIRS is NOT considered. "
        #             f"Format your answer in the json {{'Subtask': '<Where you are going next>', 'Flag': <True or False>}}. "
        #         )
        #     return planning_prompt

        # if prompt_type == 'planning':
        #     if reason != '' and subtask != '{}':
        #         planning_prompt = (
        #             f"{progress_context}"
        #             f"The agent has been tasked with navigating to a {goal_str}. The agent has sent you the following elements: "
        #             f"(1)<The observed image>: The image taken from its current location. "
        #             f"(2){reason}. This explains why you should go in this direction. "
        #             f"Your job is to describe next place to go. Human suggestions: "
        #             f"(1) If the {goal} appears in the image, directly choose the target as the next step. Note a chair must have a backrest and is not a stool/sofa. "
        #             f"(2) If the {goal} is not found and the previous subtask {subtask} has not completed, continue to complete the last subtask {subtask}. "
        #             f"(3) If the {goal} is not found and the previous subtask {subtask} has already been completed. Identify a new subtask by describing where you are going next to be more likely to find clues. "
        #             f"Note you CANNOT GO THROUGH CLOSED DOORS, and GOING UP OR DOWN STAIRS is NOT considered. "
        #             f"CRITICAL: 'Flag' must be True ONLY IF the target object is CLEARLY visible in the image. Otherwise, it MUST be False! "
        #             f"Format your answer in the json {{'Subtask': '<Where you are going next>', 'Flag': <True or False>}}. "
        #         )
        #     else:
        #         planning_prompt = (
        #             f"{progress_context}"
        #             f"The agent has been tasked with navigating to a {goal_str}. The agent has sent you an image taken from its current location. "
        #             f"Your job is to describe next place to go. Human suggestions: "
        #             f"(1) If the {goal} appears in the image, directly choose the target as the next step. Note a chair must have a backrest and is not a stool/sofa. "
        #             f"(2) If the {goal} is not found, describe where you are going next to find clues. Analyze the room type and think about whether the {goal} is likely to occur in that direction. "
        #             f"Note you CANNOT GO THROUGH CLOSED DOORS, and GOING UP OR DOWN STAIRS is NOT considered. "
        #             f"CRITICAL: 'Flag' must be True ONLY IF the target object is CLEARLY visible in the image. Otherwise, it MUST be False! "
        #             f"Format your answer in the json {{'Subtask': '<Where you are going next>', 'Flag': <True or False>}}. "
        #         )
        #     return planning_prompt

        # ==================== 3. Planning 提示词 (放宽门槛：看到关联房间也算数) ====================
        if prompt_type == 'planning':
            if reason != '' and subtask != '{}':
                planning_prompt = (
                    f"{progress_context}"
                    f"The agent has been tasked with navigating to a {goal_str}. The agent has sent you the following elements: "
                    f"(1)<The observed image>: The image taken from its current location. "
                    f"(2){reason}. This explains why you should go in this direction. "
                    f"Your job is to describe next place to go. Human suggestions: "
                    f"(1) If the {goal} appears in the image, directly choose the target as the next step. Note a chair must have a backrest and is not a stool/sofa. "
                    f"(2) If the {goal} is not found and the previous subtask {subtask} has not completed, continue to complete the last subtask {subtask}. "
                    f"(3) If the {goal} is not found and the previous subtask {subtask} has already been completed. Identify a new subtask by describing where you are going next to be more likely to find clues. "
                    f"Note you CANNOT GO THROUGH CLOSED DOORS, and GOING UP OR DOWN STAIRS is NOT considered. "
                    f"CRITICAL: 'Flag' must be True IF the target object OR a room that typically contains it (e.g., a bathroom for a toilet) is visible in the image. Otherwise, False. "
                    f"Format your answer in the json {{'Subtask': '<Where you are going next>', 'Flag': <True or False>}}. "
                )
            else:
                planning_prompt = (
                    f"{progress_context}"
                    f"The agent has been tasked with navigating to a {goal_str}. The agent has sent you an image taken from its current location. "
                    f"Your job is to describe next place to go. Human suggestions: "
                    f"(1) If the {goal} appears in the image, directly choose the target as the next step. Note a chair must have a backrest and is not a stool/sofa. "
                    f"(2) If the {goal} is not found, describe where you are going next to find clues. Analyze the room type and think about whether the {goal} is likely to occur in that direction. "
                    f"Note you CANNOT GO THROUGH CLOSED DOORS, and GOING UP OR DOWN STAIRS is NOT considered. "
                    f"CRITICAL: 'Flag' must be True IF the target object OR a room that typically contains it (e.g., a bathroom for a toilet) is visible in the image. Otherwise, False. "
                    f"Format your answer in the json {{'Subtask': '<Where you are going next>', 'Flag': <True or False>}}. "
                )
            return planning_prompt
        # ==================== 4. Action 提示词 (融合：家庭先验知识 + 严禁瞎转圈机制) ====================
        if prompt_type == 'action':
            if subtask != '{}':
                action_prompt = (
                    f"{progress_context}"
                    f"TASK: {subtask}. Your final task is to NAVIGATE TO THE NEAREST {goal_str}, and get as close to it as possible. "
                    f"Use your prior knowledge about where items are typically located within a home. "
                    f"There are {num_actions - 1} red arrows superimposed onto your observation. These are labeled with a number in a white circle. "
                    f"CRITICAL INSTRUCTION: DO NOT choose action 0 just because you don't see the target! "
                    f"You MUST choose a numbered arrow (1, 2, 3...) that leads towards an open space, door, or hallway to KEEP EXPLORING. "
                    f"Only choose 0 if you are completely trapped facing a blank wall and MUST turn around. "
                    f"In order to complete the subtask {subtask} and eventually the final task, explain which action achieves that best. "
                    f"Note you CANNOT GO THROUGH CLOSED DOORS, and you DO NOT NEED TO GO UP OR DOWN STAIRS. "
                    f"Format your answer strictly as a JSON object: {{'Reasoning': '<explain what you see and why you chose this action>', 'action': <action_key>}}"
                )
            else:
                action_prompt = (
                    f"{progress_context}"
                    f"TASK: NAVIGATE TO THE NEAREST {goal_str}, and get as close to it as possible. "
                    f"Use your prior knowledge about where items are typically located within a home. "
                    f"There are {num_actions - 1} red arrows superimposed onto your observation. These are labeled with a number in a white circle. "
                    f"CRITICAL INSTRUCTION: DO NOT choose action 0 just because you don't see the target! "
                    f"You MUST choose a numbered arrow (1, 2, 3...) that leads towards an open space, door, or hallway to KEEP EXPLORING. "
                    f"Only choose 0 if you are completely trapped facing a blank wall and MUST turn around. "
                    f"First, tell me what you see in your sensor observation, and if you have any leads on finding {goal_str}. Second, tell me which general direction you should go in. "
                    f"Note you CANNOT GO THROUGH CLOSED DOORS, and you DO NOT NEED TO GO UP OR DOWN STAIRS. "
                    f"Format your answer strictly as a JSON object: {{'Reasoning': '<explain what you see and why you chose this action>', 'action': <action_key>}}"
                )
            return action_prompt

        raise ValueError('Prompt type must be goal, predicting, planning, or action')