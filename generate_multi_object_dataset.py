"""
Multi-Object Episode Generator for HM3D
生成多目标导航数据集示例
"""

import json
import gzip
import random
import os
from pathlib import Path
import numpy as np
import glob  # 新增这一行

class MultiObjectEpisodeGenerator:
    def __init__(self, original_dataset_path, output_path):
        """
        Args:
            original_dataset_path: 原始单目标数据集路径
                例如: "data/objectnav_hm3d_v2/val/val.json.gz"
            output_path: 输出的多目标数据集路径
                例如: "data/multi_objectnav_hm3d/val/val.json.gz"
        """
        self.original_path = original_dataset_path
        self.output_path = output_path
        self.episodes = []
        self.goals_by_category = {}  # ✅ 新增：存储场景中物体的位置信息
        
    def load_original_dataset(self):
        """加载原始单目标数据集（分片版本）+ goals_by_category"""
        print(f"📂 Loading dataset from: {self.original_path}")
        
        # 1. 读取主文件
        with gzip.open(self.original_path, 'rt') as f:
            main_data = json.load(f)
        
        # 2. 读取 content 目录下的所有场景文件
        content_dir = os.path.join(os.path.dirname(self.original_path), 'content')
        
        all_episodes = []
        scene_files = sorted(glob.glob(f"{content_dir}/*.json.gz"))
        
        print(f"📊 Found {len(scene_files)} scene files")
        
        for scene_file in scene_files:
            with gzip.open(scene_file, 'rt') as f:
                scene_data = json.load(f)
                episodes = scene_data.get('episodes', [])
                all_episodes.extend(episodes)
                
                # ✅ 加载 goals_by_category（包含物体位置信息）
                scene_hash = os.path.basename(scene_file).split('.')[0]
                self.goals_by_category[scene_hash] = scene_data.get('goals_by_category', {})
                
                print(f"  ✅ {os.path.basename(scene_file)}: {len(episodes)} episodes, "
                      f"{len(self.goals_by_category[scene_hash])} goal categories")
        
        print(f"\n✅ Total loaded: {len(all_episodes)} episodes")
        
        return {'episodes': all_episodes}
    
    # 1. 在 MultiObjectEpisodeGenerator 类中新增此函数
    def _get_nearby_categories(self, scene_hash, start_pos, max_dist=15.0, max_y_diff=1.5):
        """
        ✅ 新增：筛选出距离起点较近（15米内），且大致在同一楼层（高度差1.5米内）的物体类别
        """
        nearby_cats = set()
        goals_dict = self.goals_by_category.get(scene_hash, {})
        
        for cat_key, instances in goals_dict.items():
            # 解析出真正的类别名 (例如从 "TEEsavR23oF_bed" 提取出 "bed")
            cat_name = cat_key.split('_', 1)[1] if '_' in cat_key else cat_key
            
            for inst in instances:
                obj_pos = inst['position']
                
                # 计算水平距离和垂直高度差 (Habitat 中 y 轴通常是高度)
                dx = start_pos[0] - obj_pos[0]
                dy = start_pos[1] - obj_pos[1]
                dz = start_pos[2] - obj_pos[2]
                
                horizontal_dist = (dx**2 + dz**2)**0.5
                
                # 🔴 过滤条件：水平距离在 15 米内，且高度差在 1.5 米以内
                if horizontal_dist < max_dist and abs(dy) < max_y_diff:
                    nearby_cats.add(cat_name)
                    break # 这个类别只要有一个实例在附近，就满足条件
                    
        return list(nearby_cats)

    
    def group_episodes_by_scene(self, data):
        """按场景分组episodes"""
        scene_episodes = {}
        
        for episode in data['episodes']:
            scene_id = episode['scene_id']
            if scene_id not in scene_episodes:
                scene_episodes[scene_id] = []
            scene_episodes[scene_id].append(episode)
        
        print(f"📊 Found {len(scene_episodes)} unique scenes")
        return scene_episodes
    
    def get_object_categories(self, scene_episodes):
        """获取每个场景中的物体类别"""
        scene_objects = {}
        
        for scene_id, episodes in scene_episodes.items():
            # 收集该场景中所有出现过的物体类别
            categories = set()
            for ep in episodes:
                categories.add(ep['object_category'])
            
            scene_objects[scene_id] = list(categories)
            print(f"  Scene {scene_id}: {len(categories)} object categories")
        
        return scene_objects
    
    # def create_multi_object_episode(self, scene_id, episodes, num_objects=3):
    #     """
    #     创建一个多目标episode
        
    #     Args:
    #         scene_id: 场景ID
    #         episodes: 该场景的所有单目标episodes
    #         num_objects: 目标物体数量（默认3个）
        
    #     Returns:
    #         multi_episode: 多目标episode字典
    #     """
    #     # 1. 获取不同类别的物体
    #     category_episodes = {}
    #     for ep in episodes:
    #         cat = ep['object_category']
    #         if cat not in category_episodes:
    #             category_episodes[cat] = []
    #         category_episodes[cat].append(ep)
    #     print(f"    DEBUG: {len(category_episodes)} categories found")
        
    #     # 2. 如果类别数量不够，返回None
    #     if len(category_episodes) < num_objects:
    #         print(f"    DEBUG: Not enough categories")
    #         return None
        
    #     # 3. 随机选择num_objects个不同类别
    #     selected_categories = random.sample(list(category_episodes.keys()), num_objects)
    #     print(f"    DEBUG: Selected categories: {selected_categories}")
        
    #     # 4. 为每个类别选择一个episode
    #     selected_episodes = []
    #     for cat in selected_categories:
    #         valid_eps = category_episodes[cat]
            
    #         print(f"    DEBUG: {cat} has {len(valid_eps)} valid episodes")
            
    #         if not valid_eps:
    #             print(f"    DEBUG: No valid episodes for {cat}, returning None")
    #             return None
            
    #         ep = random.choice(valid_eps)
    #         selected_episodes.append(ep)
        
    #     print(f"    DEBUG: Successfully created episode!")
        
    #     # 5. 选择起始位置（使用第一个episode的起始位置）
    #     base_episode = selected_episodes[0]
        
    #     # ✅ 6. 从 goals_by_category 提取真实位置信息
    #     # 提取场景哈希值（从 scene_id 中获取）
    #     # 例如 "hm3d_v0.2/val/00877-4ok3usBNeis/4ok3usBNeis.basis.glb" 
    #     # -> scene_hash = "4ok3usBNeis"
    #     scene_parts = scene_id.split('/')
    #     scene_hash = scene_parts[-2].split('-')[1] if len(scene_parts) >= 2 else None
    #     scene_name = scene_parts[-1].replace('.basis.glb', '') if len(scene_parts) >= 1 else None
        
    #     # ✅ 调试输出
    #     print(f"    DEBUG: scene_id = {scene_id}")
    #     print(f"    DEBUG: scene_hash = {scene_hash}")
    #     print(f"    DEBUG: scene_name = {scene_name}")
    #     print(f"    DEBUG: available scene hashes in goals_by_category: {list(self.goals_by_category.keys())[:3]}")
        
    #     goals_list = []
    #     for ep in selected_episodes:
    #         obj_cat = ep["object_category"]
    #         obj_id = ep["info"].get("closest_goal_object_id", None)
            
    #         # 查找该物体的位置信息
    #         position = None
    #         room_id = None
    #         room_name = None
            
    #         # ✅ 从 goals_by_category 中查找
    #         if scene_hash in self.goals_by_category:
    #             goals_dict = self.goals_by_category[scene_hash]
    #             # ✅ 修复：键名格式应该是 "scene_name.basis.glb_category"
    #             obj_key = f"{scene_name}.basis.glb_{obj_cat}"
                
    #             print(f"    DEBUG: Looking for obj_key = {obj_key}")
    #             print(f"    DEBUG: Available keys in goals_dict: {list(goals_dict.keys())[:5]}")
                
    #             if obj_key in goals_dict:
    #                 objects_list = goals_dict[obj_key]
    #                 print(f"    DEBUG: Found {len(objects_list)} objects for {obj_key}")
                    
    #                 # 查找匹配的 object_id
    #                 for obj in objects_list:
    #                     if obj_id is not None and obj.get('id') == obj_id:
    #                         position = obj.get('position')
    #                         room_id = obj.get('room_id')
    #                         room_name = obj.get('room_name')
    #                         print(f"    DEBUG: Found matching object with id={obj_id}, position={position}")
    #                         break
                    
    #                 # 如果没找到特定 ID，就用第一个
    #                 if position is None and len(objects_list) > 0:
    #                     position = objects_list[0].get('position')
    #                     room_id = objects_list[0].get('room_id')
    #                     room_name = objects_list[0].get('room_name')
    #                     print(f"    DEBUG: Using first object, position={position}")
    #             else:
    #                 print(f"    WARNING: obj_key '{obj_key}' not found in goals_dict!")
    #         else:
    #             print(f"    WARNING: scene_hash '{scene_hash}' not found in goals_by_category!")
            
    #         goals_list.append({
    #             "object_id": obj_id,
    #             "object_category": obj_cat,
    #             "position": position,  # ✅ 真实位置（可能仍为 None）
    #             "room_id": room_id,
    #             "room_name": room_name
    #         })
        
    #     # 7. 创建多目标episode
    #     multi_episode = {
    #         "episode_id": f"multi_{scene_id}_{random.randint(1000, 9999)}",
    #         "scene_id": scene_id,
    #         "start_position": base_episode["start_position"],
    #         "start_rotation": base_episode["start_rotation"],
            
    #         # 多目标信息
    #         "object_categories": [ep["object_category"] for ep in selected_episodes],
    #         "goals": goals_list,  # ✅ 使用提取的真实位置
            
    #         # 元信息
    #         "info": {
    #             "difficulty": self._estimate_difficulty(selected_episodes),
    #             "geodesic_distance": sum(ep.get("info", {}).get("geodesic_distance", 0) 
    #                                     for ep in selected_episodes),
    #             "num_objects": num_objects,
    #             "source_episodes": [ep["episode_id"] for ep in selected_episodes]
    #         }
    #     }
        
    #     return multi_episode
    def create_multi_object_episode(self, scene_id, episodes, num_objects=3):
        """
        创建一个多目标episode，保证目标都在起点附近
        """
        # 1. 提前解析 scene_hash (因为后面算距离需要用到)
        scene_parts = scene_id.split('/')
        scene_hash = scene_parts[-2].split('-')[1] if len(scene_parts) >= 2 else None
        scene_name = scene_parts[-1].replace('.basis.glb', '') if len(scene_parts) >= 1 else None

        # 2. 随机选择一个 episode 作为真正的“出生点” (Base)
        base_episode = random.choice(episodes)
        start_pos = base_episode['start_position']

        # 3. 🔴 核心修复：只获取该出生点方圆 15 米内、且在同一楼层的物体类别
        available_categories = self._get_nearby_categories(scene_hash, start_pos)
        
        # 4. 检查起点附近的类别是否足够拼凑一个多目标任务
        if len(available_categories) < num_objects:
            print(f"    DEBUG: Not enough nearby categories for start pos {start_pos} (found {len(available_categories)}), returning None")
            return None
            
        # 5. 从附近的类别中随机抽取 num_objects 个作为最终目标
        selected_categories = random.sample(available_categories, num_objects)
        print(f"    DEBUG: Selected nearby categories: {selected_categories}")

        # 6. 为选中的类别，去原始 episodes 列表里随便找一个对应的 episode 壳子（为了继承一些元数据）
        selected_episodes = []
        for cat in selected_categories:
            valid_eps = [ep for ep in episodes if ep['object_category'] == cat]
            if not valid_eps:
                print(f"    DEBUG: Missing valid episode for category {cat}, returning None")
                return None
            ep = random.choice(valid_eps)
            selected_episodes.append(ep)

        # 7. 从 goals_by_category 提取这些目标的真实位置信息
        goals_list = []
        for ep in selected_episodes:
            obj_cat = ep["object_category"]
            obj_id = ep["info"].get("closest_goal_object_id", None)
            
            position = None
            room_id = None
            room_name = None
            
            if scene_hash in self.goals_by_category:
                goals_dict = self.goals_by_category[scene_hash]
                obj_key = f"{scene_name}.basis.glb_{obj_cat}"
                
                if obj_key in goals_dict:
                    objects_list = goals_dict[obj_key]
                    
                    # 尝试精确匹配 object_id
                    for obj in objects_list:
                        if obj_id is not None and obj.get('id') == obj_id:
                            position = obj.get('position')
                            room_id = obj.get('room_id')
                            room_name = obj.get('room_name')
                            break
                    
                    # 匹配不到 ID 就默认取最近的一个（这里取第一个做简化）
                    if position is None and len(objects_list) > 0:
                        position = objects_list[0].get('position')
                        room_id = objects_list[0].get('room_id')
                        room_name = objects_list[0].get('room_name')
            
            goals_list.append({
                "object_id": obj_id,
                "object_category": obj_cat,
                "position": position,
                "room_id": room_id,
                "room_name": room_name
            })
        
        # 8. 组装最终的多目标 episode
        multi_episode = {
            "episode_id": f"multi_{scene_id}_{random.randint(1000, 9999)}",
            "scene_id": scene_id,
            "start_position": start_pos, # ✅ 严格使用筛选附近的那个出生点
            "start_rotation": base_episode["start_rotation"],
            
            "object_categories": selected_categories,
            "goals": goals_list,
            
            "info": {
                "difficulty": self._estimate_difficulty(selected_episodes),
                "geodesic_distance": sum(ep.get("info", {}).get("geodesic_distance", 0) for ep in selected_episodes),
                "num_objects": num_objects,
                "source_episodes": [ep["episode_id"] for ep in selected_episodes]
            }
        }
        
        return multi_episode
    
    def _estimate_difficulty(self, episodes):
        """估算难度等级"""
        # 简单规则：基于geodesic distance的总和
        total_distance = sum(ep.get("info", {}).get("geodesic_distance", 5.0) 
                           for ep in episodes)
        
        if total_distance < 15:
            return "easy"
        elif total_distance < 30:
            return "medium"
        else:
            return "hard"
    
    def generate_multi_object_dataset(self, episodes_per_scene=5, num_objects=3):
        """
        生成完整的多目标数据集
        
        Args:
            episodes_per_scene: 每个场景生成多少个多目标episodes
            num_objects: 每个episode包含多少个目标物体
        """
        print("\n🔧 Starting multi-object dataset generation...")
        
        # 1. 加载原始数据
        original_data = self.load_original_dataset()
        
        # 2. 按场景分组
        scene_episodes = self.group_episodes_by_scene(original_data)
        
        # 3. 生成多目标episodes
        multi_episodes = []
        
        for scene_id, episodes in scene_episodes.items():
            print(f"\n📍 Processing scene: {scene_id}")
            
            # 检查该场景是否有足够的物体类别
            categories = set(ep['object_category'] for ep in episodes)
            print(f"  📊 Categories: {len(categories)} - {categories}")  # ← 添加这行
            if len(categories) < num_objects:
                print(f"  ⚠️  Skip: Only {len(categories)} categories (need {num_objects})")
                continue
            
            # 为该场景生成多个多目标episodes
            scene_multi_eps = 0
            attempts = 0
            max_attempts = episodes_per_scene * 3
            
            while scene_multi_eps < episodes_per_scene and attempts < max_attempts:
                attempts += 1
                
                multi_ep = self.create_multi_object_episode(
                    scene_id, episodes, num_objects
                )
                
                if multi_ep is not None:
                    multi_episodes.append(multi_ep)
                    scene_multi_eps += 1
                    print(f"  ✅ Generated {scene_multi_eps}/{episodes_per_scene}: "
                          f"{multi_ep['object_categories']} "
                          f"({multi_ep['info']['difficulty']})")
        
        print(f"\n✅ Total generated: {len(multi_episodes)} multi-object episodes")
        
        # 4. 统计信息
        self._print_statistics(multi_episodes)
        
        return multi_episodes
    
    def _print_statistics(self, episodes):
        """打印数据集统计信息"""
        print("\n📊 Dataset Statistics:")
        print(f"  Total episodes: {len(episodes)}")
        
        # 难度分布
        difficulties = [ep['info']['difficulty'] for ep in episodes]
        print(f"  Difficulty distribution:")
        print(f"    Easy: {difficulties.count('easy')}")
        print(f"    Medium: {difficulties.count('medium')}")
        print(f"    Hard: {difficulties.count('hard')}")
        
        # 物体类别统计
        all_categories = []
        for ep in episodes:
            all_categories.extend(ep['object_categories'])
        unique_categories = set(all_categories)
        print(f"  Unique object categories: {len(unique_categories)}")
        print(f"    {', '.join(sorted(unique_categories))}")
    
    def save_dataset(self, episodes, output_path=None):
        """保存数据集"""
        if output_path is None:
            output_path = self.output_path
        
        # 创建输出目录
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 构造完整数据集格式
        dataset = {
            "episodes": episodes
        }
        
        # 保存为gzip压缩的JSON
        print(f"\n💾 Saving dataset to: {output_path}")
        with gzip.open(output_path, 'wt') as f:
            json.dump(dataset, f, indent=2)
        
        print(f"✅ Saved successfully!")
        
        # 同时保存一个未压缩版本便于查看
        json_path = output_path.replace('.json.gz', '.json')
        with open(json_path, 'w') as f:
            json.dump(dataset, f, indent=2)
        print(f"📄 Also saved uncompressed version: {json_path}")


def main():
    """主函数：生成示例数据集"""
    
    # 配置路径
    DATASET_ROOT = "/home/hoho/WM_ws/WMNavigation-master/data"
    
    # 原始单目标数据集
    original_dataset = f"{DATASET_ROOT}/objectnav_hm3d_v2/val/val.json.gz"
    
    # 输出的多目标数据集
    output_dataset = f"{DATASET_ROOT}/multi_objectnav_hm3d/val/val.json.gz"
    
    # 创建生成器
    generator = MultiObjectEpisodeGenerator(
        original_dataset_path=original_dataset,
        output_path=output_dataset
    )
    
    # 生成数据集
    # episodes_per_scene=2: 每个场景只生成2个episodes（快速测试）
    # num_objects=3: 每个episode包含3个目标物体
    multi_episodes = generator.generate_multi_object_dataset(
        episodes_per_scene=2,  # 每个场景生成多少个多目标 episode
        num_objects=2
    )
    
    # 保存数据集
    generator.save_dataset(multi_episodes)
    
    print("\n" + "="*60)
    print("🎉 Multi-object dataset generation completed!")
    print("="*60)
    
    # 打印示例episode
    if len(multi_episodes) > 0:
        print("\n📋 Example episode:")
        example = multi_episodes[0]
        print(json.dumps(example, indent=2))


if __name__ == "__main__":
    main()