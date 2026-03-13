"""
快速测试多目标数据集
验证生成的数据集是否正确
"""

import json
import gzip
import sys

def test_multi_object_dataset(dataset_path):
    """测试数据集格式"""
    
    print("🔍 Testing multi-object dataset...")
    print(f"📂 Path: {dataset_path}\n")
    
    # 1. 加载数据集
    try:
        with gzip.open(dataset_path, 'rt') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("❌ Error: Dataset file not found!")
        print("   Please run generate_multi_object_dataset.py first")
        return False
    
    episodes = data['episodes']
    print(f"✅ Loaded {len(episodes)} episodes\n")
    
    # 2. 验证格式
    print("🔍 Validating format...")
    
    for i, ep in enumerate(episodes[:3]):  # 只检查前3个
        print(f"\n📋 Episode {i+1}:")
        print(f"  ID: {ep['episode_id']}")
        print(f"  Scene: {ep['scene_id']}")
        print(f"  Num objects: {ep['info']['num_objects']}")
        print(f"  Categories: {ep['object_categories']}")
        print(f"  Difficulty: {ep['info']['difficulty']}")
        print(f"  Total distance: {ep['info']['geodesic_distance']:.2f}m")
        
        # 验证必需字段
        required_fields = ['episode_id', 'scene_id', 'start_position', 
                          'start_rotation', 'object_categories', 'goals']
        
        for field in required_fields:
            if field not in ep:
                print(f"  ❌ Missing field: {field}")
                return False
        
        # 验证goals数量
        if len(ep['goals']) != len(ep['object_categories']):
            print(f"  ❌ Goals count mismatch!")
            return False
        
        print(f"  ✅ Format valid")
    
    # 3. 统计信息
    print("\n" + "="*60)
    print("📊 Dataset Statistics:")
    print("="*60)
    
    difficulties = [ep['info']['difficulty'] for ep in episodes]
    print(f"Difficulty distribution:")
    print(f"  Easy: {difficulties.count('easy')} ({difficulties.count('easy')/len(episodes)*100:.1f}%)")
    print(f"  Medium: {difficulties.count('medium')} ({difficulties.count('medium')/len(episodes)*100:.1f}%)")
    print(f"  Hard: {difficulties.count('hard')} ({difficulties.count('hard')/len(episodes)*100:.1f}%)")
    
    all_categories = []
    for ep in episodes:
        all_categories.extend(ep['object_categories'])
    
    from collections import Counter
    category_counts = Counter(all_categories)
    
    print(f"\nObject category distribution:")
    for cat, count in category_counts.most_common(10):
        print(f"  {cat}: {count}")
    
    print("\n✅ All tests passed!")
    print("🎉 Dataset is ready to use!\n")
    
    return True


if __name__ == "__main__":
    dataset_path = "/home/hoho/WM_ws/WMNavigation-master/data/multi_objectnav_hm3d/val/val.json.gz"
    
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    
    test_multi_object_dataset(dataset_path)
