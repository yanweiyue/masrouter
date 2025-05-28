import os
import json
import random
import shutil
import math

# 设置路径
test_dir = '/Users/lby/Desktop/master/code/MAR/Datasets/MATH/test'
output_dir = '/Users/lby/Desktop/master/code/MAR/Datasets/MATH/sampled_test'

random.seed(42)  # 固定随机种子以确保可重复性

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 遍历每个类别目录
for category in os.listdir(test_dir):
    category_path = os.path.join(test_dir, category)
    if not os.path.isdir(category_path):
        continue  # 跳过非目录文件
    
    level_groups = {}  # 按level分组存储问题文件信息
    
    # 遍历该类别下的所有JSON文件
    for problem_file in os.listdir(category_path):
        if not problem_file.endswith('.json'):
            continue  # 仅处理JSON文件
        
        file_path = os.path.join(category_path, problem_file)
        
        # 读取问题数据
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                problem_data = json.load(f)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
        
        level = problem_data.get('level', 'Unknown')
        
        if level not in level_groups:
            level_groups[level] = []
        level_groups[level].append((problem_file, file_path))
    
    # 对每个level进行抽样并收集结果
    sampled_files = []
    for level, files in level_groups.items():
        total = len(files)
        sample_size = max(1, math.ceil(total * 0.1))  # 向上取整，至少1个
        
        # 随机抽样
        try:
            sampled = random.sample(files, sample_size)
        except ValueError as e:
            print(f"Error sampling {level} in {category}: {e}")
            continue
        
        sampled_files.extend(sampled)
    
    # 创建输出目录并复制文件
    output_category_dir = os.path.join(output_dir, category)
    os.makedirs(output_category_dir, exist_ok=True)
    
    for problem_file, src_path in sampled_files:
        dest_path = os.path.join(output_category_dir, problem_file)
        shutil.copy2(src_path, dest_path)  # 保留元数据

print("抽样完成！结果保存在:", output_dir)