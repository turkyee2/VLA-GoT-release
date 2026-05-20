import json
import os

# 定义输入和输出文件名
input_json_file = '../processed_data/convs/libero_10_his_1_val_ind_third_view_wrist_a2i_512.json'  # <-- 将这里替换成你的JSON文件名
output_json_file = '10_val_ind_trajectory_paths.json'

# 使用集合(set)来存储唯一的路径，可以自动处理重复项
unique_traj_paths = set()

print(f"正在从 {input_json_file} 读取数据...")

try:
    # 读取JSON文件
    with open(input_json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 遍历JSON中的每一个字典对象
    for item in data:
        # 将 'image' 和 'action' 列表中的所有路径合并到一个列表中
        # 使用 .get(key, []) 避免当某个key不存在时报错
        all_paths = item.get('image', []) + item.get('action', [])
        
        # 遍历所有文件路径
        for file_path in all_paths:
            # 检查路径是否为空或无效
            if not file_path or not isinstance(file_path, str):
                continue
            
            # 提取上上级目录，即trj路径
            # os.path.dirname('/.../trj_5/imgs_third_view/image_0.png') -> '/.../trj_5/imgs_third_view'
            # os.path.dirname('/.../trj_5/imgs_third_view')            -> '/.../trj_5'
            traj_path = os.path.dirname(os.path.dirname(file_path))
            
            # 将提取到的路径添加到集合中
            unique_traj_paths.add(traj_path)

    # 将集合转换为列表，并进行排序以便观察
    sorted_paths_list = sorted(list(unique_traj_paths))

    print(f"处理完成！共找到 {len(sorted_paths_list)} 条不同的 'trj' 路径。")

    # 将结果列表保存到新的JSON文件中
    with open(output_json_file, 'w', encoding='utf-8') as f:
        # indent=4 让输出的JSON文件格式更美观，易于阅读
        json.dump(sorted_paths_list, f, indent=4)
        
    print(f"所有唯一的 'trj' 路径已保存到文件: {output_json_file}")

except FileNotFoundError:
    print(f"错误: 输入文件 '{input_json_file}' 未找到。")
except json.JSONDecodeError:
    print(f"错误: 文件 '{input_json_file}' 不是有效的JSON格式。")
except Exception as e:
    print(f"发生未知错误: {e}")

