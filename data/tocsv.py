import os
import numpy as np
import pandas as pd
from tqdm import tqdm 

input_path = "data/shenzhen/traj_shenzhen_min_one_by_one"  # 替换为实际输入路径
output_csv_path = "data/data/shenzhen/simulation/traj_shenzhen_simulation.csv"  # 替换为实际输出路径

# 获取所有.npy文件的路径
npy_files = [f for f in os.listdir(input_path) if f.endswith('.npy')]

combined_strings = []

for file in tqdm(npy_files):
    file_path = os.path.join(input_path, file)
    # 加载numpy数组（假设每个文件存储一个一维数组）
    arr = np.load(file_path)
    # 将数组元素转为字符串并用"-"连接
    str_values = '-'.join(map(str, arr))
    combined_strings.append(str_values)

# 将字符串列表转换为DataFrame（每行一个字符串）
df = pd.DataFrame({"Points": combined_strings})
# 保存到CSV，不保留索引
df.to_csv(output_csv_path, index=False)