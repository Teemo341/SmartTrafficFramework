import os 
import numpy as np

dir_path  = 'data/simulation/new_task4_data_one_by_one'
data_path = 'data/simulation/new_task4_data.npy'

data = np.load(data_path, allow_pickle=True)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
for i in range(data.shape[0]):
    file_path = os.path.join(dir_path, f'{i}.npy')
    np.save(file_path, data[i])
print(f"Data saved to {dir_path} successfully.")