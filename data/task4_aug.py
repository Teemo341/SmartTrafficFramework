import numpy as np
from tqdm import tqdm

#  trajs_edge = read_traj('data/simulation/trajectories_10*10_repeat.csv')
trajs_edge = np.load('data/simulation/new_task4_data_add.npy')


print(trajs_edge.shape)
print(trajs_edge[0])
# mask = trajs_edge >= 0

# random_add = np.random.randint(1, 11, size=trajs_edge.shape[0] * trajs_edge.shape[1] * trajs_edge.shape[2]* trajs_edge.shape[3])
# random_add = random_add.reshape(mask.shape)
# # 修改数据
# print(mask.shape)
# print(random_add.shape)
# # print(random_add[0:100])
# trajs_edge = trajs_edge +  random_add * mask
# print(trajs_edge[0])
# # new_task4 = np.zeros_like(trajs_edge, dtype=int)
# # for i in tqdm(range(trajs_edge.shape[0]), desc='Processing trajectories'):
# #     idx = np.arange(trajs_edge.shape[0])
# #     choice = np.random.choice(idx, int(0.3*trajs_edge.shape[0]), replace=False)
# #     new_task4[i] = np.sum(trajs_edge[choice], axis=0)
# # print(new_task4.shape)
# # trajs_edge[trajs_edge < 0] = -1
# np.save('data/simulation/new_task4_data_add.npy', trajs_edge)
