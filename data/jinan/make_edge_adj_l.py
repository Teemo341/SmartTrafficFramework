import numpy as np
import torch

def adj_m2adj_l(adj_matrix,max_connection):
    #adj_matrix: 0_index
    #jinan:max_connection=10
    n = len(adj_matrix)
    adj_list = torch.zeros([n,max_connection,2],dtype=torch.int)
    for i in range(n):
        adj_nodes = np.nonzero(adj_matrix[i])[0]
 
        if len(adj_nodes) > max_connection:
            print(len(adj_nodes))
            raise ValueError('Error: Max connection is wrong')
 
        for j in range(len(adj_nodes)):
  
            adj_list[i,j,0] = int(adj_nodes[j]+1)
            adj_list[i,j,1] = adj_matrix[i][adj_nodes[j]]
    
    return adj_list


edge_node_map_dict_path = 'data/jinan/edge_node_map_dict.pkl'
node_edge_map_dict_path = 'data/jinan/node_edge_map_dict.pkl'
adjcent = 'data/jinan/adj_l.npy'

edge_node_map_dict = np.load(edge_node_map_dict_path, allow_pickle=True)
node_edge_map_dict = np.load(node_edge_map_dict_path, allow_pickle=True)
adjcent = np.load(adjcent, allow_pickle=True)
# adj_l = adj_m2adj_l(adjcent, max_connection=10)
# adj_l =  adj_l[:,:,0]
print(adjcent.shape)
adj_l = adjcent
# np.save('data/jinan/adj_l.npy', adj_l)
# print(edge_node_map_dict)
idx = [8709, 1290, 8709, 3477, 8712, 1597, 3485, 8791, 8664]
print(adj_l[1575])
print(adj_l[7974])
# # idx = [9143,9142,9143,9142,9143,9144,9146,9147]
# edge_adj_l=np.load('data/jinan/edge_adj_l.npy')
# print(edge_adj_l[2172])
# for i in range(len(idx)):
#     print(adj_l[idx[i]-1])
# edge_adj_l = []
# # # print(node_edge_map_dict.keys())

# for key in range(0,23313):
#     l = []
    
#     if key == 0:
#         edge_adj_l.append([0])
#         continue
    
#     key = str(key)
#     node = edge_node_map_dict[key]
#     # print(node)
#     # print(node)
#     # node1 = adj_l[node[0]].tolist()
#     # print(node1)
#     node2 = adj_l[node[1]].tolist()
#     # print(node2)
#     # for i in range(len(node1)):
#     #     if node1[i] == 0:
#     #         break
#     #     if node1[i] not in node:
#     #         node_key = str(node[0])+'_'+str(node1[i]-1)
#     #         if node_edge_map_dict[node_key] != int(key):
#     #             l.append(node_edge_map_dict[node_key])
#     for i in range(len(node2)):
#         if node2[i] == 0:
#             break
#         if node2[i] not in node:
#             node_key = str(node[1])+'_'+str(node2[i]-1)
#             # print(node_key)
#             if node_edge_map_dict[node_key] != int(key):
#                 l.append(node_edge_map_dict[node_key])
#     edge_adj_l.append(l)
#     # print(edge_adj_l)
#     # break
# original_list = edge_adj_l
# max_length = max(len(sub_list) for sub_list in original_list)
# padded_list = [sub_list + [0] * (max_length - len(sub_list)) for sub_list in original_list]
# np_array = np.array(padded_list, dtype=np.int32)


# print(np_array)
# print(np_array.shape)
# np.save('data/jinan/edge_adj_l.npy', np_array)