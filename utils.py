from unicodedata import name
import numpy as np
from typing import List
import pandas as pd
import torch
from tqdm import tqdm
import os
from itertools import groupby

def adj_m2adj_l(adj_matrix:np.ndarray,max_connection:int=10)->torch.Tensor:
    #jinan:max_connection=10
    n = len(adj_matrix)
    adj_list = torch.zeros([n,max_connection,2],dtype=torch.float)
  
    for i in range(n):
        adj_nodes = np.nonzero(adj_matrix[i])[0]
 
        if len(adj_nodes) > max_connection:
            print(len(adj_nodes))
            raise ValueError('Error: Max connection is wrong')
 
        for j in range(len(adj_nodes)):
  
            adj_list[i,j,0] = adj_nodes[j]
            adj_list[i,j,1] = adj_matrix[i][adj_nodes[j]]
    
    return adj_list

def edge2node(map_,lst):
    # map: np.array([[edge,o,d...],...])
    nodes = []
    for edge in lst:
        print(map_[map_[:,0]==edge])
        nodes.append(int(map_[map_[:,0]==edge][0][1]))
        nodes.append(int(map_[map_[:,0]==edge][0][2]))
    nodes_lst = []
    if nodes[0] not in nodes[1:]:
        nodes_lst.append(nodes[0])
        nodes_lst.append(nodes[1])
    else:
        nodes_lst.append(nodes[1])
        nodes_lst.append(nodes[0])
    nodes = nodes[2:]
    for i in range(0,len(nodes)//2):
        if nodes[2*i] == nodes_lst[-1]:
            nodes_lst.append(nodes[2*i+1])
        else:
            nodes_lst.append(nodes[2*i])
    return nodes_lst

def jinan2adj(file_path='data/jinan/edge_node_jinan.csv')->np.ndarray:    
    
    u_nodes = []
    v_nodes = []
    lengths = []
    data = pd.read_csv(file_path)
    
    for i in range(len(data)):
        u_nodes.append(data.loc[i, 'Origin'])
        v_nodes.append(data.loc[i, 'Destination'])
        lengths.append(data.loc[i, 'Length'])

    num_of_nodes = np.max((np.max(u_nodes), np.max(v_nodes)))
    adj_matrix = np.zeros((num_of_nodes + 1, num_of_nodes + 1))
    for u, v, length in zip(u_nodes, v_nodes, lengths):
        adj_matrix[u][v] = length
        adj_matrix[v][u] = length
    
    return adj_matrix

def jinan_read_node_type(file_path=''):
    #TODO
    return

def jinan_read_traj(file_path='data/jinan/traj_jinan.csv'):
    #columns:['VehicleID', 'TripID', 'Points', 'DepartureTime', 'Duration', 'Length']
    data = pd.read_csv(file_path)
    data = [list(traj.split("_")) for traj in data['Points']]
    trajs=[]
    time = []
    for i in tqdm(range(len(data)),desc='Reading Traj'):
        trajs.append([int(x.split('-')[0]) for x in data[i]])
        time.append([float(x.split('-')[1]) for x in data[i]])
    
    return trajs,time

def jinan_trajs_repeat(trajs:List[List[int]],time:List[List[float]])->List[List[int]]:
    #repeat the traj every 5s
    for i in tqdm(range(len(trajs)),desc='Repeating Traj'):
        traj=[[trajs[i][j]]*int(((time[i][j+1]-time[i][j])//5+1)) for j in range(len(trajs[i])-1)] +[[trajs[i][-1]]]
        trajs[i] = [item for sublist in traj for item in sublist]
    return trajs

def node2edge(map_,lst):
    # map: Dict{node_o,node_d:edge}
    nodes = [key for key, _ in groupby(lst)]
    counts= [sum(1 for _ in group) for _, group in groupby(lst)]
    counts = counts[:-1]
    edge = []
    for i in range(len(nodes)-1):
        key=str(nodes[i])+'_'+str(nodes[i+1])
        edge.append(map_[key])
    result = [item for item, count in zip(edge, counts) for _ in range(count)]
    return result
        

    edges = []
    e_l = []
    repeats = []
    nodes = [key for key, _ in groupby(lst)]
    for i in range(len(nodes)-1):
        bool_index = [list([int(x) for x in maps])==[nodes[i],nodes[i+1]] for maps in map_[:,1:3]]
        bool_index_r = [list([int(x) for x in maps])==[nodes[i+1],nodes[i]] for maps in map_[:,1:3]]
        try:
            e_l.append(int(map_[bool_index][0][0]))
        except:
            e_l.append(int(map_[bool_index_r][0][0]))
    lst = lst[:-1]
    for _, group in groupby(lst):
        repeats.append(sum(1 for _ in group))
    for i in range(len(e_l)):
        edges.append([e_l[i]]*repeats[i])
    edges = [item for sublist in edges for item in sublist]
    return edges

def padding_zero(traj:List,max_len:int)->np.ndarray:
    if len(traj) >= max_len:
        return np.array(traj[:max_len],dtype=np.int64)
    return np.concatenate([traj,np.zeros((max_len-len(traj),),dtype=np.int64)],axis=0)

def remove_consecutive_duplicates(lst:List)->List:
    return [lst[i] for i in range(len(lst)) if i == 0 or lst[i] != lst[i - 1]]



if __name__ == '__main__':
    #save the traj in data/jinan/traj_repeat_one_by_one,name:1.npy,2.npy,...
    # trajs,time= jinan_read_traj()
    # trajs = jinan_trajs_repeat(trajs,time)
    # max_len = max([len(traj) for traj in trajs])
    # print(max_len)
    # print(trajs[0])
    # path = 'data/jinan/traj_repeat_one_by_one/'
    # os.makedirs(path, exist_ok=True)
    # for i in range(len(trajs)):
    #     np.save('data/jinan/traj_repeat_one_by_one/'+str(i+1)+'.npy',trajs[i],allow_pickle=True)
    # data = jinan2adj()
    # np.save('data/jinan/adjcent.npy',data)
    # path = 'data/jinan/node_traj_repeat_one_by_one/'
    # edgepath='data/jinan/edge_traj_repeat_one_by_one/'
    # edgefiles = os.listdir(edgepath)
    # files = os.listdir(path)
    map_ = pd.read_csv('data/jinan/edge_node_jinan.csv')
    map_ = np.array(map_[['EdgeID','Origin','Destination']])
    map_dict = {}
    # for i in tqdm(range(len(map_)),desc='Building Map'):
    #     map_dict[str(map_[i][1])+'_'+str(map_[i][2])] = map_[i][0]
    #     map_dict[str(map_[i][2])+'_'+str(map_[i][1])] = map_[i][0]

    # for file in tqdm(files,desc='Processing'):
    #     name_='data/jinan/edge_traj_repeat_one_by_one/'+file
    #     if name_ in edgefiles:
    #         continue
    #     traj = np.load(path+file)
    #     #print(traj)
    #     edges = node2edge(map_dict,traj)
    #     #print(edges)
    #     np.save(name_,edges)
    #     # break  

    #['trunk_link','residential','secondary','trunk','tertiary','secondary_link','motorway_link','primary','living_street','unclassified','motorway','primary_link','tertiary_link']
    # size_order = {
    # 'living_street': 1,
    # 'unclassified': 2,
    # 'residential': 3,
    # 'tertiary_link': 4,
    # 'tertiary': 5,
    # 'secondary_link': 6,
    # 'secondary': 7,
    # 'primary_link': 8,
    # 'primary': 9,
    # 'trunk_link': 10,
    # 'trunk': 11,
    # 'motorway_link': 12,
    # 'motorway': 13
    # }
    # map_ = pd.read_csv('data/jinan/edge_node_jinan.csv')
    # map_ = np.array(map_[['Class','Origin','Destination']])
    # map_dict = {}
    # for i in tqdm(range(len(map_)),desc='Building Map'):
    #     map_dict[str(map_[i][1])+'_'+str(map_[i][2])] = size_order[map_[i][0]]
    #     map_dict[str(map_[i][2])+'_'+str(map_[i][1])] = size_order[map_[i][0]]
    # path = 'data/jinan/adjcent.npy'
    # data = np.load(path)
    # for i in range(len(data)):
    #     for j in range(i,len(data)):
    #         if data[i][j] != 0:
    #             data[i][j] = int(map_dict[str(i)+'_'+str(j)])
    #             data[j][i] = int(map_dict[str(i)+'_'+str(j)])
    
    # np.save('data/jinan/adjcent_class.npy',data)