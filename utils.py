from unicodedata import name
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import os
from itertools import groupby
import pickle
import networkx as nx
from typing import List
import random
import math

def adj_m2adj_l(adj_matrix:np.ndarray,max_connection:int=10)->torch.Tensor:
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

def calculate_bounds(pos):
    x_values = [node[0] for node in pos.values()]
    y_values = [node[1] for node in pos.values()]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    # x_min, x_max = x_min - 0.1 * abs(x_max - x_min), x_max + 0.1 * abs(x_max - x_min)
    # y_min, y_max = y_min - 0.1 * abs(y_max - y_min), y_max + 0.1 * abs(y_max - y_min)

    return x_min, x_max, y_min, y_max

def calculate_load(traj_probability, method = 'map'):
    # traj_probability: B, T, V

    if method == 'map':
        max_indices = np.argmax(traj_probability, axis=-1)
        one_hot = np.zeros_like(traj_probability)
        B, T, V = traj_probability.shape
        one_hot[np.arange(B)[:, None], np.arange(T), max_indices] = 1
        load = np.sum(one_hot, axis = 0) # T, V
        load = load[:,1:] # remove the special token
    elif method == 'single_prob':
        max_indices = np.argmax(traj_probability, axis=-1)
        one_hot = np.zeros_like(traj_probability)
        B, T, V = traj_probability.shape
        one_hot[np.arange(B)[:, None], np.arange(T), max_indices] = 1
        filtered = traj_probability*one_hot
        load = np.sum(filtered, axis = 0)
        load = load[:,1:]
    elif method == 'all_prob':
        traj_probability = traj_probability[:,:,1:] # remove the special token
        load = np.sum(traj_probability, axis = 0) # T, V-1
    else:
        raise ValueError(f"method should be 'map', 'single_prob' or 'all_prob', but got {method}")
    return load

def preprocess_traj(traj_dir):
    trajs = pd.read_csv(traj_dir)
    trajs = [
            {"VehicleID": vid,
             "TripID": tid,
            "Points": [[int(p.split("-")[0]), float(p.split("-")[1])] for p in ps.split("_")],
            "DepartureTime": dt,
            "Duration": dr
        } for _, (vid, tid, ps, dt, dr, l) in tqdm(trajs.iterrows(), desc='Loading trajectories')
    ]
    trajs_ = {}
    for traj in tqdm(trajs, desc='Processing trajectories'):
        if traj["VehicleID"] not in trajs_:
            trajs_[traj["VehicleID"]] = []
        trajs_[traj["VehicleID"]].append(traj)
    return trajs_ 

def decide_order(num:int,adj:List[int],map:dict)->List[int]:
    if num == 3:
        la_adj = [map[int(x)][0] for x in adj]
        lo_adj = [map[int(x)][1] for x in adj]
        index1 = np.argmin(la_adj)
        lo_adj[index1] = 1000
        index2 = np.argmin(lo_adj)
        ans = [1,1,1]
        ans[index1] = 2
        ans[index2] = 3
        #print(la_adj,lo_adj)
        #print(index1,index2)
        return {x:y for x,y in zip(adj,ans)}
    if num == 4:
        la_adj = [map[int(x)][0] for x in adj]
        lo_adj = [map[int(x)][1] for x in adj]
        print(la_adj,lo_adj)
        index1 = np.argmin(la_adj)
        index3 = np.argmax(la_adj)
        lo_adj[index1] = 1000
        lo_adj[index3] = 1000
        index2 = np.argmin(lo_adj)
        ans = [2,2,2,2]
        ans[index1] = 3
        ans[index2] = 1
        ans[index3] = 4
        #print(la_adj,lo_adj)
        #print(index1,index2,index3)
        return {x:y for x,y in zip(adj,ans)}
    else:
        raise ValueError('Error: The number of adj is wrong')
        

def generate_data(city = 'boston', total_trajectories = 5, max_length = 50, capacity_scale = 10, weight_quantization_scale = None, max_connection = 4):
    edges, pos = read_city(city) #! 0-indexing
    node_num = len(pos)
    edge_num = len(edges)

    all_encoded_trajectories = [] #! 1-indexing
    all_adj_list = [] #! 0-indexing
    edge_capacity = sample_capacity(capacity_scale,edge_num)
    adj_table = get_weighted_adj_table(edges, pos, edge_capacity, normalization = True, quantization_scale = weight_quantization_scale, max_connection = max_connection) #! 0-indexing
    G = transfer_graph(adj_table) #! 0-indexing
    OD = generate_OD(G, node_num) #! 1-indexing
    i = 0
    with tqdm(total=total_trajectories, desc="Processing trajectories") as pbar:

        while i < total_trajectories:
            edge_capacity = sample_capacity(capacity_scale,edge_num)
            adj_table = get_weighted_adj_table(edges, pos, edge_capacity, normalization = True, quantization_scale = weight_quantization_scale, max_connection = max_connection)
            G = transfer_graph(adj_table)
            trajectory = generate_trajectory_list(G, OD, max_length=max_length)

            if trajectory == None:
                # current OD is too long, generate a new OD and restart
                OD = generate_OD(G, node_num)
                all_encoded_trajectories = []
                all_adj_list = []
                i += 0
            else:
                all_encoded_trajectories.append(trajectory)
                all_adj_list.append(adj_table)
                i += 1
                pbar.update(1)

    # all_encoded_trajectories: [total_trajectories, max_length] #! 1-indexing
    # all_adj_list: [total_trajectories, node_num, max_connection, 2] #! 0-indexing
    return all_encoded_trajectories, all_adj_list

# generate OD pairs, return OD 1-indexing
#! 1-indexing
def generate_OD(G, node_num):
    OD = np.random.randint(1,node_num+1,2)
    while nx.has_path(G, OD[0]-1, OD[1]-1) == False or OD[0] == OD[1]:
        OD = np.random.randint(1,node_num+1,2)
    return OD

# get shortest traj, input one adj_table, one OD pair, return one trajectory 1-indexing
#! OD is 1-indexing, trajectory is 1-indexing
def generate_trajectory_list(G, OD, max_length = 50):
    trajectory = nx.shortest_path(G, (OD[0]-1), (OD[1]-1), weight='weight')
    if len(trajectory) > max_length:
        return None
    
    trajectory = [i+1 for i in trajectory]
    
    return trajectory

# get weighted adjacency table, return 0-indexing
def get_weighted_adj_table(edges, pos, capacity, normalization = True, quantization_scale = None, max_connection = 4):

    adj_table = np.zeros([len(pos),max_connection, 2]) # [node, connection, [target_node, weight]]

    # add edges to adj_table
    for i in range(len(edges)):
        if np.sum(adj_table[edges[i][0],:,0]!=0) >= max_connection: # already full
            raise ValueError('Error: max_connection is too small')
        elif adj_table[edges[i][0],np.sum(adj_table[edges[i][0],:,0]!=0),0] == max_connection: # duplicate edge
            raise ValueError('Error: duplicate edge')
        else:
            adj_table[edges[i][0],np.sum(adj_table[edges[i][0],:,0]!=0)] = [edges[i][1]+1,capacity[i]] # [target_node, weight], add to the first empty slot
            #! the adj_table[1][0][0] is the first connection of road 2,
            #! the adj_table[1][0][0] = 1 means that road 2 is connected to road 1
            #! the ajd_table[1][0][1] is the road length from road 2 to road 1
    
    if normalization:
        adj_table[:,:,1] = adj_table[:,:,1]/np.max(adj_table[:,:,1])
    if quantization_scale:
        adj_table[:,:,1] = np.ceil(adj_table[:,:,1]*quantization_scale)
        
    return adj_table #! 0-indexing

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

    #0_index
    return adj_matrix

def jinan_build_map(mode='edge2node'):
    if mode == 'node2edge':
        map_ = pd.read_csv('data/jinan/edge_node_jinan.csv')
        map_ = np.array(map_[['EdgeID','Origin','Destination']])
        map_dict = {}
        for i in tqdm(range(len(map_)),desc='Building Map'):
            map_dict[str(map_[i][1])+'_'+str(map_[i][2])] = map_[i][0]
        return map_dict
    if mode == 'edge2node':
        map_ = pd.read_csv('data/jinan/edge_node_jinan.csv')
        map_ = np.array(map_[['EdgeID','Origin','Destination']])
        map_dict = {}
        for i in tqdm(range(len(map_)),desc='Building Map'):
            map_dict[str(map_[i][0])] = (map_[i][1],map_[i][2])
        return map_dict

def transfer_points_to_traj(traj_points):
        # traj_points: [{"TripID": tid,"Points": [[id, time] for p in ps.split("_")],"DepartureTime": dt,"Duration": dr}]
        traj = []
        time_step = []
        for i in range(len(traj_points)):
            # choice time step
            if traj_points[i]["Duration"] <= 60:
                time_step.append(1)
            elif traj_points[i]["Duration"] <= 3600:
                time_step.append(60)
            else:
                time_step.append(3600)

            # repeat times
            repeat_times = []
            for j in range(len(traj_points[i]["Points"])-1): #! 0-indexing
                repeat_times.append(math.ceil((traj_points[i]["Points"][j+1][1]-traj_points[i]["Points"][j][1])/(time_step[i])))
            traj_ = []
            for j in range(len(traj_points[i]["Points"])-1):
                traj_ += [traj_points[i]["Points"][j][0]+1]*repeat_times[j] #! 1-indexing
            traj_ += [traj_points[i]["Points"][-1][0]+1] #! 1-indexing
            traj.append(torch.tensor(traj_,dtype=torch.int32))
        traj_num = len(traj)
        return traj, time_step,traj_num

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
    #repeat the traj every 10s
    for i in tqdm(range(len(trajs)),desc='Repeating Traj'):
        traj=[[trajs[i][j]]*int(((time[i][j+1]-time[i][j])//10+1)) for j in range(len(trajs[i])-1)] +[[trajs[i][-1]]]
        trajs[i] = [item for sublist in traj for item in sublist]
    return trajs

def jinan_generate_edge_traj_oyo(save_path:str='data/jinan/edge_traj_new/'):
    
    map_dict = jinan_build_map(mode='node2edge')
    trajs,time = jinan_read_traj()
    max_len = max([len(traj) for traj in trajs])
    print('max len',max_len)
    max_time_diff = []
    for i in range(len(time)):
        t = time[i]
        max_time_diff.append(max([t[i+1]-t[i] for i in range(len(t)-1)]))
    print('max time diff',max(max_time_diff))
    for i in tqdm(range(len(trajs)),desc='Processing'):
        traj = trajs[i]
        edge = node2edge(map_dict,traj)
        #repeat the traj every 10s
        edge = [[edge[j]]*int(((time[i][j+1]-time[i][j])//10+1)) for j in range(len(edge))]
        edge = [item for sublist in edge for item in sublist]
        path = save_path+str(i+1)+'.npy'
        np.save(path,edge)    

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

def preprocess_data_boston(path):
    if isinstance(path, str):
        origin_data = pd.read_csv(path).to_dict(orient='list')
    else:
        origin_data = path
    E = len(origin_data['edge_id'])
    # Record the coordinates of the nodesand calculate the bounds of weights
    #! 0-indexing
    pos = {}
    edges = []
    for i in range(E):
        u, v = origin_data['from_node_id'][i], origin_data['to_node_id'][i]
        w = origin_data['length'][i] / origin_data['speed_limit'][i]
        edges.append((u, v, w, 10*w))
        if u not in pos:
            pos[u] = (origin_data['from_lon'][i], origin_data['from_lat'][i])
        if v not in pos:
            pos[v] = (origin_data['to_lon'][i], origin_data['to_lat'][i])

    return edges, pos

def preprocess_node(node_dir):
    #! 0-indexing
    pos = pd.read_csv(node_dir)
    pos = { int(nid): (float(lon), float(lat), bool(hasc)) for _, (nid, lon, lat, hasc) in pos.iterrows() }
    return pos

def preprocess_edge(edge_dir):
    #! 0-indexing
    edges = pd.read_csv(edge_dir)
    edges = [(int(o), int(d), float(l), translate_roadtype_to_capacity(c)) for _, (o, d, c, geo, l) in edges.iterrows()
    ]
    return edges

def remove_consecutive_duplicates(lst:List)->List:
    return [lst[i] for i in range(len(lst)) if i == 0 or lst[i] != lst[i - 1]]

def jinan_node_type(path='data/jinan/adjcent.npy'):
    adjlist = adj_m2adj_l(np.load(path))
    adjlist = adjlist.numpy()
    types = []
    n = len(adjlist)
    for i in range(n):
        types.append(sum([1 for x in adjlist[i,:,0].tolist() if x != 0]))
    return types

def read_city(city, path='data'):
    if city in ['boston', 'paris']:
        origin_data = pd.read_csv(path + '/'+ city + f'/{city}_data.csv').to_dict(orient='list')
        edges, pos = preprocess_data_boston(origin_data) #! 0-indexing
    elif city in ['jinan']:
        node_dir = f"{path}/{city}/node_{city}.csv"
        edge_dir = f"{path}/{city}/edge_{city}.csv"
        pos = preprocess_node(node_dir) #! 0-indexing
        edges = preprocess_edge(edge_dir) #! 0-indexing
    elif city in ['shenzhen']:
        node_dir = f"{path}/{city}/node_{city}.csv"
        edge_dir = f"{path}/{city}/edge_{city}.csv"
        pos = preprocess_node(node_dir)
        edges = preprocess_edge(edge_dir)

    return edges, pos

def sample_capacity(capacity_scale = 10, edge_num = 100):
    edge_capacity = np.random.uniform(1,capacity_scale,edge_num)
    return edge_capacity

# transfer node, wrighted_adj to graph
#! 0-indexing
def transfer_graph(adj_table):
    G = nx.DiGraph()
    for i in range(len(adj_table)):
        G.add_node(i)
    for i in range(len(adj_table)):
        for j in range(len(adj_table[i])):
            if adj_table[i,j,1] != 0:
                G.add_edge(i,adj_table[i,j,0]-1,weight=adj_table[i,j,1])
    return G

def translate_roadtype_to_capacity(roadtype):
    dic = {'living_street': 1, 'motorway': 10, 'motorway_link': 10, 'primary': 8, 'primary_link': 8, 'residential': 2, 'secondary': 6, 'secondary_link': 6, 'service': 3, 'tertiary': 4, 'tertiary_link': 4, 'trunk': 7, 'trunk_link': 7, 'unclassified': 5}
    return dic[roadtype]



def get_task4_data(node_traj,V,adj_table,pos):
    #node_traj: [B,T,1]
    #V = 8909/243/11933

    B, T, _ = node_traj.shape
    node_traj = node_traj.numpy()
    data = torch.zeros([T,V,7])
    for i in range(0,T-2):
        datai = node_traj[:,i:i+3,:]
        car_type = decide_car_type(datai,adj_table,pos)
        for key,value in car_type.items():
            data[i+1,key,value] += 1
    return data

def decide_car_type(data,adj_table,pos):
    #data: [B,3,1]
    car_type = {}
    B, _, _ = data.shape
    for i in range(0,B):
        key ,val = decide_car_type_(data[i,0],data[i,1],data[i,2],adj_table,pos)
        if val != 100:
            car_type[key] = val
    return car_type


def decide_car_type_(start,mid,end,adj_table,pos):
    #to 0-index
    mid,start,end = mid[0]-1,start[0]-1,end[0]-1
    if sum((adj_table[mid, :, 0] != 0).tolist()) not in [3,4]:
        return 100,100
    if sum((adj_table[mid, :, 0] != 0).tolist())==3:
        #0-index
        nodes = [adj_table[mid,0,0]-1,adj_table[mid,1,0]-1,adj_table[mid,2,0]-1]
        pos_ = {int(n):pos[int(n)] for n in nodes}
        nodes_order = decide_order(3,nodes,pos_)
        try:
            start_order = nodes_order[start]
            end_order = nodes_order[end]
            type_dict = {
                (1,3):4,
                (3,1):4,
                (1,2):5,
                (2,1):5,
                (2,3):6,
                (3,2):6
            }
            return mid, type_dict[(start_order,end_order)]
        except:
            return mid,100
    if sum((adj_table[mid, :, 0] != 0).tolist())==4:
        #0-index
        nodes = [adj_table[mid,0,0]-1,adj_table[mid,1,0]-1,adj_table[mid,2,0]-1,adj_table[mid,3,0]-1]
        pos_ = {int(n):pos[int(n)] for n in nodes}
        nodes_order = decide_order(4,nodes,pos_)
        try:
            start_order = nodes_order[start]
            end_order = nodes_order[end]
            type_dict = {
                (1,3):0,
                (3,1):0,
                (1,2):1,
                (3,4):1,
                (2,4):2,
                (4,2):2,
                (4,1):3,
                (2,3):3
            }
            if (start_order,end_order) in type_dict:
                return mid, type_dict[(start_order,end_order)]
            return mid,100
        except:
            return mid,100

def decide_order(num:int,adj:List[int],map:dict)->dict:
    if num == 3:
        la_adj = [map[int(x)][0] for x in adj]
        lo_adj = [map[int(x)][1] for x in adj]
        index1 = np.argmin(la_adj)
        lo_adj[index1] = 1000
        index2 = np.argmin(lo_adj)
        ans = [1,1,1]
        ans[index1] = 2
        ans[index2] = 3
        #print(la_adj,lo_adj)
        #print(index1,index2)
        return {int(x):y for x,y in zip(adj,ans)}
    if num == 4:
        la_adj = [map[int(x)][0] for x in adj]
        lo_adj = [map[int(x)][1] for x in adj]
        index1 = np.argmin(la_adj)
        index3 = np.argmax(la_adj)
        lo_adj[index1] = 1000
        lo_adj[index3] = 1000
        index2 = np.argmin(lo_adj)
        ans = [2,2,2,2]
        ans[index1] = 3
        ans[index2] = 1
        ans[index3] = 4
        #print(la_adj,lo_adj)
        #print(index1,index2,index3)
        return {int(x):y for x,y in zip(adj,ans)}
    else:
        raise ValueError('Error: The number of adj is wrong')

if __name__ == '__main__':

    #save the traj in data/jinan/traj_repeat_one_by_one,name:1.npy,2.npy,...
    # trajs,time= jinan_read_traj()
    # trajs = jinan_trajs_repeat(trajs,time)
    # max_len = max([len(traj) for traj in trajs])
    # print(max_len)
    # print(trajs[0])
    # path = 'data/jinan/node_traj_repeat_one_by_one/'
    # os.makedirs(path, exist_ok=True)
    # for i in range(len(trajs)):
    #     np.save(path+str(i+1)+'.npy',trajs[i],allow_pickle=True)
    # data = jinan2adj()
    # np.save('data/jinan/adjcent.npy',data)
    # path = 'data/jinan/node_traj_repeat_one_by_one/'
    # edgepath='data/jinan/edge_traj_repeat_one_by_one/'
    # edgefiles = os.listdir(edgepath)
    # files = os.listdir(path)
    # map_ = pd.read_csv('data/jinan/edge_node_jinan.csv')
    # map_ = np.array(map_[['EdgeID','Origin','Destination']])
    # map_dict = {}
    # for i in tqdm(range(len(map_)),desc='Building Map'):
    #     map_dict[str(map_[i][0])] = (int(map_[i][1]),int(map_[i][2]))

    # with open('data/jinan/edge_node_map_dict.pkl','wb') as f:
    #     pickle.dump(map_dict,f)

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
    # path = 'data/jinan/adjcent.npy'
    # data = np.load(path)
    # for i in range(len(data)):
    #     for j in range(len(data)):
    #         if data[i][j] != 0:
    #             data[i][j] = int(map_dict[str(i)+'_'+str(j)]) 
    # np.save('data/jinan/adjcent_class.npy',data)

    # TODO:处理济南数据，找出三岔路口和四岔路口
    # path = 'data/jinan/node_jinan.csv'
    # data = pd.read_csv(path)
    # data = np.array(data[['NodeID','Latitude','Longitude']])
    # map_ = {}
    # for i in range(len(data)):
    #     map_[data[i][0]] = [data[i][1],data[i][2]]
    # adj_path = 'data/jinan/adjcent.npy'
    # adj = np.load(adj_path)
    # adj_l = adj_m2adj_l(adj)
    # adj_l_34 = []
    # for i in range(len(adj_l)):
    #     adj = [int(x) for x in adj_l[i,:,0].tolist() if x != 0]
    #     if len(adj) == 3 or len(adj)==4:
    #         adj_l_34.append([i,adj,decide_order(len(adj),adj,map_)])
    # map_ = 0
    # np.save('data/jinan/dict_decide_rode_type.npy',adj_l_34)

  


    #0_index to 1-index
    # path = 'data/jinan/node_traj_repeat_one_by_one'
    # files = os.listdir(path)
    # for file in tqdm(files,desc='Processing'):
    #     traj = np.load(path+'/'+file)
    #     traj = [t+1 for t in traj]
    #     np.save(path+'/'+file,traj)
    #     # break
    # path = 'data/jinan/traj_jinan_min_one_by_one'
    # files = os.listdir(path)
    # out_path = 'data/jinan/traj_min_test'
    # for i in range(1000):
    #     file = files[i]
    #     traj = np.load(path+'/'+file)
    #     np.save(out_path+'/'+str((i+1))+'.npy',traj)
        # break
    # path = 'data/jinan/edge_traj_repeat_one_by_one'
    # files = os.listdir(path)
    # lengths = torch.tensor([len(np.load(path + '/' + file)) for file in files])
    # top_lengths = torch.topk(lengths, 10).values
    # print(top_lengths)

    #task2 191
    #task1

    # map_ = pd.read_csv('data/jinan/edge_node_jinan.csv')
    # map_ = np.array(map_[['EdgeID','Origin','Destination']])
    # map_dict = {}
    # for i in tqdm(range(len(map_)),desc='Building Map'):
    #     map_dict[str(map_[i][1])+'_'+str(map_[i][2])] = map_[i][0]
    # with open('data/jinan/node_edge_map_dict.pkl','wb') as f:
    #     pickle.dump(map_dict,f)
    # print(map_dict['5_7'],map_dict['7_5'])
    # path = 'data/jinan/edge_traj_new/'
    # files = os.listdir(path)
    # i = 1
    # for file in tqdm(files,desc='Processing'):
    #     traj = np.load(path+file)
    #     if len(traj) > 300:
    #         i+=1
    # print(i)
    # path = 'data/jinan/adjcent_class.npy'
    # data = np.load(path)
    # print(len(data))
    # print(len(data[0]))
    # path = 'data/jinan/node_traj_repeat_one_by_one'
    # files = os.listdir(path)
    # i = 0
    # for file in tqdm(files,desc='Processing'):
    #     traj = np.load(path+'/'+file)
    #     if len(traj) > 60:
    #         i+=1
    # print(i)
    # types = jinan_node_type()
    # adjlist = adj_m2adj_l(np.load('data/jinan/adjcent.npy'))
    # adjlist = adjlist.numpy()
    # print(adjlist[0,:,0])
    # print(types)

    # city = 'boston'
    # edges, pos = read_city(city, path='data/boston')
    # node_num = len(pos)
    # edge_num = len(edges)
    # data_dir = 'data/boston/traj_min_one_by_one'
    # print('Start generating data...')
    # for t in tqdm(range(0, 100000), desc=f'Generating data'):
    #     all_encoded_trajectories, all_adj_list = generate_data(city = city, total_trajectories = 1, max_length = 100, capacity_scale = 10, weight_quantization_scale = 20, max_connection = 4 )
    #     if t == 0:
    #         np.save(f'data/boston/adj_table_list.npy',all_adj_list[0])
    #     path = data_dir+f'/{t}.npy'
    #     np.save(path,all_encoded_trajectories[0])
    # print(f'one by one saved successfully!')
    
    # city = 'boston'
    # edges, pos = read_city(city, path='data/boston')
    # node_num = len(pos)
    # edge_num = len(edges)
    # print(node_num,edge_num)   
    # data_dir = 'data/boston/traj_min_one_by_one'
    # save_dir = 'data/boston/traj_boston_min_one_by_one'
    # files = os.listdir(save_dir)
    # print(len(files))
    # path = 'data/boston/adj_table_list.npy'
    # data = np.load(path)
    # print(data.shape)
    # print(data)
    # path = 'data/jinan/adjcent.npy'
    # data = np.load(path)
    # print(data)
    # city = 'shenzhen'
    # pos ,edges = read_city(city)
    # #11933,27410
    # node_num = len(pos)
    # edge_num = len(edges)
    # print(node_num)
    # print(edge_num)
    # data_dir = 'data/shenzhen/traj_shenzhen_min_one_by_one'
    # os.makedirs(data_dir, exist_ok=True)
    # print('Start generating data...')
    # for t in tqdm(range(0, 400), desc=f'Generating data'):
    #     all_encoded_trajectories, all_adj_list = generate_data(city = city, total_trajectories = 1000, max_length = 100, capacity_scale = 10, weight_quantization_scale = 20, max_connection = 8 )
    #     if t == 0:
    #         np.save(f'data/shenzhen/adj_table_list.npy',all_adj_list[0])
    #     for i in range(1000):
    #         path = data_dir+f'/{t*10000+i+1}.npy'
    #         np.save(path,all_encoded_trajectories[i])
    # print(f'one by one saved successfully!')
    # path = 'data/shenzhen/traj_shenzhen_min_one_by_one/1.npy'
    # data = np.load(path)
    # print(data)

    #生成task4数据
    # pos,edges = read_city('jinan')
    # path = 'data/jinan/node_traj_repeat_one_by_one'
    # adj_path = 'data/jinan/adjcent.npy'
    # adj_l = adj_m2adj_l(np.load(adj_path))
    # adj_l = adj_l.numpy()
    # files = os.listdir(path)

    # for file in files:
    #     traj = np.load(path+'/'+file)

    #     traj = [traj[i] for i in range(len(traj)) if i == 0 or traj[i] != traj[i - 1]]
    #     traj = torch.tensor([traj]).view(1,-1,1)
    #     data = get_task4_data(traj,8909,adj_l,pos)

    #重新生成task1，3数据
    city = 'jinan'
    csv_dir = 'data/jinan/traj_jinan.csv'
    trajs = preprocess_traj(csv_dir)
    path = 'data/jinan/edge_traj_repeat_one_by_one'
    i = 1
    for tid in tqdm(range(len(trajs)), desc=f'Transfering {city} points into trajectories'):
            traj, time_step,length = transfer_points_to_traj(trajs[tid])
            for j in range(length):
                x = traj[j].numpy()
                np.save(f'{path}/{tid+1}.npy', x)
                i += 1