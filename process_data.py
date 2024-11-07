## 将数据完成时间上的重复，假设车速都为1
## 实现轨迹表示node-edge之间的转换

from itertools import groupby
import numpy as np
import pandas as pd
from torch.fx.node import map_aggregate
from tqdm import tqdm

def read_traj(file_path):
    trajs= pd.read_csv(file_path)
    trajs = [list(map(int, traj.split("_"))) for traj in trajs['Trajectory']]
    return trajs

def repeat_traj(trajs,map_):
    #map: np.array([[edge,o,d,length],...])
    #traj: [[],[],...]
    new_trajs = []
    for traj in tqdm(trajs,desc='Repeating Trajectories'):
        new_traj = []
        traj = edge_node_trans(map_,traj,is_edge=False)
        for edge in traj:
            length = map_[map_[:,0]==edge][0][3]
            new_traj.extend([edge]*int(length))
        new_trajs.append(new_traj)
    return new_trajs

def edge_node_trans(map_,list_,is_edge=True):
    # map: np.array([[edge,o,d...],...])
    if is_edge:
        nodes = []
        for edge in list_:
            nodes.append(map_[map_[:,0]==edge][0][1])
        nodes.append(map_[map_[:,0]==list_[-1]][0][2])
        return nodes
    else:
        edges = []
        e_l = []
        repeats = []
        nodes = [key for key, _ in groupby(list_)]
        for i in range(len(nodes)-1):
            bool_index = [list([int(x) for x in maps])==[nodes[i],nodes[i+1]] for maps in map_[:,1:3]]
            bool_index_r = [list([int(x) for x in maps])==[nodes[i+1],nodes[i]] for maps in map_[:,1:3]]
            try:
                e_l.append(int(map_[bool_index][0][0]))
            except:
                e_l.append(int(map_[bool_index_r][0][0]))
        list_ = list_[:-1]
        for _, group in groupby(list_):
            repeats.append(sum(1 for _ in group))
        for i in range(len(e_l)):
            edges.append(e_l[i]*repeats[i])
        return edges

def pre_map(file_path):
    data = pd.read_csv(file_path)
    map=[]
    for i in range(len(data)):
        map.append([data['EdgeID'][i],data['Orgin'][i],data['Destination'][i]])
    map = np.array(map)
    return map

if __name__ == '__main__':

    # map = pre_map('../data/jinan/edge_node_jinan.csv')
    # list = [1,1,2,3,3,5,5,7,7,8]
    # out = edge_node_trans(map,list)
    # print(out)
    trajs = read_traj('data/simulation/trajectories_10*10.csv')
    print(trajs[0:1])
    map_ = pd.read_csv('data/simulation/edge_node_10*10.csv')
    map_ = np.array(map_)
    print(map_.shape)
    new_trajs = repeat_traj(trajs[0:1],map_)
    print(new_trajs)