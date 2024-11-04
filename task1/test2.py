from itertools import groupby
import numpy as np
import pandas as pd


def edge_node_trans(map,list_,is_edge=True):
    # map: np.array([[edge,o,d],...])
    if is_edge:
        nodes = []
        for edge in list_:
            nodes.append(map[map[:,0]==edge][0][1])
        nodes.append(map[map[:,0]==list_[-1]][0][2])
        return nodes
    else:
        edges = []
        e_l = []
        repeats = []
        nodes = [key for key, _ in groupby(list_)]
        for i in range(len(nodes)):
            e_l.append(map[(map[:,1:]==[nodes[i],nodes[i+1]])][0][0])
        list_ = list_[:-1]
        for _, group in groupby(list_):
            repeats.append(sum(1 for _ in group))
        for i in range(len(e_l)):
            edges.append(e_l[i]*repeats[i])
        edges = np.concatenate(edges)
        return edges

def pre_map(file_path):
    data = pd.read_csv(file_path)
    map=[]
    for i in range(len(data)):
        map.append([data['EdgeID'][i],data['Orgin'][i],data['Destination'][i]])
    map = np.array(map)
    return map

if __name__ == '__main__':

    map = pre_map('../data/jinan/edge_node_jinan.csv')
    list = [1,1,2,3,3,5,5,7,7,8]
    out = edge_node_trans(map,list)
    print(out)