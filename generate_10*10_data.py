from click import Path
import numpy as np
import pandas as pd
import networkx as nx
import random
from tqdm import tqdm

def generate_node_edge(grid_size):
    nodes = [[i, j] for i in range(1,grid_size+1) for j in range(1,grid_size+1)]
    edges = [] 
    for i in range(1,grid_size+1):
        for j in range(1,grid_size+1):
            node = [i, j]

            if j < grid_size :
                right_node = [i, j + 1]
                length = np.random.uniform(1.0, 20.0) 
                edges.append([node, right_node, length])
            

            if i < grid_size :
                bottom_node = [i + 1, j]
                length = np.random.uniform(1.0, 20.0) 
                edges.append([node, bottom_node, length])

    nodeID = [[i+1,nodes[i]] for i in range(len(nodes))]
    edgeID = [[i+1,edges[i]] for i in range(len(edges))]

    return nodes, edges, nodeID, edgeID

def map_node_to_id(node,nodeID):
    for i in range(len(nodeID)):
        if nodeID[i][1]==node:
            return nodeID[i][0]
    return 

def save_edge_node_length(edgeID,nodeID,path='data/simulation/edge_node_10*10.csv'):
    edge_node = []
    for edge in edgeID:
        data = [edge[0],map_node_to_id(edge[1][0],nodeID),\
                map_node_to_id(edge[1][1],nodeID),edge[1][2]]
        edge_node.append(data)
    data = pd.DataFrame(edge_node,columns=['EdgeID','Origin','Destination','Length','Class'])
    data.to_csv(path,index=False)


def generate_random_trajectory(filepath, num_traj=5,min_length=3):
    # datas = pd.read_csv(filepath)
    adj = np.load(filepath)
    G = nx.DiGraph()
    for i in range(len(adj)):
        for j in range(len(adj[i])):
            if adj[i][j] != 0:
                G.add_edge(i+1,j+1,weight=adj[i][j])
                G.add_edge(j+1,i+1,weight=adj[i][j])
    trajectories = []
    nodes = list(G.nodes())
    with tqdm(total=num_traj, desc="Generating Trajectories") as pbar:
        while len(trajectories) < num_traj:
            # 随机选择起点和终点
            start = random.choice(nodes)
            end = random.choice(nodes)
            #print(start,end)
            while start == end:
                end = random.choice(nodes)
            
            try:
                # 生成路径
                path = nx.shortest_path(G, source=start, target=end, weight='weight')
                # 计算路径的加权长度
                #path_length = nx.path_weight(G, path, weight='weight')
                traj = [int(node) for node in path]
                # 检查路径的加权长度是否满足要求
                #if path_length >= min_length:
                if len(traj) >= min_length:
                    trajectories.append(traj)
                    pbar.update(1)
            except nx.NetworkXNoPath:
                pass  # 跳过无路径的情况

    return trajectories
    for _ in range(num_traj):
        start = random.choice(nodes)
        end = random.choice(nodes)
        print(start,end)
        while start == end:
            end = random.choice(nodes)
        try:
            path = nx.shortest_path(G, source=start, target=end, weight='weight')
            trajectories.append([int(node) for node in path])
        except nx.NetworkXNoPath:
            pass 
    return trajectories

def save_node_type(nodeID,grid_size=10,path='data/simulation/node_type_10*10.csv'):
    save_data = []
    for node in nodeID:
        if (node[1][0]==grid_size or node[1][0]==1) and (node[1][1]==grid_size or node[1][1]==1):
            save_data.append([node[0],'Turning',node[1]])
        elif (node[1][0]==grid_size or node[1][0]==1) or (node[1][1]==grid_size or node[1][1]==1):
            save_data.append([node[0],'T',node[1]])
        else:
            save_data.append([node[0],'C',node[1]])
    data = pd.DataFrame(save_data,columns=['NodeID','Type','coordinate'])
    data.to_csv(path,index=False)

if __name__ == '__main__':

    #nodes, edges, nodeID, edgeID = generate_node_edge(10)
    #print(nodeID)
    #save_edge_node_length(edgeID,nodeID)
    #save_node_type(nodeID)
    # trajectories = generate_random_trajectory('data/simulation/edge_node_10*10.csv',100000,min_length=20)
    # trajectories_str = ["_".join(map(str, traj)) for traj in trajectories]
    # traj = pd.DataFrame(trajectories_str,columns=['Trajectory'])
    # traj.to_csv('data/simulation/trajectories_10*10.csv',index=False)
    filepath = 'data/jinan/adjcent.npy'
    trajectories = generate_random_trajectory(filepath,100000,min_length=3)
    savepath = 'data/jinan/traj_jinan_min_one_by_one/'
    for i in range(len(trajectories)):
        filepath = savepath+str(i+1)+'.npy'
        np.save(filepath,trajectories[i],allow_pickle=True)

