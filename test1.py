from dataloader import SmartTrafficDataset, SmartTrafficDataloader
from task1.test1 import define_model
import torch
from utils import adj_m2adj_l
import numpy as np
from task3.utils import transfer_graph
import networkx as nx
import pickle
import matplotlib.pyplot as plt
from task3.utils import calculate_bounds, read_city
import pandas as pd


weights_path = 'weights/jinan/task1New/best_model_0.0004.pth'
cfg = { 'T':192,
        'max_len':193,
        'vocab_size':23313,
        'batch_size':256,
        'epochs':5,
        'learning_rate':0.001,
        'n_embd':32,
        'n_hidden':16,
        'n_layer':8,
        'dropout':0.1,
        'model_read_path':None,
        'model_save_path':'weights/jinan/task1',
        'trajs_path_train':'data/jinan/edge_traj_test1/',
        'trajs_path':'data/jinan/edge_traj_test/',
        'device':'cuda:1',
        'n_head':4,    
        }
cfg['block_size'] = cfg['T']

    
def plot_volume1(min_path, traj, fig_size=20, save_path='task1.png'):
    # G networkx graph
    # pos position of the nodes, get from read_city('boston')[1]
    # volume_single: V
    # min_path: list of edges to be shown in blue
    # traj: list of nodes representing a path, shown as points
    path_edge_node_map = 'data/jinan/edge_node_map_dict.pkl'
    with open(path_edge_node_map, 'rb') as f:
        edge_node_map = pickle.load(f)
    from collections import Counter

    
    traj_counts = Counter(traj) 
    traj = list(traj_counts.keys()) 
    cnt = list(traj_counts.values())
    cnt = np.random.choice(np.arange(4,10),len(cnt))

    edges, pos = read_city('jinan')
    weight = [edge[2] for edge in edges]
    adj_table = get_weighted_adj_table(edges, pos, weight, max_connection=9)
    G = transfer_graph(adj_table)
    for i in pos:
        pos[i] = pos[i][:-1]
    
    min_path = [(min_path[i], min_path[i + 1]) for i in range(len(min_path) - 1)]
    traj_path = []
    x_min, x_max, y_min, y_max = calculate_bounds(pos)
    fig, ax = plt.subplots(1, 1, figsize=(fig_size, fig_size * (y_max - y_min) / (x_max - x_min)))
    ax.set_facecolor('black')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(np.arange(x_min, x_max, 1))
    ax.set_yticks(np.arange(y_min, y_max, 1))
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    ax.axvline(x=0, color='k', linestyle='--', linewidth=0.5)
    edge_colors = np.array([0 for u, v in G.edges()])
    edge_colors = plt.cm.coolwarm(1 - edge_colors)
    # edge_colors = plt.cm.RdYlBu(1 - edge_colors)
    # Plot min_path as blue edges
    nx.draw_networkx_edges(G, pos, width=fig_size/15, alpha=1, edge_color='white', ax=ax, arrows=False)
    nx.draw_networkx_edges(G, pos, width=fig_size/15, alpha=1, edge_color=edge_colors, ax=ax, arrows=False)
    nx.draw_networkx_edges(G, pos, edgelist=min_path, width=fig_size / 15, alpha=1, edge_color='green', ax=ax)
    # Plot traj as points distributed along edges
    node_traj = []
    for i in range(len([x  for x in traj if x!=0])):
        start_node, end_node = edge_node_map[str(traj[i])]
        node_traj.append(start_node)
    #     traj_path.append((start_node-1, end_node-1))
    # traj_path_=[]
    # for i in range(len(traj_path)):
    #     if i == 0:
    #         if len(traj_path)>1 and(traj_path[i][0] not in traj_path[i+1]):
    #             traj_path_.append(traj_path[i][0])
    #             traj_path_.append(traj_path[i][1])
    #         else:
    #             traj_path_.append(traj_path[i][1])
    #             traj_path_.append(traj_path[i][0])
    #     else:
    #         if traj_path[i][0] in traj_path_:
    #             traj_path_.append(traj_path[i][1])
    #         else:
    #             traj_path_.append(traj_path[i][0])

        start_pos = pos[start_node+1]
        end_pos = pos[end_node+1]
        # for j in range(1, cnt[i] + 1):
        #     x = (start_pos[0] * (cnt[i] - j) + end_pos[0] * j) / float(cnt[i])
        #     y = (start_pos[1] * (cnt[i] - j) + end_pos[1] * j) / float(cnt[i])
        #     #print(x,y)
        #     ax.plot(x,y, 'ro', markersize=3)
        if i == 0:
            ax.plot(start_pos[0], start_pos[1], 'b*', markersize=10)
            ax.plot(end_pos[0], end_pos[1], 'b*', markersize=10)
        elif i == len([x  for x in traj if x!=0])-1:
            ax.plot(start_pos[0], start_pos[1], 'y*', markersize=10)
            ax.plot(end_pos[0], end_pos[1], 'y*', markersize=10)
        else:
            ax.plot(end_pos[0], end_pos[1], 'ro', markersize=3)
            ax.plot(start_pos[0], start_pos[1], 'ro', markersize=3)
        
    print('node_traj',node_traj)

    # traj_path = [(traj_path_[i], traj_path_[i + 1]) for i in range(len(traj_path_) - 1)]
    # print(traj_path)
    # nx.draw_networkx_edges(G, pos, edgelist=traj_path, width=fig_size / 15, alpha=1, edge_color='red', ax=ax)
    
    # Plot all other edges in black
    

    # Display the figure
    plt.tight_layout()
    plt.show()
    if save_path is not None:
        plt.savefig(save_path)
    return fig

def train_presention(batch_size=64,epochs=4,lr = 0.001,device='cuda:1'):
    
    cfg['batch_size'] = batch_size
    cfg['epochs'] = epochs
    cfg['learning_rate'] = lr
    cfg['device'] = device
    dataset1 = SmartTrafficDataset(None,mode="task1",trajs_path=cfg['trajs_path'],T=cfg['T'],max_len=cfg['max_len']) 
    data_loader1 = SmartTrafficDataloader(dataset1,batch_size=cfg['batch_size'],shuffle=True, num_workers=4)

    from train import train1
    train1(cfg, data_loader1)        

def test_presention(x=None,edge=None):

    cfg['model_read_path'] = weights_path
    node_edge_map_path ='data/jinan/node_edge_map_dict.pkl' 
    with open(node_edge_map_path,'rb') as f:
        map_dict = pickle.load(f)
    # adjcent_path = 'data/jinan/adjcent.npy'
    # adjcent = np.load(adjcent_path)
    # adj_l = adj_m2adj_l(adjcent)
    # G = transfer_graph(adj_l)
    if x is None:
        o,d = np.random.choice(np.arange(1,8909),2)
    else:
        o,d = x

    model = define_model(cfg)
    model.load_state_dict(torch.load(weights_path))

    model.eval()
    if edge is None:
        raise ValueError('edge is None')
    print(edge[0],edge[1])
    traj = model.generate_traj(edge[0],edge[1])

    return traj

def transfer_edge_to_node(od_list):
    node_list = []
    for i in range(len(od_list)):
        if i == 0:
            od1 = od_list[i]
            od2 = od_list[i+1]
            if od1[0] == od2[0] or od1[0] == od2[1]:
                node_list.append(od1[1])
                node_list.append(od1[0])
            else:
                node_list.append(od1[0])
                node_list.append(od1[1])
        else:
            od = od_list[i]
            if od[0] == node_list[-1]:
                node_list.append(od[1])
            else:
                node_list.append(od[0])
        if i == len(od_list)-1:
            od = od_list[i]
            if od[1] == node_list[-1]:
                node_list.append(od[0])
            else:
                node_list.append(od[1])
    return node_list


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


def translate_roadtype_to_capacity(roadtype):
    dic = {'living_street': 1, 'motorway': 10, 'motorway_link': 10, 'primary': 8, 'primary_link': 8, 'residential': 2, 'secondary': 6, 'secondary_link': 6, 'service': 3, 'tertiary': 4, 'tertiary_link': 4, 'trunk': 7, 'trunk_link': 7, 'unclassified': 5}
    return dic[roadtype]


def djikstra(start, end):
    edges, pos = read_city('jinan')
    weight = [edge[2] for edge in edges]
    adj_table = get_weighted_adj_table(edges, pos, weight, max_connection=9)
    G = transfer_graph(adj_table)
    path = nx.dijkstra_path(G, start, end)
    return path
        

def test_presention1(k):
    map_path = 'data/jinan/edge_node_map_dict.pkl'
    with open(map_path,'rb') as f:
        map_dict = pickle.load(f)
    dataset1 = SmartTrafficDataset(None,mode="task1",trajs_path=cfg['trajs_path'],T=cfg['T'],max_len=cfg['max_len'],need_repeat=False) 
    #data_loader1 = SmartTrafficDataloader(dataset1,batch_size=1,shuffle=False, num_workers=4)
    # i = 0

    x = dataset1[k]

    od = x['cond']
    e1 = od[0,0,0]
    e_1 = od[0,0,1]
    o = map_dict[str(e1.item())][0]
    d = map_dict[str(e_1.item())][1]
    traj = test_presention((o-1,d-1),(e1,e_1))

    print('traj',traj)
    # print(x['traj'].shape)
    # print('x',x['traj'].view(x['traj'].shape[0]).tolist())

    # transfer node into edge
    real_traj_ = x['traj'].view(x['traj'].shape[0]).tolist()
    real_traj_ = np.array(real_traj_)
    real_traj_ = real_traj_[real_traj_>0]
    real_traj_ = [map_dict[str(x)] for x in real_traj_]
    real_traj = transfer_edge_to_node(real_traj_)
    real_traj = np.array(real_traj)-1
    print('real_traj',real_traj)

    traj = np.array(traj)
    traj = traj[traj>0]
    pred_traj_ = [map_dict[str(traj[i])] for i in range(len(traj))]
    pred_traj = transfer_edge_to_node(pred_traj_)
    pred_traj = np.array(pred_traj)-1 # transfer to 0-based for networkx
    print('pred_traj',pred_traj)

    path = djikstra(real_traj[0],real_traj[-1])
    path = np.array(path,dtype=int)
    print('Djs:',path)

    print(map_dict['2503'],map_dict['21721'],map_dict['21720'])
    
    ims = []
    im = plot_volume1(path,traj,fig_size=20, save_path='task1_test.png')
    # for i in range(len([x for x in traj if x!=0])):
    #     ims.append(plot_volume1(path,traj[0:i],fig_size=20, save_path='img/'+f'task1_test{i}.png'))
    return ims
    
if __name__ == '__main__':
    
    test_presention1(1)
    #test_presention1(3)

