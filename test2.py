from train import train2
from dataloader import SmartTrafficDataset, SmartTrafficDataloader
import numpy as np
from utils import adj_m2adj_l
from task2.util import transfer_graph, transfer_graph_
import networkx as nx
import matplotlib.pyplot as plt
from utils import calculate_bounds, read_city
import time
from task2.process_task2 import get_model
import torch

weights_path = 'weights/jinan/task2New/best_model_0.0070.pth'
#python train.py --device cuda:3 --T 10 --max_len 20 --task_type 1 --vocab_size 8909 --batch_size 1024 --epochs 40 --learning_rate 0.001 --n_embd 32 --n_hidden 16 --n_layer 8 --dropout 0.1  --model_save_path weights/jinan/task2/ --trajs_path data/jinan/traj_jinan_min_one_by_one/

cfg = {
    'model_read_path': None,
    'model_save_path': 'weights/jinan/task2',
    'trajs_path': 'data/jinan/traj_min_test/',
    'trajs_path_train': 'data/jinan/traj_min_test1/',
    'max_len': 193,
    'vocab_size': 8909,
    'n_embd': 32,
    'n_hidden': 8,
    'n_layer': 4,
    'n_head': 4,
    'dropout': 0.1,
    'window_size': 1,
    'T': 192,
    'batch_size': 64,
    'learning_rate': 0.001,
    'epochs': 3,
    'device': 'cuda:2'
}
cfg['block_size'] = cfg['T']

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

def train_presention(epochs=3,batch_size=64,lr=0.001):
    
    cfg['epochs'] = epochs
    cfg['batch_size'] = batch_size
    cfg['learning_rate'] = lr
    dataset2 = SmartTrafficDataset(None,mode="task2",trajs_path=cfg['trajs_path_train'],T=cfg['T'],max_len=cfg['max_len']
                                   ,adjcent_path='data/jinan/adjcent.npy') 
    data_loader2 = SmartTrafficDataloader(dataset2,batch_size=cfg['batch_size'],shuffle=True, num_workers=4)
    from train import train2
  
    train2(cfg, data_loader2)        

def test_presention(num=1,od=None):
    cfg['model_read_path'] = weights_path
 
    task2_model = get_model(cfg)
    task2_model.load_state_dict(torch.load(weights_path))
    task2_model.eval()
    task2_model.to(cfg['device'])
    adjcent_path = 'data/jinan/adjcent.npy'
    adjcent = np.load(adjcent_path)
    adj_l = adj_m2adj_l(adjcent)
    indices ,values =adj_l[:,:,0],adj_l[:,:,1]
    indices = torch.tensor(indices,dtype=torch.int).to(cfg['device'])
    values = torch.tensor(values,dtype=torch.float).to(cfg['device'])
    idx = torch.zeros(1,1,1,dtype=torch.int).to(cfg['device'])
    condation = torch.zeros(1,1,1,1,dtype=torch.int).to(cfg['device'])
    time_diff = []
    od_list = []
    model_trajs_list = []
    Djs_trajs = []
    for i in range(num):
        #o,d= np.random.choice(np.arange(1,cfg['vocab_size']),2).tolist()
        idx[0,0,0] = od[i][0]
        condation[0,0,0,0] = od[i][1]
        od_list.append([od[i][0],od[i][1]])
        G = transfer_graph_(adj_l)
        print('Origin_Destination:',od_list[-1])
        x0 = time.time()
        path = nx.shortest_path(G, source=od[i][0], target=od[i][1], weight='weight')
        time.sleep(1)
        x1 = time.time()
        Djs_trajs.append(path)
        print('Djs spend time:',x1-x0)
        y0 = time.time()
        with torch.no_grad():
            trajs =task2_model.generate(idx,condation,(indices,values))
        y1 = time.time()
        print('model spend time:',y1-y0)
        time_diff.append(x1-x0-y1+y0)
        model_trajs_list.append(trajs)

    print('Origin_Destination:',od_list)
    for j in range(len(model_trajs_list)):
        print(f'car{j+1}',model_trajs_list[j])
    print('Djs:',Djs_trajs)

    return model_trajs_list,Djs_trajs

def plot_volume1(min_path, traj, fig_size=20, save_path='task2_test.png'):
    # G networkx graph
    # pos position of the nodes, get from read_city('boston')[1]
    # volume_single: V
    # min_path: list of edges to be shown in blue
    # traj: list of nodes representing a path, shown as points

    edges, pos = read_city('jinan',path='data/jinan')
    weight = [edge[2] for edge in edges]
    adj_table = get_weighted_adj_table(edges, pos, weight, max_connection=9)
    G = transfer_graph(adj_table)
    for i in pos:
        pos[i] = pos[i][:-1]
    
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
    nx.draw_networkx_edges(G, pos, width=fig_size/15, alpha=1, edge_color=edge_colors, ax=ax, arrows=False)
    nx.draw_networkx_edges(G, pos, width=fig_size / 15, alpha=1, edge_color='white', ax=ax, arrows=False)

    min_path = [(min_path[i], min_path[i + 1]) for i in range(len(min_path) - 1)]
    traj_path = [(traj[i], traj[i + 1]) for i in range(len(traj) - 1)]
    nx.draw_networkx_edges(G, pos, edgelist=min_path, width=fig_size / 5, alpha=1, edge_color='blue', ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=traj_path, width=fig_size / 5, alpha=0.3, edge_color='red', ax=ax)
   
    # Plot traj as points distributed along edges
    

    # Plot all other edges in black
    

    # Display the figure
    plt.tight_layout()
    plt.show()
    if save_path is not None:
        plt.savefig(save_path)
    return fig

def task2_test(k):
    #train_presention()
    i = 0
    dataset2 = SmartTrafficDataset(None,mode="task2",trajs_path=cfg['trajs_path'],T=cfg['T'],max_len=cfg['max_len'],adjcent_path='data/jinan/adjcent.npy')
    for traj_ ,valid_length,od,traj_targ,indices, values in dataset2:
        i+=1
        if i<k:
            continue
        #print(indices,values)
        #print(traj_)
        traj,path = test_presention(1,[[od[0,0,1].item(),od[0,0,0].item()]])
        break
    path = np.array(path[0])-1
    traj = np.array(traj[0])-1
    im = plot_volume1(path,traj,save_path='task2_test.png')
    return im