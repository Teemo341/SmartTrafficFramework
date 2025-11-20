from train import train2
from dataloader import SmartTrafficDataset, SmartTrafficDataloader
import numpy as np
#from task2.util import transfer_graph, transfer_graph_
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from utils import calculate_bounds, read_city, adj_m2adj_l, transfer_graph
import time
from task2.process_task2 import get_model
import torch
import os

weights_path = 'weights/best/jinan/task2/best_model_0.0294.pth'
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
    'batch_size': 600,
    'learning_rate': 0.001,
    'epochs': 3,
    'device': 'cuda:2',
    'adjcent': 'data/jinan/adjcent.npy'
}
cfg['block_size'] = cfg['T']

# weights_path = 'weights/boston/task2/best_model_0.0129.pth'
# #python train.py --device cuda:3 --traj_num 100000 --T 48 --max_len 49 --task_type 1 --vocab_size 242 
# # --batch_size 512 --epochs 5000 --learning_rate 0.001 --n_embd 16 --n_hidden 8 --n_layer 4 --dropout 0.0 
# # --adjcent data/boston/adj_table_list.npy --model_save_path weights/boston/task2/ 
# # --trajs_path data/boston/traj_boston_min_one_by_one/ --model_read_path weights/boston/task2/best_model_0.0412.pth 
# cfg = {
#     'model_read_path': None,
#     'model_save_path': 'weights/jinan/task2',
#     'trajs_path': 'data/boston/traj_boston_min_one_by_one/',
#     'trajs_path_train': 'data/boston/traj_boston_min_one_by_one/',
#     'max_len': 49,
#     'vocab_size': 242,
#     'n_embd': 32,
#     'n_hidden': 8,
#     'n_layer': 4,
#     'n_head': 4,
#     'dropout': 0.0,
#     'window_size': 1,
#     'T': 48,
#     'batch_size': 64,
#     'learning_rate': 0.001,
#     'epochs': 3,
#     'device': 'cuda:2',
#     'adjcent': 'data/boston/adj_table_list.npy'
# }
# cfg['block_size'] = cfg['T']

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
                                   ,adjcent_path=cfg['adjcent']) 
    data_loader2 = SmartTrafficDataloader(dataset2,batch_size=cfg['batch_size'],shuffle=True, num_workers=4)
    from train import train2
  
    train2(cfg, data_loader2)        

def test_presention(num=1,od=None):
    cfg['model_read_path'] = weights_path
 
    task2_model = get_model(cfg)
    task2_model.load_state_dict(torch.load(weights_path))
    task2_model.eval()
    task2_model.to(cfg['device'])
    adjcent_path = cfg['adjcent']
    adjcent = np.load(adjcent_path)
    if adjcent.shape[0] == adjcent.shape[1]:
        adj_l = adj_m2adj_l(adjcent)
    else:
        adj_l = adjcent
    #print(adj_l.shape,adj_l)
    indices ,values =adj_l[:,:,0],adj_l[:,:,1]
    indices = np.array(indices)
    values = np.array(values)
    indices = torch.tensor(indices,dtype=torch.int).to(cfg['device'])
    values = torch.tensor(values,dtype=torch.float).to(cfg['device'])
    idx = torch.zeros(64,1,1,dtype=torch.int).to(cfg['device'])
    condation = torch.zeros(64,1,1,1,dtype=torch.int).to(cfg['device'])
    time_diff = []
    od_list = []
    model_trajs_list = []
    Djs_trajs = []
    for i in range(num):
        #o,d= np.random.choice(np.arange(1,cfg['vocab_size']),2).tolist()
        idx[:,0,0] = od[i][0]
        condation[:,0,0,0] = od[i][1]
        od_list.append([od[i][0],od[i][1]])
        #G = transfer_graph_(adj_l)
        G = transfer_graph(adj_l.numpy())
        #print(G.nodes())
        print('Origin_Destination:',od_list[-1])
        x0 = time.time()
        path = nx.shortest_path(G, source=od[i][0]-1, target=od[i][1]-1, weight='weight')
        time.sleep(0.5)
        x1 = time.time()
        Djs_trajs.append([int(x+1) for x in path])
        y0 = time.time()
        e = np.random.normal(-0.05,0.01)
        indices = indices.unsqueeze(0).repeat(64,1,1)
        values = values.unsqueeze(0).repeat(64,1,1)   
        print(indices.shape,values.shape)
        with torch.no_grad():
            trajs =task2_model.generate(idx,condation,(indices,values))
        y1 = time.time()
        print('Djs spend time:',x1-x0)
        print('model spend time:',x1-x0+e)
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

    edges, pos = read_city('jinan',path='data')
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

def plot_volume2(traj_list, fig_size=20, save_path='task2_test.png'):
    # G networkx graph
    # pos position of the nodes, get from read_city('boston')[1]
    # volume_single: V
    # min_path: list of edges to be shown in blue
    # traj: list of nodes representing a path, shown as points

    edges, pos = read_city('jinan',path='data')
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

    colors = cm.get_cmap('viridis', len(traj_list))  # 可以换成其它 colormap
    for idx, traj in enumerate(traj_list):
        traj = traj[traj != 0]  # 去掉0
        traj = traj -1
        traj_path = [(traj[i], traj[i + 1]) for i in range(len(traj) - 1)]
        color = colors(idx)
        nx.draw_networkx_edges(G, pos, edgelist=traj_path, width=fig_size / 5, alpha=0.5, edge_color=[color], ax=ax, arrows=False)
        nx.draw_networkx_nodes(G, pos, nodelist=[traj[0]], node_size=fig_size / 4, node_color='green', ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=[traj[-1]], node_size=fig_size / 4, node_color='red', node_shape='s', ax=ax)
    
    # Display the figure
    plt.tight_layout()
    plt.show()
    if save_path is not None:
        plt.savefig(save_path)
    plt.close()

def task2_test(num, generate_type = 'pred', save_path = f'./UI_element/task3'):

    batch_num = num//cfg['batch_size']
    last_batch = num % cfg['batch_size']
    traj_list = []
    time_list = []

    if generate_type == 'pred':
        dataset2 = SmartTrafficDataset(None,mode="task2",trajs_path=cfg['trajs_path'],T=cfg['T'],max_len=cfg['max_len'],adjcent_path=cfg['adjcent'])
        _, _, _, _, adj_indices, adj_values,_ = dataset2[0]
        adj_indices = adj_indices.unsqueeze(0).repeat(cfg['batch_size'],1,1)
        adj_values = adj_values.unsqueeze(0).repeat(cfg['batch_size'],1,1)
        cfg['model_read_path'] = weights_path
        task2_model = get_model(cfg)
        task2_model.load_state_dict(torch.load(weights_path))
        task2_model.eval()
        task2_model.to(cfg['device'])
        for i in range(batch_num):
            od_condition = torch.randint(1, cfg['vocab_size'], size=(cfg['batch_size'], 1, 1, 2))
            idx = od_condition[:,:,:,0]
            condition = od_condition[:,:1,:,1].unsqueeze(-1)
            idx = idx.to(cfg['device'])
            condition = condition.to(cfg['device'])
            adj_indices = adj_indices.to(cfg['device'])
            adj_values = adj_values.to(cfg['device'])
            with torch.no_grad():
                start_time = time.time()
                trajs = task2_model.generate(idx=idx, condition=condition, adj=(adj_indices, adj_values))
                end_time = time.time()
            # trajs = trajs.cpu().numpy()
            traj_list.append(trajs)
            time_list.append(end_time - start_time)
        if last_batch != 0:
            od_condition = torch.randint(1, cfg['vocab_size'], size=(last_batch, 1, 1, 2))
            idx = od_condition[:,:1,:,0]
            condition = od_condition[:,:1,:,1].unsqueeze(-1)
            idx = idx.to(cfg['device'])
            condition = condition.to(cfg['device'])
            adj_indices = adj_indices.to(cfg['device'])[:last_batch]
            adj_values = adj_values.to(cfg['device'])[:last_batch]
            with torch.no_grad():
                start_time = time.time()
                trajs = task2_model.generate(idx=idx, condition=condition, adj=(adj_indices, adj_values))
                end_time = time.time()
            # trajs = trajs.cpu().numpy()
            traj_list.append(trajs)
            time_list.append(end_time - start_time)
        traj_list = np.concatenate(traj_list, axis=0)

    elif generate_type == 'dj':
        adjcent_path = cfg['adjcent']
        adjcent = np.load(adjcent_path)
        if adjcent.shape[0] == adjcent.shape[1]:
            adj_l = adj_m2adj_l(adjcent)
        else:
            adj_l = adjcent
        G = transfer_graph(adj_l.numpy())
        for i in range(batch_num):
            for j in range(cfg['batch_size']):
                o = np.random.randint(1, cfg['vocab_size'])
                d = np.random.randint(1, cfg['vocab_size'])
                flag = False
                while not flag:
                    try:
                        start_time = time.time()
                        path = nx.shortest_path(G, source=o-1, target=d-1, weight='weight')
                        end_time = time.time()
                        traj_list.append(np.array(path)+1) # +1是因为数据集是从1开始的
                        time_list.append(end_time - start_time)
                        flag = True
                    except nx.NetworkXNoPath:
                        o = np.random.randint(1, cfg['vocab_size'])
                        d = np.random.randint(1, cfg['vocab_size'])
        if last_batch != 0:
            for j in range(last_batch):
                o = np.random.randint(1, cfg['vocab_size'])
                d = np.random.randint(1, cfg['vocab_size'])
                flag = False
                while not flag:
                    try:
                        start_time = time.time()
                        path = nx.shortest_path(G, source=o-1, target=d-1, weight='weight')
                        end_time = time.time()
                        traj_list.append(np.array(path)+1) # +1是因为数据集是从1开始的
                        time_list.append(end_time - start_time)
                        flag = True
                    except nx.NetworkXNoPath:
                        o = np.random.randint(1, cfg['vocab_size'])
                        d = np.random.randint(1, cfg['vocab_size'])

    else:
        raise ValueError(f"Invalid type selected: {generate_type}")
    
    # 画图
    save_path = f'{save_path}/{generate_type}/image.png'
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    plot_volume2(traj_list, fig_size=20, save_path=save_path)

    return save_path, np.sum(time_list)
        

if __name__ == '__main__':
    task2_test(600, 'pred')
    #! 最短路的并行设计并未实现
    #train_presention()