import torch
from task1.test1 import define_model as model1
from task2.process_task2 import get_model as model2
from task3.train_mae import define_model as model3
import numpy as np
import matplotlib.pyplot as plt
from process_data import edge_node_trans
import pandas as pd
import ast
from dataloader import simulation2adj,adj_m2adj_l
import networkx as nx
from dataloader import SmartTrafficDataset, SmartTrafficDataloader
from process_data import read_traj
from device_selection import get_local_device

def merge_and_remove_zeros(lst):
    if not lst:
        return []
    
    merged_list = [lst[0]] if lst[0] != 0 else []
    for i in range(1, len(lst)):
        if lst[i] != lst[i - 1]:
            if lst[i] != 0:
                merged_list.append(lst[i])
    while merged_list[-1] == 0:
        merged_list.pop()
    return merged_list

def plot_traj(trajs, picture_path = 'generated_traj.png',
              map_path = 'data/simulation/edge_node_10*10.csv' , 
              nodes_coordinate_path = 'data/simulation/node_type_10*10.csv',
              is_edge = True):
    
    map_ = pd.read_csv(map_path)
    map_ = map_.to_numpy()
    trajs = [merge_and_remove_zeros(traj) for traj in trajs]
    if is_edge:
        for j in range(len(trajs)):
            print(f'edge car{j+1}',trajs[j])
        trajs = [edge_node_trans( map_,traj, is_edge=True) for traj in trajs]
    for j in range(len(trajs)):
        print(f'node car{j+1}',trajs[j])

    nodes = pd.read_csv(nodes_coordinate_path)
    nodes = nodes.to_numpy()

    for i in range(len(trajs)):
        if i >= 10: break
        coords = [ast.literal_eval(nodes[nodes[:,0]==traj][0][2]) for traj in trajs[i]]

        trajectory_x = [coord[0] for coord in coords]
        trajectory_y = [coord[1] for coord in coords]
        traj_shape = len(trajectory_x)
        trajectory_x = np.array(trajectory_x)
        trajectory_y = np.array(trajectory_y)
        perturb = 3e-2
        plt.plot(trajectory_x+perturb*np.random.randn(traj_shape), 
                 trajectory_y+perturb*np.random.randn(traj_shape),
                 marker='o', label=f'Car {i+1}', alpha=0.7)
        plt.scatter(trajectory_x[0], trajectory_y[0], marker='x', color='red')
        plt.scatter(trajectory_x[-1], trajectory_y[-1], marker='x', color='green')

    # Plot configuration
    grid_size = 10
    plt.legend(loc='upper left',bbox_to_anchor=(1, 1))
    plt.xlim(0, grid_size + 1)
    plt.ylim(0, grid_size + 1)
            
    plt.xticks(np.arange(1, grid_size + 1, 1))
    plt.yticks(np.arange(1, grid_size + 1, 1))
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Trajectories from (1, 1) to (10, 10)')
    plt.savefig(picture_path)

def task1_generate_traj(cfg ,ckp = 'weights/model_task1.pth',num = 10,picture_path='task1_generated_traj.png'):
       
    task1_model = model1(cfg)
    task1_model.load_state_dict(torch.load(ckp))
    task1_model.eval()

    od_list = []
    trajs_list = []
    for _ in range(num):
        o,d = np.random.choice(np.arange(1,181),2).tolist()
        od_list.append([o,d])
        trajs = task1_model.generate_traj(o,d)
        trajs_list.append(trajs)

    print('Origin_Destination:',od_list)
    for j in range(len(trajs_list)):
        print(f'car{j+1}',trajs_list[j])
    plot_traj(trajs_list,picture_path)
    #print('Generated Trajectory:',trajs_list)

def task2_generate_traj(cfg ,ckp = 'weights/model_task2_1.pth',num = 10,picture_path='task2_generated_traj.png'):
       
    task2_model = model2(cfg)
    task2_model.load_state_dict(torch.load(ckp))
    task2_model.eval()

    od_list = []
    trajs_list = []

    map_path = 'data/simulation/edge_node_10*10.csv'
    adjacent= simulation2adj(map_path)
    adjacent = adj_m2adj_l(adjacent)
    indices ,values =adjacent[:,:,0],adjacent[:,:,1]
    indices = torch.tensor(indices,dtype=torch.int).to(cfg2['device'])
    values = torch.tensor(values,dtype=torch.float).to(cfg2['device'])
    idx = torch.zeros(1,1,1,dtype=torch.int).to(cfg2['device'])
    condation = torch.zeros(1,1,1,1,dtype=torch.int).to(cfg2['device']) 
    for _ in range(num):
        o,d= np.random.choice(np.arange(1,101),2).tolist()
        idx[0,0,0] = o
        condation[0,0,0,0] = d
        od_list.append([o,d])
        trajs =task2_model.generate(idx,condation,(indices,values))
        trajs_list.append(trajs)

    print('Origin_Destination:',od_list)
    for j in range(len(trajs_list)):
        print(f'car{j+1}',trajs_list[j])
    plot_traj(trajs_list,picture_path,is_edge=False)
    #print('Generated Trajectory:',trajs_list)

def preprocess_node(node_dir):
    #! 0-indexing
    data = pd.read_csv(node_dir)
    pos = {}
    for i in range(len(data)):
        nid = data.loc[i, 'NodeID']
        coordinate = ast.literal_eval(data.loc[i, 'coordinate'])
        lon = coordinate[0]
        lat = coordinate[1]
        pos[int(nid)] = (float(lon), float(lat))
    return pos

def transfer_graph(adj_table):
    # adj_table: B x N x V x 4 x 2, 1-indexing, 0 is special token, 0 is not exist, 1 is exist
    #! G is 0-indexing
    G = nx.DiGraph()
    for i in range(len(adj_table)):
        G.add_node(i)
    for i in range(len(adj_table)):
        for j in range(len(adj_table[i])):
            if adj_table[i,j,1] != 0:
                G.add_edge(i,adj_table[i,j,0]-1,weight=adj_table[i,j,1]) #adj_table is 1-indexing, G is 0-indexing
    return G

def task3_show(real_load,predic_load):
        # plot pure graph
    # from utils import plot_volume
    import matplotlib.colors as mcolors
    import cv2

    def plot_volume(G, pos, volume_single, max_volume, figure_size=20, save_path=None):
        
        # G networkx graph
        # pos position of the nodes, get from  read_city('boston')[1]
        # volume_single: V

        x_min, x_max, y_min, y_max = 0,11,0,11
        fig, ax = plt.subplots(1, 1, figsize=(fig_size, fig_size*(y_max-y_min)/(x_max-x_min)))
        ax.set_facecolor('black')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(np.arange(x_min, x_max, 1))
        ax.set_yticks(np.arange(y_min, y_max, 1))
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
        ax.axvline(x=0, color='k', linestyle='--', linewidth=0.5)

        edge_colors = np.array([(volume_single[int(u)-1]+volume_single[int(v)-1])/2 for u, v in G.edges()])
        edge_colors = edge_colors / max_volume
        print(edge_colors.min(), edge_colors.max())
        edge_colors = plt.cm.RdYlBu(1 - edge_colors)
        # edge_colors = plt.cm.rainbow(edge_colors)
        # colors = ["white","blue", "green", "yellow", "red"]
        # cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors)
        # edge_colors = cmap(edge_colors)

        # nx.draw_networkx_nodes(G, pos, node_size=100, node_color='gray',ax=ax)
        nx.draw_networkx_edges(G, pos, width=figure_size/15, alpha=1, edge_color='black', ax=ax, arrows=False)
        nx.draw_networkx_edges(G, pos, width=figure_size/15, alpha=1, edge_color=edge_colors, ax=ax, arrows=False)

        # ax.legend()
        plt.tight_layout()
        plt.show()
        if save_path != None:
            plt.savefig(save_path)

    def make_volume_frames(G, pos, volume_all, max_volume, figure_size=20, save_path = './videos'):
        # G: networkx graph
        # pos: position of the nodes
        # volume_all: (T, V) where T is time and V is volume
        # max_volume: maximum volume for normalization

        # Calculate bounds for the plot
        x_min, x_max, y_min, y_max = 0, 11, 0, 11

        colors = ["white", "blue", "green", "yellow", "red"]
        cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors)

        for i in range(len(volume_all)):
            volume_single = volume_all[i]
            edge_colors = np.array([(volume_single[int(u)-1] + volume_single[int(v)-1]) / 2 for u, v in G.edges()])
            edge_colors = edge_colors / max_volume
            # edge_colors = cmap(edge_colors)
            edge_colors = plt.cm.RdYlBu(1 - edge_colors)

            # Create a new figure for the current frame
            fig, ax = plt.subplots(figsize=(figure_size, figure_size * (y_max - y_min) / (x_max - x_min)))
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_xticks(np.arange(x_min, x_max, 1))
            ax.set_yticks(np.arange(y_min, y_max, 1))
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
            ax.axvline(x=0, color='k', linestyle='--', linewidth=0.5)

            # Draw nodes and edges
            nx.draw_networkx_edges(G, pos, width=figure_size / 15, alpha=1, edge_color='black', ax=ax, arrows=False)
            nx.draw_networkx_edges(G, pos, width=figure_size / 15, alpha=1, edge_color=edge_colors, ax=ax, arrows=False)

            # Convert the figure to an image and write it to the video
            plt.axis('off')  # Hide axes
            plt.draw()
            plt.tight_layout()
            if save_path != None:
                plt.savefig(f'{save_path}/frame_{i}.png')

            plt.close(fig)  # Close the figure


    # _, pos = read_city('boston')
    pos = preprocess_node('data/simulation/node_type_10*10.csv')

    fig_size = 20

    adjacent= simulation2adj()
    adj_l = adj_m2adj_l(adjacent)

    G = transfer_graph(adj_l) # 0-indexed

    volume = real_load[20]
    max_volume = np.max(real_load)/10
    plot_volume(G, pos, volume, max_volume, fig_size)
    volume = predic_load[20]
    plot_volume(G, pos, volume, max_volume, fig_size)
    # make_volume_frames(G, pos, real_load, max_volume, fig_size, './videos/jinan/real/frames')
    # make_video_from_frames('./videos/jinan/real/frames','./videos/jinan/real/video.mp4')


if __name__ == '__main__':
    cfg1 = {
            "vocab_size": 180+1,
            "device": get_local_device(0),
            "block_size": 122 //2,
            "n_embd": 64,
            "n_head": 4,
            "n_layer": 2,
            "dropout": 0.1,
            "n_hidden": 64,
            "use_agent_mask": True,
            'model_save_path': "weights/model_task1.pth",
            'ta_sliding_window':1,
            'use_model':'sd',
            'use_ne':True
            }
    # cfg2 = {
    #     'device':get_local_device,
    #     'block_size':122 //2, # max length of trajectory
    #     'n_embd':64,
    #     'n_head':4,
    #     'n_layer':2,
    #     'dropout':0.1,
    #     'n_hidden':64,
    #     'n_embed_adj':64,
    #     'vocab_size':100+1,
    #     'window_size':1,
    #     'max_iters':100,
    #     'learning_rate':0.01,
    #     'batch_size':32,
    #     'model_save_path': "weights/model_task2_1.pth"
    #     }
    cfg2 = {
        'device':get_local_device,
        'block_size':21-1, # max length of trajectory
        'n_embd':20,
        'n_head':4,
        'n_layer':8,
        'dropout':0.1,
        'n_hidden':16,
        'n_embed_adj':20,
        'vocab_size':100+1,
        'window_size':1,
        'max_epochs':20,
        'learning_rate':0.001,
        'batch_size':256,
        'model_read_path': None,
        'model_save_path': "weights/model_task2_1.pth"
        }
    cfg3 = {
        'vocab_size':101,
        'n_embd' : 64,
        'n_head' : 4,
        'n_layer' : 2,
        'dropout' : 0.1,
        'device' :get_local_device,
        "block_size":122 //2,
        'weight_quantization_scale': 20,
        'use_adj_table':True,
        'learning_rate':0.001,
        'max_epochs':100,
        'observe_ratio':0.5,
        'special_mask_value':0.0001,
        'model_read_path': "weights/model_task3.pth"

    }
    #task2_generate_traj(cfg2,num=5)
    #task1_generate_traj(cfg1,num=5)
    model = model3(cfg3)
    model.eval()
    trajs = read_traj('data/simulation/trajectories_10*10_repeat_node.csv')
    trajs = trajs[:1000]
    dataset3 = SmartTrafficDataset(trajs,mode="task3",is_edge=False)
    data_loader3 = SmartTrafficDataloader(dataset3,batch_size=32,shuffle=False)
    output = []
    device = cfg3['device']
    use_adj_table = True
    special_mask_value = 0.0001
    data_loader3.randomize_condition(cfg3['observe_ratio'])
    for(j, (condition, time_step, special_mask, adj_table)) in enumerate(data_loader3):
            
            # return trajectory: [B x N x T], time_step: [B x N], special_mask: [B x N x T], adj_table: [B x N x V x 4 x 2]

            # random choice a traj as input, the rest as condition
            shuffled_indices = torch.randperm(condition.size(1))
            condition = condition[:,shuffled_indices,:]
            #time_step = time_step[:,shuffled_indices]
            special_mask = special_mask[:,shuffled_indices,:]

            # get y, filter trajecotry into condition and get x
            condition = condition.to(device)
            y = condition[:,0,:] # [B x T]
            y = y.long()
            # todo try another filter method
            condition_ = data_loader3.filter_condition(condition) # remove unboservable nodes

            x = condition_[:,0,:] # [B x T]
            condition = condition[:,1:,:] # [B x N-1 x T]
            # condition = None

            if use_adj_table:
                if isinstance(adj_table, torch.FloatTensor):
                    #print(shuffled_indices[0])
                    adj_table = adj_table[:,shuffled_indices[0],:,:,:].to(device) # [B x V x 4 x 2]
                    adj_table = [adj_table[...,0],adj_table[...,1]] # [B x V x 4], [B x V x 4]
                elif isinstance(adj_table, torch.sparse.FloatTensor):
                    adj_table = adj_table.to_dense()[:,shuffled_indices[0],:,:,:] # [B x V x 4 x 2]
                    adj_table = [adj_table[...,0].to(device),adj_table[...,1].to(device)] # [B x V x 4], [B x V x 4]
                else:
                    raise ValueError('No adj matrix type should be torch.FloatTensor or torch.sparse.FloatTensor')
            else:
                raise ValueError('No adj matrix in current version, please use adj table')
            
            #time_step = time_step.to(device)
            special_mask = special_mask[:,0,:].to(device)
            special_mask_ = (special_mask+special_mask_value).clamp(0,1).float()

            logits, loss = model(x, condition, adj_table, y, None , None, special_mask_)
            output.append(logits)
    print(output[0])
    print(output[0].shape)

    # task2_model = model2(cfg2)
    # task2_model.load_state_dict(torch.load(cfg2['model_save_path']))    
    # task2_model.eval()

    # x = {
    #         'traj': torch.zeros(1, 20, 1, dtype=torch.int64).to(cfg2['device']),
    #         'cond': torch.zeros(1,20, 1,2, dtype=torch.int64).to(cfg2['device']),
    #         'reagent_mask': torch.ones(1,21,1 ,dtype=torch.int64).to(cfg2['device']),
    #         }
    # x['traj'][0,0,0] = 1
    # x['cond'][:,:,:,0] = 23
    # x['cond'][:,:,:,1] = 23
    # map_path = 'data/simulation/edge_node_10*10.csv'
    # adjacent= simulation2adj(map_path)
    # adjacent = adj_m2adj_l(adjacent)
    # indices ,values =adjacent[:,:,0],adjacent[:,:,1]
    # indices = torch.tensor(indices,dtype=torch.int).to(cfg2['device'])
    # values = torch.tensor(values,dtype=torch.float).to(cfg2['device'])
    # idx = 1
    # i = 0
    # while idx>0 and i<20-1 and idx != 15:
    #     logits, _ = task2_model(x['traj'],None,None,x['cond'],(indices,values))
    #     idx = torch.topk(logits[0,i,0,:],1).indices
    #     x['traj'][0,i+1,0] = idx
    #     i+=1
    # print(x['traj'][0,:,0].detach().cpu().tolist())
    




