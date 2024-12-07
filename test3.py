from task3.model_mae import no_diffusion_model_cross_attention_parallel as task3_model
import argparse
import pickle
import random
import numpy as np
import torch
from tqdm import tqdm
from dataloader import SmartTrafficDataset, SmartTrafficDataloader
from task3.utils import calculate_load,transfer_graph,calculate_bounds,read_city
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cv2
import os
import networkx as nx
from train import train3
import sys
import time


weight_path = 'weights/jinan/task3/best_model1.pth'
device = 'cuda:3'
args_path = 'weights/jinan/task3/args.pkl'
#load argparse from pkl
with open(args_path, 'rb') as f:
    args = pickle.load(f)
#for arg in vars(args):
    #print(f"{arg:<30}: {getattr(args, arg)}")



def train_presention(observe_ratio=0.5,device='cuda:3',batch_size=16,epochs=3,lr=0.01):
    # load model
    device = device
    vocab_size = 8909

    n_embd, n_embd, n_layer, n_head, block_size, dropout, use_adj_table, weight_quantization_scale = args.n_embd, args.n_embd, args.n_layer, args.n_head, args.block_size, args.dropout, args.use_adj_table, args.weight_quantization_scale
    model= task3_model(vocab_size, n_embd, n_embd, n_layer, n_head, block_size, dropout, weight_quantization_scale = weight_quantization_scale, use_adj_table=use_adj_table, use_ne=True, use_ge=True, use_agent_mask=False, norm_position='prenorm')

    model.to(device)
    dataset = SmartTrafficDataset(trajs = None,mode="task3",T=60,max_len=120,adjcent_path='data/jinan/adjcent_class.npy',trajs_path='data/jinan/node_traj_test/')
    dataloader = SmartTrafficDataloader(dataset,batch_size=batch_size,shuffle=False, num_workers=4)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=0, last_epoch=-1)
   
    dataloader.randomize_condition(observe_ratio)
    start = 0
    for i in range (start+1, epochs+1):
        model.train()
        # dataloader.randomize_condition(observe_ratio)

        epoch_time = time.time()
        load_data_time = 0
        preprocess_data_time = 0
        forward_time = 0
        backward_time = 0
        special_mask_value = 0.0001

        for condition, time_step, special_mask, adj_table in tqdm(dataloader, desc=f'Train epoch {i:>6}/{epochs:<6}'):
            loss1 = []
            load_data_time += time.time()-epoch_time
            epoch_time = time.time()
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
            condition_ = dataloader.filter_condition(condition) # remove unboservable nodes

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

            preprocess_data_time += time.time()-epoch_time
            epoch_time = time.time()
        
            logits, loss = model(x, condition, adj_table, y, None , None, special_mask_)
            loss = torch.mean(loss)
            loss1.append(loss)

            forward_time += time.time()-epoch_time
            epoch_time = time.time()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            
            backward_time += time.time()-epoch_time
            epoch_time = time.time()

        lr_sched.step()
        print(f'Train epoch {i:>6}/{epochs:<6}|  Loss: {loss.item():<10.8f}  | LR: {lr_sched.get_last_lr()[0]:<10.8f}  | Load data time: {load_data_time/60:.<7.2f}m  |  Preprocess data time: {preprocess_data_time/60:<7.2f}m  |  Forward time: {forward_time/60:<7.2f}m  |  Backward time: {backward_time/60:<7.2f}m  |  Total time: {(load_data_time + preprocess_data_time + forward_time + backward_time)/60:<7.2f}m')
        epoch_time = time.time()
    # for epoch in range(epochs):
    #     for condition, time_step, special_mask, adj_table in tqdm(dataloader, desc='Testing', total=len(dataloader)):
    #         # return trajectory: [B x N x T], special_mask: [B x N x T], adj_table: [B x N x V x 4 x 2]

    #         # random choice a traj as input, the rest as condition
    #         shuffled_indices = torch.randperm(condition.size(1))
    #         condition = condition[:,shuffled_indices,:]
    #         special_mask = special_mask[:,shuffled_indices,:]
    #         adj_table = adj_table[:,shuffled_indices,:,:,:]

    #         # get y, filter trajecotry into condition and get x
    #         condition = condition.to(device)
    #         y = condition[:,0,:] # [B x T]
    #         y = y.long()
    #         # todo try another filter method
    #         condition_ = dataloader.filter_condition(condition) # remove unboservable nodes
    #         x = condition_[:,0,:] # [B x T]
    #         condition = condition[:,1:,:] # [B x N-1 x T]
    #         # condition = None

    #         if use_adj_table:
    #             if isinstance(adj_table, torch.FloatTensor):
    #                 adj_table = adj_table[:,shuffled_indices[0],:,:,:].to(device) # [B x V x 4 x 2]
    #                 adj_table = [adj_table[...,0],adj_table[...,1]] # [B x V x 4], [B x V x 4]
    #             elif isinstance(adj_table, torch.sparse.FloatTensor):
    #                 adj_table = adj_table.to_dense()[:,shuffled_indices[0],:,:,:].to(device) # [B x V x 4 x 2]
    #                 adj_table = [adj_table[...,0].to_sparse(),adj_table[...,1].to_sparse()] # [B x V x 4], [B x V x 4]
    #             else:
    #                 raise ValueError('No adj matrix type should be torch.FloatTensor or torch.sparse.FloatTensor')
    #         else:
    #             raise ValueError('No adj matrix in current version, please use adj table')
            
        
    #         special_mask = special_mask[:,0,:].to(device)
    #         special_mask_ = special_mask.clamp(0,1).float()

    #         logits, loss = model(x, condition, adj_table, y, None, None, special_mask_)
    #         loss = torch.mean(loss)
    #         print(loss)
    





def test_presention(observe_ratio):
    
    #load model
    vocab_size = 8909

    n_embd, n_embd, n_layer, n_head, block_size, dropout, use_adj_table, weight_quantization_scale = args.n_embd, args.n_embd, args.n_layer, args.n_head, args.block_size, args.dropout, args.use_adj_table, args.weight_quantization_scale
    model= task3_model(vocab_size, n_embd, n_embd, n_layer, n_head, 60, 0.1, weight_quantization_scale =30 , use_adj_table=use_adj_table, use_ne=True, use_ge=True, use_agent_mask=False, norm_position='prenorm')
    model.to(device)
    model = model.eval()
    #weights = torch.load(weight_path)
    model.load_state_dict(torch.load(weight_path))
    # load data
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    test_num = 10000
    dataset = SmartTrafficDataset(trajs = None,mode="task3",T=60,max_len=120,adjcent_path='data/jinan/adjcent_class.npy',trajs_path='data/jinan/node_traj_repeat_one_by_one/',vocab_size=vocab_size)
    #dataset = dataset[:100]
    #print(dataset[0])
    sampler = torch.utils.data.SequentialSampler(np.arange(100000))
    dataloader = SmartTrafficDataloader(dataset,batch_size=64,shuffle=False, num_workers=4,sampler=sampler)
    dataloader.randomize_condition(observe_ratio)

    test_trajectory = []
    test_condition = []
    test_adj_table = []
    test_logits = []

    with torch.no_grad():
        for condition, time_step, special_mask, adj_table in tqdm(dataloader, desc='Testing', total=len(dataloader)):
            # return trajectory: [B x N x T], special_mask: [B x N x T], adj_table: [B x N x V x 4 x 2]

            # random choice a traj as input, the rest as condition
            shuffled_indices = torch.randperm(condition.size(1))
            condition = condition[:,shuffled_indices,:]
            special_mask = special_mask[:,shuffled_indices,:]
            adj_table = adj_table[:,shuffled_indices,:,:,:]

            # get y, filter trajecotry into condition and get x
            condition = condition.to(device)
            y = condition[:,0,:] # [B x T]
            y = y.long()
            # todo try another filter method
            condition_ = dataloader.filter_condition(condition) # remove unboservable nodes
            x = condition_[:,0,:] # [B x T]
            condition = condition[:,1:,:] # [B x N-1 x T]
            # condition = None

            if use_adj_table:
                if isinstance(adj_table, torch.FloatTensor):
                    adj_table = adj_table[:,shuffled_indices[0],:,:,:].to(device) # [B x V x 4 x 2]
                    adj_table = [adj_table[...,0],adj_table[...,1]] # [B x V x 4], [B x V x 4]
                elif isinstance(adj_table, torch.sparse.FloatTensor):
                    adj_table = adj_table.to_dense()[:,shuffled_indices[0],:,:,:].to(device) # [B x V x 4 x 2]
                    adj_table = [adj_table[...,0].to_sparse(),adj_table[...,1].to_sparse()] # [B x V x 4], [B x V x 4]
                else:
                    raise ValueError('No adj matrix type should be torch.FloatTensor or torch.sparse.FloatTensor')
            else:
                raise ValueError('No adj matrix in current version, please use adj table')
            
        
            special_mask = special_mask[:,0,:].to(device)
            special_mask_ = special_mask.clamp(0,1).float()

            logits, loss = model(x, condition, adj_table, y, None, None, special_mask_)
            loss = torch.mean(loss)
            
            test_trajectory.append(y.cpu().numpy().astype(np.int32))
            test_condition.append(x.cpu().numpy().astype(np.int32))
            test_adj_table.append(torch.stack(adj_table,dim=-1).cpu().numpy())
            test_logits.append(logits.detach().cpu().numpy())



        test_trajectory = np.concatenate(test_trajectory, axis=0)
        test_condition = np.concatenate(test_condition, axis=0)
        test_adj_table = np.concatenate(test_adj_table, axis=0)
        test_logits = np.concatenate(test_logits, axis=0)

        np.save('data/jinan/test/task3_test_trajectory.npy',test_trajectory)
        np.save('data/jinan/test/task3_test_condition.npy',test_condition)
        np.save('data/jinan/test/task3_test_adj_table.npy',test_adj_table)
        np.save('data/jinan/test/task3_test_logits.npy',test_logits)
        print('save done!')


        vocab_size = 8909

        

        print('load done!')

        e_x = np.exp(test_logits)
        test_logits = e_x / np.sum(e_x, axis=-1, keepdims=True) # [B x T x V]

        print(test_logits.shape)
        print('test_logits done!')

        predic_trajectory = np.argmax(test_logits, axis=-1)

        real_probability = np.eye(vocab_size)[test_trajectory]

        print('real_probability done!')

        methods = ['map', 'single_prob', 'all_prob']
        method = methods[1]

        real_load = calculate_load(real_probability, method) # [T x V]
        predic_load = calculate_load(test_logits, method) # [T x V]
        np.save('data/jinan/test/task3_real_load.npy',real_load)
        np.save('data/jinan/test/task3_predic_load.npy',predic_load)
        print('calculate done!')
        

        real_load_per_time = np.sum(real_load, axis=1)
        predic_load_per_time = np.sum(predic_load, axis=1)
        real_load_per_road = np.sum(real_load, axis=0)
        predic_load_per_road = np.sum(predic_load, axis=0)

        real_load_total = np.sum(real_load)
        predic_load_total = np.sum(predic_load)

        print('trajectory num:', len(real_load))
        print('difference per time:', predic_load_per_time - real_load_per_time)
        print('difference per road:', predic_load_per_road - real_load_per_road)
        print('difference total:', predic_load_total - real_load_total)
        print('mae', np.abs(predic_load - real_load).mean())
        print('mae per time overall:', np.abs(predic_load_per_time - real_load_per_time).mean())
        print('mae per road overall', np.abs(predic_load_per_road - real_load_per_road).mean())

        plt.figure(figsize=(20, 4))
        plt.bar(range(len(real_load_per_time)), real_load_per_time, alpha=0.4, label='real')
        plt.bar(range(len(predic_load_per_time)), predic_load_per_time, alpha=0.4, label='predic')
        plt.legend()
        plt.tight_layout()
        # plt.ylim(0, 100)
        plt.savefig('task3_load_per_time.png')

def plot_presention():

    test_adj_table = np.load('data/jinan/test/task3_test_adj_table.npy')
    real_load = np.load('data/jinan/test/task3_real_load.npy')
    predic_load = np.load('data/jinan/test/task3_predic_load.npy')

    def plot_volume(G, pos, volume_single, max_volume, figure_size=20, save_path=None):
        # G networkx graph
        # pos position of the nodes, get from  read_city('boston')[1]
        # volume_single: V

        x_min, x_max, y_min, y_max = calculate_bounds(pos)
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
        x_min, x_max, y_min, y_max = calculate_bounds(pos)

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

    def make_video_from_frames(frames_dir, video_path):
        # Create a VideoCapture object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        frame = cv2.imread(f'{frames_dir}/frame_{0}.png')
        frame_height, frame_width, _ = frame.shape
        out = cv2.VideoWriter(video_path, fourcc, 2, (frame_width, frame_height))

        # Iterate over all the frames
        for i in range(len(os.listdir(frames_dir))):
            frame = cv2.imread(f'{frames_dir}/frame_{i}.png')
            out.write(frame)

        # Release the VideoCapture object
        out.release()

    # _, pos = read_city('boston')
    pos = read_city('jinan',path='data/')

    for i in pos:
        pos[i] = pos[i][:-1]
    print(pos[0])

    fig_size = 20

    adj_table = test_adj_table[0] #[v,e,2]
    G = transfer_graph(adj_table) # 0-indexed

    volume = real_load[20]
    max_volume = np.max(real_load)/10
    plot_volume(G, pos, volume, max_volume, fig_size,save_path='task3_real_load.png')
    volume = predic_load[20]
    plot_volume(G, pos, volume, max_volume, fig_size,save_path='task3_predic_load.png')

if __name__ == '__main__':
    train_presention(0.5)
    #test_presention(0.5)
    #plot_presention()