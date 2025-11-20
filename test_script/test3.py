# 将 task3 目录加入环境变量与 sys.path
import os
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
TASK3_PATH = os.path.join(PROJECT_ROOT, 'task3')
if TASK3_PATH not in sys.path:
    sys.path.insert(0, TASK3_PATH)
os.environ['PYTHONPATH'] = TASK3_PATH + os.pathsep + os.environ.get('PYTHONPATH', '')

from task3.model_mae import no_diffusion_model_cross_attention_parallel as task3_model
import argparse
import pickle
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import pytorch_lightning as pl
from tqdm import tqdm
from dataloader import SmartTrafficDataset, SmartTrafficDataloader
from utils import calculate_load,transfer_graph,calculate_bounds,read_city
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cv2
import networkx as nx
from train import train3
import time


weight_path = 'weights/best/jinan/task3/best_model.pth'
device = 'cuda:3'
args_path = 'weights/best/jinan/task3/args.pkl'
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
                    # print(adj_table.shape)
                    # print(shuffled_indices[0])
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


def test_presentation(num, observe_ratio, save_path=None):
    # Load model
    vocab_size = 8909
    n_embd, n_embd, n_layer, n_head, block_size, dropout, use_adj_table, weight_quantization_scale = args.n_embd, args.n_embd, args.n_layer, args.n_head, args.block_size, args.dropout, args.use_adj_table, args.weight_quantization_scale
    model = torch.load(weight_path)
    model.to(device)
    model.eval()

    # Load data
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    dataset = SmartTrafficDataset(
        trajs=None, mode="task3", T=60, max_len=120,
        adjcent_path='data/jinan/adjcent_class.npy',
        trajs_path='data/jinan/node_traj_repeat_one_by_one/',
        vocab_size=vocab_size
    )
    
    # Choose num random samples from dataset
    indices = np.random.choice(len(dataset), size=num, replace=False)
    sampler = torch.utils.data.SubsetRandomSampler(indices)
    dataloader = SmartTrafficDataloader(dataset, batch_size=160, num_workers=4, sampler=sampler)
    dataloader.randomize_condition(0.5)

    real_volumes = 0
    pred_volumes = 0

    with torch.no_grad():
        for condition, time_step, special_mask, adj_table in tqdm(dataloader, desc='Testing', total=len(dataloader)):
            # Process trajectory: [B x N x T], special_mask: [B x N x T], adj_table: [B x N x V x 4 x 2]

            # Randomly choose a traj as input, the rest as condition
            adj_table = adj_table.repeat(1, condition.size(1), 1, 1, 1)
            shuffled_indices = torch.randperm(condition.size(1))
            condition = condition[:, shuffled_indices, :]
            adj_table = adj_table[:, shuffled_indices, :, :, :]

            # Get y, filter trajectory into condition and get x
            condition = condition.to(device)
            y = condition[:, 0, :]  # [B x T]
            
            condition_ = dataloader.filter_condition(condition)  # Remove unobservable nodes
            x = condition_[:, 0, :]  # [B x T]
            condition = condition[:, 1:, :]  # [B x N-1 x T]
            
            if use_adj_table:
                if isinstance(adj_table, torch.FloatTensor):
                    adj_table = adj_table[:, shuffled_indices[0], :, :, :].to(device)  # [B x V x 4 x 2]
                    adj_table = [adj_table[..., 0], adj_table[..., 1]]  # [B x V x 4], [B x V x 4]
                elif isinstance(adj_table, torch.sparse.FloatTensor):
                    adj_table = adj_table.to_dense()[:, shuffled_indices[0], :, :, :].to(device)  # [B x V x 4 x 2]
                    adj_table = [adj_table[..., 0].to_sparse(), adj_table[..., 1].to_sparse()]  # [B x V x 4], [B x V x 4]
                else:
                    raise ValueError('Adj matrix should be torch.FloatTensor or torch.sparse.FloatTensor')
            else:
                raise ValueError('No adj matrix in current version, please use adj table')
            
            # Get logits
            logits, _ = model(x, condition, adj_table, None, None, None, None)
            
            # Apply softmax
            logits = torch.softmax(logits, dim=-1)  # [B x T x V]
            pred_load = calculate_load(logits, 'single_prob')  # [T x V]
            real_prob = F.one_hot(y, num_classes=logits.size(-1))  # [B x T x V]
            real_load = calculate_load(real_prob, 'single_prob')  # [T x V]

            # Aggregate real and predicted volumes
            real_volumes += real_load.cpu().numpy()
            pred_volumes += pred_load.detach().cpu().numpy()

        methods = ['map', 'single_prob', 'all_prob']

        print('make video')
        real_load = real_volumes
        predic_load = pred_volumes
        real_path,pred_path =plot_presention(real_load,predic_load,observe_ratio,save_path=save_path)
        
    return  real_path ,pred_path

class lightning_wrapper(pl.LightningModule):
    def __init__(self, model, dataloader, use_adj_table, observe_ratio):
        super(lightning_wrapper, self).__init__()
        self.model = model
        self.dataloader = dataloader
        self.use_adj_table = use_adj_table
        self.observe_ratio = observe_ratio
        self.real_volumes = 0
        self.pred_volumes = 0
        self.real_path = None
        self.pred_path = None

    def forward(self, batch):
        condition, time_step, special_mask, adj_table = batch
        # Process trajectory: [B x N x T], special_mask: [B x N x T], adj_table: [B x N x V x 4 x 2]

        # Randomly choose a traj as input, the rest as condition
        adj_table = adj_table.repeat(1, condition.size(1), 1, 1, 1)
        shuffled_indices = torch.randperm(condition.size(1))
        condition = condition[:, shuffled_indices, :]
        adj_table = adj_table[:, shuffled_indices, :, :, :]

        # Get y, filter trajectory into condition and get x
        condition = condition
        y = condition[:, 0, :]  # [B x T]
        
        condition_ = self.dataloader.filter_condition(condition)  # Remove unobservable nodes
        x = condition_[:, 0, :]  # [B x T]
        condition = condition[:, 1:, :]  # [B x N-1 x T]
        
        if self.use_adj_table:
            if isinstance(adj_table, torch.sparse.FloatTensor):
                adj_table = adj_table.to_dense()[:, shuffled_indices[0], :, :, :]
                adj_table = [adj_table[..., 0].to_sparse(), adj_table[..., 1].to_sparse()]
            else:
                adj_table = adj_table[:, shuffled_indices[0], :, :, :]  # [B x V x 4 x 2]
                adj_table = [adj_table[..., 0], adj_table[..., 1]]  # [B x V x 4], [B x V x 4]
        else:
            raise ValueError('No adj matrix in current version, please use adj table')
        
        # Get logits
        logits, _ = self.model(x, condition, adj_table, None, None, None, None)
        
        # Apply softmax
        logits = torch.softmax(logits, dim=-1)  # [B x T x V]
        pred_load = calculate_load(logits, 'single_prob')  # [T x V]
        real_prob = F.one_hot(y, num_classes=logits.size(-1))  # [B x T x V]
        real_load = calculate_load(real_prob, 'single_prob')  # [T x V]

        return real_load, pred_load

    
    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            real_load, pred_load = self(batch)
        self.real_volumes += real_load.detach()
        self.pred_volumes += pred_load.detach()
    
    def on_test_epoch_end(self):
        # Synchronize results across GPUs (multi-GPU setup)
        if dist.is_initialized() and dist.get_world_size() > 1:
            # Sync real_volumes across all devices
            dist.all_reduce(self.real_volumes, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.pred_volumes, op=dist.ReduceOp.SUM)

        # Aggregate results on CPU
        real_volumes = self.real_volumes.cpu().numpy()
        pred_volumes = self.pred_volumes.cpu().numpy()

        # Only let the main process handle these operations
        if dist.get_rank() == 0:  # Check if we are on the main process
            print('make video')
            real_load = real_volumes
            predic_load = pred_volumes
            real_path, pred_path = plot_presention(real_load, predic_load, self.observe_ratio, save_path='./UI_element/task2')
        
            self.real_path = real_path
            self.pred_path = pred_path
            print('video done')

def test_presentation_lightning(num, observe_ratio, save_path=None):
    torch.set_float32_matmul_precision('medium')
    # Load model
    vocab_size = 8909
    n_embd, n_embd, n_layer, n_head, block_size, dropout, use_adj_table, weight_quantization_scale = args.n_embd, args.n_embd, args.n_layer, args.n_head, args.block_size, args.dropout, args.use_adj_table, args.weight_quantization_scale
    model = torch.load(weight_path)
    model.eval()

    # Load data
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    dataset = SmartTrafficDataset(
        trajs=None, mode="task3", T=60, max_len=120,
        adjcent_path='data/jinan/adjcent_class.npy',
        trajs_path='data/jinan/node_traj_repeat_one_by_one/',
        vocab_size=vocab_size
    )
    
    # Choose num random samples from dataset
    indices = np.random.choice(len(dataset), size=num, replace=False)
    sampler = torch.utils.data.SubsetRandomSampler(indices)
    dataloader = SmartTrafficDataloader(dataset, batch_size=160, num_workers=4, sampler=sampler)
    dataloader.randomize_condition(0.5)

    # Create a Lightning wrapper for the model
    lightning_model = lightning_wrapper(model=model, dataloader=dataloader, use_adj_table=use_adj_table, observe_ratio=observe_ratio)

    # Enable DDP for multi-GPU testing
    trainer = pl.Trainer(
        max_epochs=1,
        precision="16-mixed",
        enable_progress_bar=True,
        accelerator="gpu",
        devices=-1,  # Use all available GPUs
        strategy="ddp"
    )

    # Run test
    trainer.test(lightning_model, dataloaders=dataloader)

    if dist.get_rank() == 0:
        real_path = lightning_model.real_path
        pred_path = lightning_model.pred_path
        return real_path, pred_path
    else:
        # In case of non-main processes, return None
        return None, None


def plot_presention(real_load,predic_load,observe_ratio,save_path=None):

    if save_path is None:
        save_path = './UI_element/task2'

    adj_table = np.load('data/jinan/adj_l.npy')
    G = transfer_graph(adj_table) # 0-indexed

    # _, pos = read_city('boston')
    _, pos = read_city('jinan',path='data/')
    for i in pos:
        # print(i)
        pos[i] = pos[i][:-1]

    fig_size = 20

    max_volume = np.max(real_load)/10
    real_path = make_volume_video(G, pos, real_load, max_volume, observe_ratio, fig_size=fig_size, save_video_path = f'{save_path}/videos/real')
    pred_path = make_volume_video(G, pos, predic_load, max_volume, observe_ratio, fig_size=fig_size, save_video_path = f'{save_path}/videos/pred')

    return real_path, pred_path

def make_volume_video(G, pos, volume_all, max_volume, observe_ratio, fig_size=20, save_video_path = None):
    # G: networkx graph
    # pos: position of the nodes
    # volume_all: (T, V) where T is time and V is volume
    # max_volume: maximum volume for normalization

    if save_video_path is None:
        save_video_path = './UI_element/task2/pred'
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Calculate bounds for the plot
    x_min, x_max, y_min, y_max = calculate_bounds(pos)

    colors = ["white", "blue", "green", "yellow", "red"]
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors)
    nodes = G.nodes()
    observed_nodes = np.random.choice(nodes, int(len(nodes) * observe_ratio*0.1), replace=False)
    for i in tqdm(range(len(volume_all))):
        volume_single = volume_all[i]
        edge_colors = np.array([(volume_single[int(u)-1] + volume_single[int(v)-1]) / 2 for u, v in G.edges()])
        edge_colors = edge_colors / max_volume
        edge_colors = cmap(edge_colors)
        # edge_colors = plt.cm.RdYlBu(1 - edge_colors)

        # Create a new figure for the current frame
        fig, ax = plt.subplots(1, 1, figsize=(fig_size, fig_size * (y_max - y_min) / (x_max - x_min)))
        ax.set_facecolor('black')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(np.arange(x_min, x_max, 1))
        ax.set_yticks(np.arange(y_min, y_max, 1))
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
        ax.axvline(x=0, color='k', linestyle='--', linewidth=0.5)

        # Draw nodes and edges
        # nx.draw_networkx_edges(G, pos, width=fig_size / 15, alpha=1, edge_color='black', ax=ax, arrows=False)
        nx.draw_networkx_edges(G, pos, width=fig_size / 15, alpha=1, edge_color=edge_colors, ax=ax, arrows=False)

        # Draw observed nodes
        nx.draw_networkx_nodes(G, pos, nodelist=observed_nodes, node_size=15, node_color='none', edgecolors='red', linewidths=1.5, ax=ax)
        
        # Convert the figure to an image and write it to the video
        plt.tight_layout()
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        frame = frame[:,:,1:]
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        plt.close()

        # 2. 初始化 VideoWriter（如果是第一帧）
        if i == 0:
            frame_height, frame_width, _ = frame.shape
            out = cv2.VideoWriter(f'{save_video_path}/video.mp4', fourcc, 2, (frame_width, frame_height))
        out.write(frame)

    # Release the VideoWriter object
    out.release()
    return f'{save_video_path}/video.mp4'

if __name__ == '__main__':
    #train_presention(0.5)
    # test_presentation(10000,0.5,save_path='./UI_element/task2')
    # real_path,pred_path = test_presentation_lightning(10000,0.5,save_path='./UI_element/task2')
    # plot_presention()


    args_ = argparse.ArgumentParser()
    args_.add_argument('--num', type=int, default=10000, help='number of samples')
    args_.add_argument('--observe_ratio', type=float, default=0.5, help='observation ratio')
    args_.add_argument('--save_path', type=str, default='./UI_element/task2', help='save path')

    args_ = args_.parse_args()
    real_path, pred_path = test_presentation_lightning(args_.num, args_.observe_ratio, args_.save_path)
    print('test3 success')