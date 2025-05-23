import numpy as np
import time
import torch
import os
import sys
import pickle
import argparse
import random
from tqdm import tqdm
#from ..utils import generate_node_type

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
sys.path.append('..')
sys.path.append('../data')

from .DQN import DQNAgent
from dataloader import SmartTrafficDataset, SmartTrafficDataloader, read_node_type
from process_data import read_traj

print(torch.cuda.is_available())

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

def generate_node_type(adj_l):
    node_type = []
    if isinstance(adj_l,str):
        adj_l = np.load(adj_l)
    if len(adj_l.shape) == 2:
        adj_l = adj_m2adj_l(adj_l)
    for i in range(len(adj_l)-1):
        e = sum([1 for y in adj_l[i] if y[0] != 0])
        if e == 3:
            node_type.append([i,'T'])
        elif e == 4:
            node_type.append([i,'C'])
        # else:
        #     node_type.append(['O'])
    return node_type

def train(agent:DQNAgent, cfg, dataloader, epochs = 1, log_dir = './log'):

    mask_ratio = cfg['mask_ratio']
    wait_quantization = cfg['wait_quantization']
    device = cfg['device']

    agent.model = agent.model.to(device)

    reward_fn = torch.nn.CrossEntropyLoss(reduction='none')

    # if cfg['adjcent'] is not None:
    #     cross_type_ = generate_node_type(cfg['adjcent'])
    # cross_type = [3 if i[1] == 'C' else 4 for i in cross_type_]
    # node_slice = [x[0] for x in cross_type_] 
    # cross_type = torch.tensor(cross_type, dtype=torch.int, device = device) # (V,)

    cross_type_ = read_node_type('/home/shenshiyu/SmartTrafficFramework1/data/simulation/node_type_10*10.csv') # 1,...,100 V
    cross_type = [3 if i == 'C' else 4 for i in cross_type_]
    cross_type = torch.tensor(cross_type, dtype=torch.int, device = device) # (V,)

    if mask_ratio:
        mask_id = np.random.choice(len(cross_type), int(len(cross_type)*mask_ratio), replace=False)+1
        mask = torch.ones(len(cross_type), dtype=torch.int) # (V,)
        mask[mask_id] = False
    else:
        mask = torch.ones(len(cross_type), dtype=torch.int) # (V,)
    mask = mask.to(device)

    logger_lr = []
    logger_loss = []
    best_loss = float('inf')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    with torch.no_grad():
        epoch_time = time.time()
        loss_epoch = []
        iteration_time = time.time()

        for i, wait in tqdm(enumerate(dataloader)):
            # wait = wait[:,:30,node_slice,:]
            data_time = time.time()

            wait = wait[:,:,1:,:] # remove the special token, (B, T, V, 7)
            
            wait = wait.to(device)
            wait = torch.clamp(wait, -1, wait_quantization) # (B, T, V, 7), all negative values become special token, clamp the wait value max to wait_quantization
            full_wait = wait.int().clone()
            B, T, V, _ = wait.shape
            # print(wait.shape,mask.shape)
            wait = wait*mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1) - torch.ones_like(wait,dtype=int)*(1-mask).unsqueeze(0).unsqueeze(0).unsqueeze(-1) # change the value of the masked position to -1
            wait= wait.int()
            light = torch.randint(0, 4, (B, V),device=device)*(cross_type-3).unsqueeze(0) + torch.randint(4,7, (B,V),device=device)*(4-cross_type).unsqueeze(0) # (B, V)
            light = light.int()

            preprocess_time = time.time()

            for t in range(T):
                state = (wait[:,t,:,:], cross_type, light)
                action = agent.act(state[0],state[1],state[2],agent.epsilon) # (B, V)
                # next light
                light = agent.turn_light(cross_type, light, action) # (B, V)
                # current reward
                best_light = agent.best_light(full_wait[:,t,:,:]) # (B, V, 7)
                reward = reward_fn(best_light.view(-1,7),light.view(-1).long()) # (B*V)
                reward = reward.view(B,V) # (B, V)

                if t == T-1:
                    next_state = (torch.zeros_like(wait[:,t,:,:]), cross_type, light)
                    done = True
                else:
                    next_state = (wait[:,t+1,:,:], cross_type, light)
                    done = False

                agent.remember(state, action, reward, next_state, done)
            act_time = time.time()

            loss = agent.replay(optimize=False)
            replay_time = time.time()
            loss_epoch.append(loss)
            logger_loss.append(loss)
            logger_lr.append(agent.optimizer.param_groups[0]['lr'])

            if i >= 5:
                break
        
        print(f' Waiting time: {np.mean(loss_epoch)}\n')

            

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--memory_len', type=int, default=2000)
    parser.add_argument('--n_embd', type=int, default=16)
    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--n_layer', type=int, default=4)
    parser.add_argument('--mask_ratio', type=float, default=0.0)
    parser.add_argument('--wait_quantization', type=int, default=15)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--memory_device', type=str, default='cuda:2')

    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--log_dir', type=str, default='./task4/log')
    parser.add_argument('--batch_size', type=int, default=256)

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # dataloader
    # trajs_edge = read_traj('data/simulation/trajectories_10*10_repeat.csv')
    
    trajs_edge = np.load('data/simulation/task4.npy')
    dataset4 = SmartTrafficDataset(trajs_edge,mode="task4")
    data_loader4 = SmartTrafficDataloader(dataset4,batch_size=args.batch_size,shuffle=True, num_workers=4)

    agent = DQNAgent(args.device, args.memory_device, args.memory_len, args.n_layer, args.n_embd, args.n_head, args.wait_quantization, args.dropout)

    train(agent, vars(args), data_loader4, epochs = args.epochs, log_dir = args.log_dir)

    # torch.save(agent.model.state_dict(), os.path.join(log_dir, 'best_model.pth'))
    agent.model.load_state_dict(torch.load(os.path.join(args.log_dir, 'best_model.pth')))
    train(agent, vars(args), data_loader4, epochs = args.epochs, log_dir = args.log_dir)

    print('Finished!')