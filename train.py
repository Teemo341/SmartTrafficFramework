import numpy as np
import time
import torch
import os
import sys
import pickle
import argparse
import random
from task1.test1 import train as train1
from task2.process_task2 import train as train2
from task3.train_mae import train as train3
from torch.utils.data import SequentialSampler

os.environ['CUDA_LAUNCH_BLOCKING'] = '1' 
sys.path.append('..')
sys.path.append('../data')

from task4.DQN import DQNAgent
from dataloader import SmartTrafficDataset, SmartTrafficDataloader, read_node_type
from process_data import read_traj


def train(agent:DQNAgent, cfg, dataloader, epochs = 10, log_dir = './log'):

    mask_ratio = cfg['mask_ratio']
    wait_quantization = cfg['wait_quantization']
    device = cfg['device']

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(agent.optimizer, epochs)
    agent.model = agent.model.to(device)

    reward_fn = torch.nn.CrossEntropyLoss(reduction='none')

    cross_type = read_node_type('data/simulation/node_type_10*10.csv') # 1,...,100 V
    cross_type = [3 if i == 'C' else 4 for i in cross_type]
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

    for epoch in range(epochs):
        epoch_time = time.time()
        loss_epoch = []
        print(f'Epoch {epoch}\n')
        iteration_time = time.time()

        for i, wait in enumerate(dataloader):
            data_time = time.time()
            wait = wait[:,:,1:,:] # remove the special token, (B, T, V, 7)
            wait = wait.to(device)
            wait = torch.clamp(wait, -1, wait_quantization) # (B, T, V, 7), all negative values become special token, clamp the wait value max to wait_quantization
            full_wait = wait.int().clone()
            B, T, V, _ = wait.shape
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

            loss = agent.replay()
            replay_time = time.time()
            loss_epoch.append(loss)
            logger_loss.append(loss)
            logger_lr.append(agent.optimizer.param_groups[0]['lr'])

            print(f'Step: {i:<6}  |  Loss: {loss.item():<10.8f}  | Random act prob: {agent.epsilon:<10.8f}  |  LR {agent.optimizer.param_groups[0]["lr"]}  |  Data Time: {(data_time-iteration_time)/60:<7.2f}m  |  Preprocess Time: {(preprocess_time-data_time)/60:<7.2f}m  |  Act Time: {(act_time-preprocess_time)/60:<7.2f}m  |  Replay Time: {(replay_time-act_time)/60:<7.2f}m  |  Total Time: {(time.time()-iteration_time)/60:<7.2f}m')
            iteration_time = time.time()

        lr_scheduler.step()
        with open(os.path.join(log_dir, 'loss.pkl'), 'wb') as f:
            pickle.dump(logger_loss, f)
        with open(os.path.join(log_dir, 'lr.pkl'), 'wb') as f:
            pickle.dump(logger_lr, f)
        
        print(f'Epoch: {epoch:<6}  |  Time: {(time.time()-epoch_time)/60:<7.2f}m  |  Loss {np.mean(loss_epoch)}\n')

        if np.mean(loss_epoch) < best_loss:
            print(f'Best Model saved, best loss: {best_loss} -> {np.mean(loss_epoch)}')
            best_loss = np.mean(loss_epoch)
            torch.save(agent.model.state_dict(), os.path.join(log_dir, 'best_model.pth'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--model_read_path', type=str, default=None)
    parser.add_argument('--model_save_path', type=str, default=None)

    parser.add_argument('--memory_len', type=int, default=2000)
    parser.add_argument('--n_embd', type=int, default=64)
    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--n_layer', type=int, default=6)
    parser.add_argument('--mask_ratio', type=float, default=0.0)
    parser.add_argument('--wait_quantization', type=int, default=15)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--memory_device', type=str, default='cuda:0')
    parser.add_argument('--max_len', type=int, default=122)
    parser.add_argument('--vocab_size', type=int, default=181)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--n_hidden', type=int, default=64)
    parser.add_argument('--window_size', type=int, default=1)

    parser.add_argument('--weight_quantization_scale', type=int, default=20, help='task3')
    parser.add_argument('--observe_ratio', type=float, default=0.5, help='task3')
    parser.add_argument('--use_adj_table', type=float, default=True, help='task3')
    parser.add_argument('--special_mask_value', type=float, default=0.0001, help='task3')
    
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--log_dir', type=str, default='./task4/log')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--task_type', type=int, default=0)
    parser.add_argument('--trajs_path', type=str, default='data/jinan/traj_repeat_one_by_one/')
    parser.add_argument('--T', type=int, default=100)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)




    # dataloader
    if args.task_type == 0:

        cfg = vars(args)
        trajs_edge = None
        dataset = SmartTrafficDataset(trajs_edge,mode="task1",trajs_path=cfg['trajs_path'],T=cfg['T'],max_len=cfg['max_len'],need_repeat=False)
        data_loader = SmartTrafficDataloader(dataset,batch_size=args.batch_size,max_len=cfg['max_len'],vocab_size=cfg['vocab_size'],shuffle=False, num_workers=4)
        train_dataloader = data_loader.get_train_data()
        cfg['block_size'] = dataset.T
        train1(cfg, data_loader)

    elif args.task_type == 1:
        cfg = vars(args)
        trajs_node_notrepeat = None
        dataset = SmartTrafficDataset(trajs_node_notrepeat,mode="task2",
                                      trajs_path=cfg['trajs_path'],
                                      adjcent_path='data/jinan/adjcent.npy',
                                      vocab_size=args.vocab_size,T=args.T,max_len=args.max_len)
        
        data_loader = SmartTrafficDataloader(dataset,batch_size=args.batch_size,shuffle=False, num_workers=4)

        cfg['block_size'] = dataset.max_len-cfg['window_size']
        print(cfg)
        train2(cfg, data_loader)

    elif args.task_type == 2:
        cfg = vars(args)
        trajs_node_repeat = None
        dataset = SmartTrafficDataset(trajs_node_repeat,mode="task3",
                                      trajs_path=cfg['trajs_path'],
                                      adjcent_path='data/jinan/adjcent_class.npy',
                                      T=cfg['T'],max_len=cfg['max_len'])
        
        data_loader = SmartTrafficDataloader(dataset,batch_size=args.batch_size,shuffle=False,num_workers=4)
        train_dataloader = data_loader.get_train_data()
        cfg['block_size'] = cfg['T']
        train3(cfg, train_dataloader)

    elif args.task_type == 3:
        trajs_edge = read_traj('data/simulation/trajectories_10*10_repeat.csv')
        dataset4 = SmartTrafficDataset(trajs_edge,mode="task4")
        data_loader4 = SmartTrafficDataloader(dataset4,batch_size=args.batch_size,shuffle=True, num_workers=4)
        agent = DQNAgent(args.device, args.memory_device, args.memory_len, 1, args.n_layer, args.n_embd, args.n_head, args.wait_quantization, args.dropout)
        train(agent, vars(args), data_loader4, epochs = args.epochs, log_dir = args.log_dir)
    else:
        raise ValueError('task_type should be 0, 1, 2, 3')
  
    print('Training finished!')