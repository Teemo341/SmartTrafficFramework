import numpy as np
import torch
import os
import pickle
import argparse

from .DQN import DQNAgent
from dataloader import SmartTrafficDataset, SmartTrafficDataloader, read_node_type
from process_data import read_traj


def train(cfg, dataloader, epochs = 10, log_dir = './log'):
    import time

    memory_len = cfg['memory_len']
    batch_size = cfg['batch_size']  
    n_embd = cfg['n_embd']
    n_head = cfg['n_head']
    n_layer = cfg['n_layer']
    mask_ratio = cfg['mask_ratio']
    wait_quantization = cfg['wait_quantization']
    dropout = cfg['dropout']
    device = cfg['device']

    agent = DQNAgent(memory_len, batch_size, n_layer, n_embd, n_head, wait_quantization, dropout)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(agent.optimizer, epochs)
    agent.model = agent.model.to(device)

    reward_fn = torch.nn.CrossEntropyLoss(reduction='none')

    cross_type = read_node_type('data/simulation/node_type_10*10.csv') # 1,...,100 V
    cross_type = [3 if i == 'C' else 4 for i in cross_type]
    cross_type = torch.tensor(cross_type, dtype=torch.int) # (V,)

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

        for i, wait in enumerate(dataloader):
            iteration_time = time.time()
            wait = wait[:,:,1:,:] # remove the special token, (B, T, V, 7)
            wait = wait.to(device)
            wait[wait<0] = -1 # change the value of the masked position to -1
            full_wait = wait.clone()
            B, T, V, _ = wait.shape
            wait = wait*mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1) - torch.ones_like(wait,dtype=int)*~mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1) # change the value of the masked position to -1
            # wait = wait.int() # (B, T, V, 7)
            wait= wait.int()
            light = torch.randint(0, 7, (B, V), dtype=torch.int) # initial light, (B, V)
            light = light.to(device)

            for time in range(T):
                state = (wait[:,time,:,:], cross_type, light)
                action = agent.act(state[0],state[1],state[2],agent.epsilon) # (B, V)
                # next light
                light = agent.turn_light(cross_type, light, action) # (B, V)
                # current reward
                best_light = agent.best_light(full_wait[:,time,:,:]) # (B, V, 7)
                reward = reward_fn(best_light, action) # (B, V)

                if time == T-1:
                    next_state = (torch.zeros_like(wait[:,time,:,:]), cross_type, light)
                    done = True
                else:
                    next_state = (wait[:,time+1,:,:], cross_type, light)
                    done = False

                agent.remember(state, action, reward, next_state, done)

            loss = agent.replay()
            loss_epoch.append(loss)
            logger_loss.append(loss)
            logger_lr.append(agent.optimizer.param_groups[0]['lr'])

            print(f'Step: {i:<6}  |  Time: {(time.time()-iteration_time)/60:<7.2f}m  |  Loss: {loss.item():<10.8f}  |  LR {agent.optimizer.param_groups[0]["lr"]}')

        lr_scheduler.step()
        with open(os.path.join(log_dir, 'loss.pkl'), 'wb') as f:
            pickle.dump(logger_loss, f)
        with open(os.path.join(log_dir, 'lr.pkl'), 'wb') as f:
            pickle.dump(logger_lr, f)
        
        print(f'Epoch: {epoch:<6}  |  Time: {(time.time()-epoch_time)/60:<7.2f}m  |  Loss {np.mean(loss_epoch)}\n')

        if np.mean(loss_epoch) < best_loss:
            best_loss = np.mean(loss_epoch)
            torch.save(agent.model.state_dict(), os.path.join(log_dir, 'best_model.pth'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--memory_len', type=int, default=2000)
    parser.add_argument('--n_embd', type=int, default=64)
    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--n_layer', type=int, default=6)
    parser.add_argument('--mask_ratio', type=float, default=0.0)
    parser.add_argument('--wait_quantization', type=int, default=10)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--device', type=str, default='cuda:3')

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--log_dir', type=str, default='./log')
    parser.add_argument('--batch_size', type=int, default=4)

    args = parser.parse_args()

    # dataloader
    trajs_edge = read_traj('data/simulation/trajectories_10*10_repeat.csv')
    dataset4 = SmartTrafficDataset(trajs_edge,mode="task4",is_edge=True)
    data_loader4 = SmartTrafficDataloader(dataset4,batch_size=args.batch_size,shuffle=True)

    train(vars(args), data_loader4, epochs = args.epochs, log_dir = args.log_dir)
    print('Training finished!')