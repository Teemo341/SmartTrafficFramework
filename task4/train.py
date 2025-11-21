import numpy as np
import time
import torch
import torch.distributed as dist
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import os
import sys
import pickle
import argparse
import random
#from ..utils import generate_node_type

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
sys.path.append('..')
sys.path.append('../data')

from .DQN import DQNAgent
from dataloader import SmartTrafficDataset, SmartTrafficDataloader, read_node_type
from process_data import read_traj

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    if "A40" in gpu_name or "A100" in gpu_name or "A30" in gpu_name:
        torch.set_float32_matmul_precision('medium') # highest, high, medium
        print(f'device is {gpu_name}, set float32_matmul_precision to medium')

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

def weight_quantization(wait, wait_quantization):
    """
    Quantize the wait time to the nearest multiple of wait_quantization.
    """
    wait_max = wait.max()
    if wait_max > 0:
        wait = wait / wait_max * wait_quantization
    wait[wait < 0] = -1  # Set negative values to -1 (special token)
    wait = torch.round(wait).int()  # Round to the nearest integer
    return wait

def pass_rate(wait, light):
    """wait: (B, V, 7), light: (B, V)"""

    wait[wait < 0] = 0  # Set negative values to 0 (special token)
    no_zero_BV_mask = (wait > 0).any(dim=-1)  # (B, V)
    if no_zero_BV_mask.sum() == 0:
        return None
    light_7 = torch.nn.functional.one_hot(light, num_classes=7).float()  # (B, V, 7)
    val = wait * light_7  # (B, V, 7)
    pass_rate = val.sum(dim=-1) / (wait.sum(dim=-1)+1e-32)  # (B, V)
    pass_rate = torch.sum(pass_rate)/ (no_zero_BV_mask.sum())  # average pass rate over no zero crossings
    return pass_rate

def train(agent:DQNAgent, cfg, dataloader, epochs = 10, log_dir = './log'):

    mask_ratio = cfg['mask_ratio']
    wait_quantization = cfg['wait_quantization']
    device = cfg['device']
    log_dir = cfg['model_save_path'] if cfg['model_save_path'] is not None else log_dir

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(agent.optimizer, epochs)
    agent.model = agent.model.to(device)

    reward_fn = torch.nn.CrossEntropyLoss(reduction='none')
    # if cfg['adjcent'] is not None:
    #     cross_type_ = generate_node_type(cfg['adjcent'])
        #cross_type = read_node_type(cfg['node_type']) # 1,...,100 V
    # cross_type = [3 if i[1] == 'C' else 4 for i in cross_type_]
    # node_slice = [x[0] for x in cross_type_] 
    # cross_type = torch.tensor(cross_type, dtype=torch.int, device = device) # (V,)
    cross_type = read_node_type() # 1,...,100 V
    node_slice = []
    cross_type_ = []
    for i in range(len(cross_type)):
        if cross_type[i] == 'C' or cross_type[i] == 'T':
            node_slice.append(i+1) 
            cross_type_.append(cross_type[i])
    cross_type = [4 if i == 'C' else 3 for i in cross_type_]
    cross_type = torch.tensor(cross_type, dtype=torch.int, device = device) # (V,)

    if mask_ratio:
        mask_id = np.random.choice(len(cross_type), int(len(cross_type)*mask_ratio), replace=False)+1
        mask = torch.ones(len(cross_type), dtype=torch.int) # (V,)
        mask[mask_id] = False
        mask = mask.to(device)
    else:
        mask = None

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
            wait = wait[:,:,node_slice,:]
            data_time = time.time()

            # wait = wait[:,:,1:,:] # remove the special token, (B, T, V, 7)
            
            wait = wait.to(device)
            wait = weight_quantization(wait, wait_quantization) # (B, T, V, 7), all negative values become special token, round the wait value to the nearest integer
            full_wait = wait.int().clone()
            B, T, V, _ = wait.shape
            if mask is not None:
                wait = wait*mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1) - torch.ones_like(wait,dtype=int)*(1-mask).unsqueeze(0).unsqueeze(0).unsqueeze(-1) # change the value of the masked position to -1
            wait= wait.int()
            light = torch.randint(0, 4, (B, V),device=device)*(cross_type-3).unsqueeze(0) + torch.randint(4,7, (B,V),device=device)*(4-cross_type).unsqueeze(0) # (B, V)
            light = light.int()

            preprocess_time = time.time()

            flow_rate_list = []
            for t in range(T-1):
                state = (wait[:,t,:,:], cross_type, light)
                action = agent.act(state[0],state[1],state[2],agent.epsilon) # (B, V)
                # next light
                light = agent.turn_light(cross_type, light, action) # (B, V)
                # current reward
                best_light = agent.best_light(full_wait[:,t,:,:]) # (B, V, 7)
                reward = reward_fn(best_light.view(-1,7),light.view(-1).long()) # (B*V)
                reward = reward.view(B,V) # (B, V)

                next_state = (wait[:,t+1,:,:], cross_type, light)
                if t == T-2:
                    done = True
                else:
                    done = False

                agent.remember(state, action, reward, next_state, done)
                flow_rate = pass_rate(full_wait[:,t,:,:], light)
                if flow_rate is not None:
                    flow_rate_list.append(flow_rate.item())
            
            flow_rate = np.mean(flow_rate_list)
            act_time = time.time()

            loss = agent.replay()
            replay_time = time.time()
            loss_epoch.append(loss)
            logger_loss.append(loss)
            logger_lr.append(agent.optimizer.param_groups[0]['lr'])

            print(f'Step: {i:<6}  |  Loss: {loss.item():<10.8f}  | Random act prob: {agent.epsilon:<10.8f}  |  LR {agent.optimizer.param_groups[0]["lr"]}  |  Data Time: {(data_time-iteration_time)/60:<7.2f}m  |  Preprocess Time: {(preprocess_time-data_time)/60:<7.2f}m  |  Act Time: {(act_time-preprocess_time)/60:<7.2f}m  |  Replay Time: {(replay_time-act_time)/60:<7.2f}m  |  Total Time: {(time.time()-iteration_time)/60:<7.2f}m  | Flow Rate:{flow_rate:<10.8f}')
            iteration_time = time.time()

        lr_scheduler.step()
        with open(os.path.join(log_dir, 'loss.pkl'), 'wb') as f:
            pickle.dump(logger_loss, f)
        with open(os.path.join(log_dir, 'lr.pkl'), 'wb') as f:
            pickle.dump(logger_lr, f)
        
        loss_epoch = torch.tensor(loss_epoch)
        print(f'Epoch: {epoch:<6}  |  Time: {(time.time()-epoch_time)/60:<7.2f}m  |  Loss {torch.mean(loss_epoch)}\n')

        if torch.mean(loss_epoch) < best_loss:
            print(f'Best Model saved, best loss: {best_loss} -> {torch.mean(loss_epoch)}')
            best_loss = torch.mean(loss_epoch)
            torch.save(agent.model.state_dict(), os.path.join(log_dir, 'best_model.pth'))
            
class lightning_wrapper(pl.LightningModule):
    """
    This class is used to integrate the DQN agent with PyTorch Lightning for training.
    """
    def __init__(self, agent:DQNAgent, cfg, dataloader, epochs = 10, log_dir = './log'):
        super(lightning_wrapper, self).__init__()
        self.agent = agent
        self.cfg = cfg
        self.dataloader = dataloader
        self.epochs = epochs
        self.log_dir = log_dir
        self.mask_ratio = cfg['mask_ratio']
        self.wait_quantization = cfg['wait_quantization']
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.agent.optimizer, epochs)
        self.automatic_optimization = False
        self.reward_fn = torch.nn.CrossEntropyLoss(reduction='none')
        self.cross_type, self.node_slice, self.mask = self.make_cross_type()
        
    def make_cross_type(self):
        cross_type = read_node_type()  # 1,...,100 V
        node_slice = []
        cross_type_ = []
        for i in range(len(cross_type)):
            if cross_type[i] == 'C' or cross_type[i] == 'T':
                node_slice.append(i+1) 
                cross_type_.append(cross_type[i])
        cross_type = [4 if i == 'C' else 3 for i in cross_type_]
        cross_type = torch.tensor(cross_type, dtype=torch.int)  # (V,)
        
        if self.mask_ratio:
            mask_id = np.random.choice(len(cross_type), int(len(cross_type)*self.mask_ratio), replace=False)+1
            mask = torch.ones(len(cross_type), dtype=torch.int) # (V,)
            mask[mask_id] = False
        else:
            mask = None

        return cross_type, node_slice, mask

    def configure_optimizers(self):
        return None
    
    def on_train_start(self):
        self.agent.device = self.device
        self.cross_type = self.cross_type.to(self.device)

    def sync_model_parameters(self, model):
        if dist.is_initialized():
            for param in model.parameters():
                dist.all_reduce(param.data, op=dist.ReduceOp.AVG)  # 聚合所有进程的模型参数
        
    def training_step(self, batch, batch_idx):

        wait = batch[:,:,self.node_slice,:]
        wait = weight_quantization(wait, self.wait_quantization)  # (B, T, V, 7), all negative values become special token, round the wait value to the nearest integer
        full_wait = wait.int().clone()
        B, T, V, _ = wait.shape
        if self.mask is not None:
            wait = wait * self.mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1) - torch.ones_like(wait, dtype=int) * (1 - self.mask).unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # change the value of the masked position to -1
        wait = wait.int()
        light = torch.randint(0, 4, (B, V), device = self.device) * (self.cross_type - 3).unsqueeze(0) + torch.randint(4, 7, (B, V), device = self.device) * (4 - self.cross_type).unsqueeze(0)  # (B, V)
        light = light.int().to(self.device)  # (B, V)

        flow_rate = 0
        no_zero_num = 0
        for t in range(T-1):
            state = (wait[:,t,:,:], self.cross_type, light)
            # print(self.agent.device, state[0].device, state[1].device, state[2].device)
            # break
            action = self.agent.act(state[0], state[1], state[2], self.agent.epsilon)
            # next light
            light = self.agent.turn_light(self.cross_type, light, action)  # (B, V)
            # current reward
            best_light = self.agent.best_light(full_wait[:,t,:,:])
            reward = self.reward_fn(best_light.view(-1, 7), light.view(-1).long())  # (B*V)
            reward = reward.view(B, V)  # (B, V)
            next_state = (wait[:,t+1,:,:], self.cross_type, light)
            if t == T-2:
                done = True
            else:
                done = False

            self.agent.remember(state, action, reward, next_state, done)
            flow_rate_ = pass_rate(full_wait[:,t,:,:], light)
            if flow_rate_ is not None:
                flow_rate += flow_rate_
                no_zero_num += 1
        flow_rate = flow_rate / no_zero_num # average flow rate over the time steps
        loss = self.agent.replay()
        loss = loss.float()

        if dist.is_initialized():
            self.sync_model_parameters(self.agent.model)

        self.log('train_loss', loss.item(), on_step=True, on_epoch=True, prog_bar=True, sync_dist = True)
        self.log('train_flow_rate', flow_rate.item(), on_step=True, on_epoch=True, prog_bar=True, sync_dist = True)
        self.log('train_random_act_prob', self.agent.epsilon, on_step=True, on_epoch=True, prog_bar=True, sync_dist = True)
        self.log('train_lr', self.agent.optimizer.param_groups[0]['lr'], on_step=True, on_epoch=True, prog_bar=True, sync_dist = True)
        return loss

    def on_train_epoch_end(self):
        super().on_train_epoch_end()
        self.lr_scheduler.step()

class MyCheckpoint(ModelCheckpoint):

    def _save_checkpoint(self, trainer: pl.Trainer, file_path: str):
        model = trainer.lightning_module.agent.model
        if trainer.is_global_zero:
            checkpoint_path = os.path.join(self.dirpath, 'pl_checkpoint', f'{trainer.current_epoch:03d}.pth')
            if not os.path.exists(os.path.dirname(checkpoint_path)):
                os.makedirs(os.path.dirname(checkpoint_path))
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at epoch {trainer.current_epoch} to {checkpoint_path}")
    
    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)
        self._save_checkpoint(trainer, self.dirpath)
        


def train_pl(agent:DQNAgent, cfg, dataloader, epochs = 10, log_dir = './log'):
    """
    Train the DQN agent using PyTorch Lightning.
    """
    model = lightning_wrapper(agent, cfg, dataloader, epochs, log_dir)
    checkpoint_callback = MyCheckpoint(
        dirpath=log_dir,
        filename="pl_model_{epoch:02d}",
        monitor='train_loss',
        mode='min',
        every_n_epochs=1,
        verbose=True,
    )

    logger = pl.loggers.TensorBoardLogger(
        save_dir=log_dir,
        name='lightning_logs',
        default_hp_metric=False
    )
    
    trainer = pl.Trainer(
        max_epochs=epochs,
        callbacks=[checkpoint_callback],
        precision="32",
        enable_progress_bar=True,
        accelerator="gpu",
        devices=-1,  # Use all available GPUs
        strategy="ddp",
        logger=logger,
        log_every_n_steps=1,
        sync_batchnorm=True,
    )
    
    trainer.fit(model, dataloader)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--memory_len', type=int, default=2000)
    parser.add_argument('--n_embd', type=int, default=128)
    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--n_layer', type=int, default=8)
    parser.add_argument('--mask_ratio', type=float, default=0.0)
    parser.add_argument('--wait_quantization', type=int, default=20)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--device', type=str, default='cuda:2')
    parser.add_argument('--memory_device', type=str, default='cpu')

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--log_dir', type=str, default='./task4/log')
    parser.add_argument('--batch_size', type=int, default=256)

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # dataloader
    # trajs_edge = read_traj('data/simulation/trajectories_10*10_repeat.csv')
    # trajs_edge = np.load('data/simulation/task4_data.npy')
    # trajs_edge = './data/simulation/task4_data_one_by_one'
    # dataset4 = SmartTrafficDataset(trajs_edge,mode="task4",task4_num=1000)
    
    #如果比较慢换下边代码
    trajs_edge = './data/simulation/new_task4_data_one_by_one'
    dataset4 = SmartTrafficDataset(trajs_edge,mode="task4",task4_num=1)
    print(f'len(dataset4): {len(dataset4)}')

    data_loader4 = SmartTrafficDataloader(dataset4,batch_size=args.batch_size,shuffle=True, num_workers=4)

    # agent = DQNAgent(args.device, args.memory_device, args.memory_len, args.n_layer, args.n_embd, args.n_head, args.wait_quantization, args.dropout)
    # train(agent, vars(args), data_loader4, epochs = args.epochs, log_dir = args.log_dir)

    agent = DQNAgent('cpu', args.memory_device, args.memory_len, args.n_layer, args.n_embd, args.n_head, args.wait_quantization, args.dropout)
    train_pl(agent, vars(args), data_loader4, epochs = args.epochs, log_dir = args.log_dir)
    print('Training finished!')