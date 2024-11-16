import time
from task1.dataset import preprocess_edge,preprocess_node,preprocess_traj
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from task1.util import get_model
from task1.ma_model import SpatialTemporalMultiAgentModel
import torch.optim as optim
from tqdm import tqdm
import os

def decide_time_type(trajs):
     time_list = []
     for traj in trajs[0:100]:
         for i in range(len(traj['Points'])-1):
             time = traj['Points'][i+1][1] - traj['Points'][i][1]
             time_list.append(time)
     max_time = np.max(time_list)
     min_time = np.min(time_list)
     time_list = np.array(time_list)
     x,y,z = np.sum(time_list[time_list<10]),np.sum(time_list[(time_list<60) & (time_list>=10)]),np.sum(time_list[time_list>60])
     time_type = 1 if x == max(x, y, z) else (10 if y == max(x, y, z) else 60)
     return time_type,max_time,min_time



class task1Dataset(Dataset):
    #trajs:data of trajs [node_id,timestamp]
    #T:train window size
    #with_time:whether to use time information--repeat the node_id by time
    #time_type:time type(1,10,60,3600)
    def __init__(self,trajs,with_time=False,time_type:int=1):
        super(task1Dataset,self).__init__()
        self.trajs = trajs
        self.with_time = with_time
        self.time_type = time_type

    def __len__(self):
        return len(self.trajs)
    
    def __getitem__(self,idx):
        if self.with_time:
            traj = []
            for i in range(len(self.trajs[idx]['Points'])-1):
                time = self.trajs[idx]['Points'][i+1][1] - self.trajs[idx]['Points'][i][1]
                traj.append([self.trajs[idx]['Points'][i][0]]*int(1+time//self.time_type))
            traj.append([self.trajs[idx]['Points'][-1][0]])
            traj = np.concatenate(traj,axis=0)
            points =  torch.tensor(traj,dtype=torch.int64)
            od = torch.tensor([self.trajs[idx]['Points'][0][0],self.trajs[idx]['Points'][-1][0]],dtype=torch.int64)
            return {'points':points,'od':od}
        
        return {'points':torch.tensor([int(x[0]) for x in self.trajs[idx]['Points']],dtype=torch.int64),
                'od':torch.tensor([self.trajs[idx]['Points'][0][0],self.trajs[idx]['Points'][-1][0]],dtype=torch.int64)}


def _padding_zero(traj,max_len):
        # return traj + [0]*(max_len-len(traj))
        return np.concatenate([traj,np.zeros((max_len-len(traj),),dtype=np.int64)],axis=0)

def create_collate_fn(max_len,T=-1,window_size=1):
    def task1_collate_fn(batches):
            # 'traj':, # [B,T,1]
            # 'cond':cond, # [B, T, 1, 2]
            # 'agent_mask':agent_mask, # [B,T,1]
            B = len(batches)
            len_list = [batch['points'].shape[0] for batch in batches]
            traj_list = [torch.tensor(_padding_zero(batch['points'],max_len)) for batch in batches]
            cond_list = [torch.tensor(batch['od'].clone().detach()) for batch in batches]
            agent_mask = [torch.tensor([1]*len_list[i]+[0]*(max_len-len_list[i])) for i in range(len(len_list))]
            traj = torch.stack(traj_list).unsqueeze(-1)
            traj_targ = torch.stack(traj_list).unsqueeze(-1)
            cond = torch.stack(cond_list).unsqueeze(1).unsqueeze(1).repeat(1,max_len,1,1)
            reagent_mask = torch.stack(agent_mask).reshape(B,max_len,-1)
            if T > 0:
                if T + window_size > max_len:
                    raise ValueError("T+window_size should be less than max_len")
                index = np.random.randint(0,max_len-T+1)
                return {'traj':traj[:,index:index+T,:],'traj_targ':traj_targ[:,index+window_size:index+window_size+T,:],
                        'cond':cond[:,index:index+T,:,:],'reagent_mask':reagent_mask[:,index:index+T,:]}
           
            return {'traj':traj,'traj_targ':traj_targ,'cond':cond,'reagent_mask':reagent_mask}
    return task1_collate_fn


    
class task1DataLoader(DataLoader):
    def __init__(self,dataset,batchsize,collate_fn,shuffle=True,**kwargs):
        super(task1DataLoader,self).__init__(dataset,batch_size=batchsize,collate_fn=collate_fn,shuffle=shuffle,**kwargs)
    


def define_model(cfg):
     device = cfg['device']
     block_size, n_embd, n_head, n_layer, dropout = cfg['block_size'], cfg['n_embd'], cfg['n_head'], cfg['n_layer'], cfg['dropout']
     n_hidden = cfg['n_hidden']
     vocab_size = cfg['vocab_size']
     model = SpatialTemporalMultiAgentModel(   vocab_size, 
                                               n_embd,
                                               n_hidden, 
                                               n_layer,
                                               n_head,
                                               block_size=block_size,
                                               dropout=dropout,
                                               use_agent_mask=True,
                                               )
     return model.to(device)

def train(cfg, dataloader):
    """
    训练模型的函数.

    参数:
    - model: 训练的模型
    - dataloader: 数据加载器
    - criterion: 损失函数
    - optimizer: 优化器
    - num_epochs: 训练的轮数
    - device: 计算设备 (如 'cuda' 或 'cpu')

    返回:
    - 每个epoch的损失列表
    """
    device = cfg['device']
    num_epochs = cfg['epochs']
    model = define_model(cfg)
    old_path = None
    if cfg['model_read_path']:
        model.load_state_dict(torch.load(cfg['model_read_path']))
        if 'best_model' in cfg['model_read_path'] or 'last_model' in cfg['model_read_path']:
            last_loss = float(cfg['model_read_path'][-10:-4])
            old_path = cfg['model_read_path']
    else:
        last_loss = 10000
    model.train() 
    epoch_losses = []
    optimizer = optim.Adam(model.parameters(), lr=cfg['learning_rate'])
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4000, gamma=0.99)
    for epoch in range(num_epochs):
        running_loss = 0.0
        #
        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            

            inputs = {key: value.to(device) for key, value in batch.items() if isinstance(value, torch.Tensor)}
        
            optimizer.zero_grad()
            _, loss = model(inputs)
            state_loss, ratio_loss = loss[0].mean(), loss[1].mean()
            sl_loss:torch.Tensor = (state_loss + ratio_loss*1 ) 
            
            
            
            kl_loss = torch.tensor(0.0,device=device)

            
            total_loss:torch.Tensor = (sl_loss + 0.001 * kl_loss)/ 1 #grad_accumulation
            
            total_loss.backward()
            optimizer.step()
            lr_scheduler.step()
            running_loss += total_loss.item()

        # 计算每个epoch的平均损失
        avg_loss = running_loss / len(dataloader)
        epoch_losses.append(avg_loss)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f},lr:{optimizer.param_groups[0]['lr']:.6f}")
        if os.path.isdir(cfg['model_save_path']):
            path = os.path.join(cfg['model_save_path'],f"best_model_{avg_loss:.4f}.pth")
            if avg_loss < last_loss:
                if old_path:
                    os.remove(old_path)
                last_loss = avg_loss
                torch.save(model.state_dict(), path)
                old_path = path

    if os.path.isdir(cfg['model_save_path']):
        path = os.path.join(cfg['model_save_path'],f"last_model_{avg_loss:.4f}.pth")
        torch.save(model.state_dict(), path)

    return epoch_losses

if "__main__" == __name__:

    filepath_edge = "../data/jinan/edge_jinan.csv"
    filepath_node = "../data/jinan/node_jinan.csv"
    filepath_traj = "../data/jinan/traj_jinan.csv"


    '''
    filepath_node文件的列名如下
    Index(['NodeID', 'Longitude', 'Latitude', 'HasCamera'], dtype='object')
    filepath_edge文件的列名如下
    Index(['Origin', 'Destination', 'Class', 'Geometry', 'Length'], dtype='object')
    filepath_traj文件的列名如下
    Index(['VehicleID', 'TripID', 'Points', 'DepartureTime', 'Duration', 'Length'], dtype='object')
    '''

    edges = preprocess_edge(filepath_edge)
    pos = preprocess_node(filepath_node)
    trajs = preprocess_traj(filepath_traj)
    print(trajs[0])
    max_len = max([len(traj['Points']) for traj in trajs])
    task1_collate_fn = create_collate_fn(max_len*3,T=max_len*2)
    time_type,max_time,min_time = decide_time_type(trajs)
    print(max_len,time_type,max_time,min_time)
    dataset = task1Dataset(trajs,with_time=True,time_type=time_type)
    for data in dataset:
        print(data['points'].shape)
    print(dataset[0])
    dataloader = task1DataLoader(dataset,batchsize=2,collate_fn=task1_collate_fn)
    for batch in dataloader:
        print(batch)
        print(batch['traj'].shape)
        print(batch['cond'].shape)
        print(batch['reagent_mask'].shape)
        break
    # dataloader = task1DataLoader(dataset,batchsize=2,collate_fn=task1_collate_fn)

    # for batch in dataloader:
    #    print(batch)
    #    print(batch['traj'].shape)
    #    print(batch['cond'].shape)
    #    print(batch['reagent_mask'].shape)
    #    break

    # cfg = {
    #      "vocab_size": 9000,
    #         "device": "cuda:0",
    #         "block_size": 20,
    #         "n_embd": 64,
    #         "n_head": 4,
    #         "n_layer": 2,
    #         "dropout": 0.1,
    #         "n_hidden": 64,
    #         "use_agent_mask": True,
    # }

    # model = define_model(cfg)
    # # 示例使用
    # device = 'cuda:0'
    # optimizer = optim.Adam(model.parameters(), lr=0.001)  # 优化器

    # num_epochs = 10000  
    # epoch_losses = train(model, dataloader, optimizer, num_epochs, device)

    

   




