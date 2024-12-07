import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any, Callable, Union, Optional
from multiprocessing import Pool
from multiprocess import Pool as pPool 
# pyright: reportAttributeAccessIssue=false
import argparse
import torch
from torch.utils.data import DataLoader, Dataset
from functools import partial
from torch import nn
import itertools
import sys
import os
import contextlib 
import time
import fire
# from ma_model import SpatialTemporalCrossMultiAgentModel
import pickle
from copy import deepcopy
import networkx as nx

class EarlyStopping:
    def __init__(self, patience=100, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.4e} --> {val_loss:.4e}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

def cut_generation(y_hat_ods=None):
    """Cut tokens after 0
    
    Args:
        y_hat_ods: torch.Tensor (B, T, N)
    Return:
        y_hat_ods: torch.Tensor(B, T, N)
    """
    B, T, N = y_hat_ods.shape
    y_hat_ods = y_hat_ods.transpose(-1, -2).reshape(B*N, T)
    padding = torch.zeros((B*N, 1), device=y_hat_ods.device, dtype=y_hat_ods.dtype)
    
    zero_indices = (torch.cat((y_hat_ods, padding), dim=-1)==0).to(torch.long).argmax(dim=1)
    mask = torch.arange(T, device=y_hat_ods.device).unsqueeze(0).tile((B*N, 1)) > zero_indices.unsqueeze(-1)
    y_hat_ods[mask] = 0
    y_hat_ods = y_hat_ods.reshape(B, N, T).transpose(-1, -2)
        
    return y_hat_ods

@contextlib.contextmanager
def log_and_print(log_dir: Optional[str]='.'):
    if log_dir is None:
        yield
        return
    
    old_print = sys.stdout.write
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_handle = open(os.path.join(log_dir,'log.txt'), 'a')
    def _log_print(*args, **kwargs):
        log_handle.write(*args, **kwargs)
        old_print(*args, **kwargs)
        # sys.stdout.flush()
        log_handle.flush()
    sys.stdout.write = _log_print
    try:
        yield
    finally:
        sys.stdout.write = old_print
        log_handle.close()

def pdb_decorator(fn):
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            print(">>BUG: ",e)
            import pdb;pdb.post_mortem()
    return wrapper

def dev_decorator(fn):
    def wrapper(*args, **kwargs):
        print(">>DEV: ",fn.__name__)
        print(">>DEV: This Func is under development.")
        return fn(*args, **kwargs)
    return wrapper

#@pdb_decorator
def smart_run(cmd_dict: Union[dict[ str, Callable], Callable[[List[str]], Any]], 
              **kwargs):
    # supported kwargs:
        # log_dir: str
        # fire: bool
    log_dir = kwargs.get('log_dir', None)
    use_fire = kwargs.get('fire', True)
    assert use_fire or not isinstance(cmd_dict, dict), "cmd_dict should be a callable if fire is False"
    with log_and_print(log_dir):
        args = sys.argv[1:]
        print("-----"*20)
        print("Args: \t", args)
        print("Join Args: \t", ' '.join(args))
        print("Log at: \t", log_dir)
        print("-----"*20)
        start_time = time.time()
        if use_fire:
            ret= fire.Fire(cmd_dict)
        else:
            assert callable(cmd_dict), "cmd_dict should be a callable if fire is False"
            ret = cmd_dict(args)
        end_time = time.time()
        print("-----"*20)
        print("Result: \t",ret)
        print("Args: \t", args)
        print("Join Args: \t", ' '.join(args))
        print("Log at: \t", log_dir)
        print("Program Running time: ", end_time-start_time)


def lazy_readlines(filepath):
    with open(filepath) as f:
        id = 1
        while True:
            line =  f.readline()
            if id%1000==0:
                print(id)
            id+=1
            if not line:
                break
            yield line

# Grid Setting
def encode_single_fn(x, y,   grid_size)->str:
    # x,y, in [1,10]
    # if (x,y)==end:
    #     return '101'
    assert 1 <= x <= grid_size and 1 <= y <= grid_size, f'Invalid grid cell ({x}, {y})'
    return str((x - 1) * grid_size + y)

def decode_single_fn(code:str, grid_size)->Tuple[int, int]:
    # if code=='101':
    #     return end
    if code =='0':
        return (0,0)
    if not(1 <= int(code) <= grid_size * grid_size):
        print(f'Warning: Invalid code {code}')
    icode = int(code)
    x = (icode - 1) // grid_size + 1
    y = (icode - 1) % grid_size + 1
    return x, y

def encode_all_fn(xy_list, grid_size)->str:
    return ';'.join([encode_single_fn(x, y, grid_size) for x, y in xy_list])

def decode_all_fn(code:str, grid_size)->List[Tuple[int, int]]:
    return [decode_single_fn(x, grid_size) for x in code.split(';')]


def _decode(x,N):
    if x == '0':
        return [0] * N
    return list(map(int, x.split(';')))

def read_encoded_data(filename,N):
    with open(filename, 'r') as file:
        data = file.read().strip().split()
    _par_decode = partial(_decode, N=N)
    # return [int(code) for code in data]
    return list(map(_par_decode,data))

def read_encoded_data_parallel(filename, num_processes,N):
    with open(filename, 'r') as file:
        data = file.read().strip().split()

    # _decode = lambda x: [0]*N if x=='0' else list(map(int,x.split(';')))
    _par_decode = partial(_decode, N=N)
    with Pool(processes=num_processes) as pool:
        # pool._pickle.dumps = dill.dumps
        
        decoded_data = pool.map(_par_decode, data)

    return decoded_data

def get_condition(input_data,N):
    conditionEmbedding = []
    for i, carsT in enumerate(input_data):
        if carsT[0]==0 and i==len(input_data)-1:
            startPadding = [0]*N
            endPadding = [0]*N
            conditionEmbedding.append(list(zip(startPadding, endPadding)))
        elif carsT[0]==0:
            j = 1
            while input_data[i+j][0]!=0:
                j += 1
            ends = input_data[i+j-1]
            starts = input_data[i+1]
            
            for k in range(j):
                conditionEmbedding.append(list(zip(starts, ends)))
    return conditionEmbedding
    
def get_condition_folded(folded_traj, agent_mask, num_proc=10):
    # it also relabel agent_mask
    
    N,T = len(folded_traj),len(folded_traj[0])
    
    def _get_cond(agent_traj,_agent_mask):
        _agent_mask = deepcopy(_agent_mask)
        rev_cond = []
        start = 0
        end = 0
        last_zero = T
        for i in range(T-1, -1, -1):
            if agent_traj[i]==0:
                rev_cond.extend([(start,end)]*(last_zero-i) )
                if start!=0 and end!=0:
                    _agent_mask[i]=1
                last_zero = i
                start=0
                end=0
            else:
                if i==T-1 or agent_traj[i+1]==0:
                    end=agent_traj[i]
                if i==0 or agent_traj[i-1]==0:
                    start=agent_traj[i]
                
        # end at i=0
        rev_cond.extend([(start,end)]*(last_zero) )
        
        return rev_cond[::-1],_agent_mask
        ...
    with pPool(num_proc) as p:
        res = p.starmap(_get_cond,zip(folded_traj,agent_mask))
    # res = []
    # for agent_traj,_agent_mask in zip(folded_traj,agent_mask):
    #     res.append(_get_cond(agent_traj,_agent_mask))
    # import pdb;pdb.set_trace()
    return zip(*res)
    ...

# data loading
def _old_get_get_batch(train_data, val_data, block_size, batch_size, device, adjmat, with_closed=False):
    adjmat = adjmat.to(device)
    def get_batch(split):
        # generate a small batch of data of inputs x and targets y
        # Shape: [batch_size, block_size, N]
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        # print(ix)
        X = torch.stack([data[i:i + block_size] for i in ix]).to(device)
        Y = torch.stack([data[i + 1:i + block_size + 1] for i in ix]).to(device)
        
        if with_closed:
            pass
        else:
            A = get_adjmask(X[:,:,:,0], adjmat)
        # import pdb;pdb.set_trace()
        return X, Y, A
    return get_batch

def old_build_dataloader(cfg, adjmat, filename='data.txt'):
    N = cfg['N']
    input_data = [[0]*N]
    if cfg['with_closed']:
        filename = 'data_closed.txt'
        
    print("Loading data from %s" % (filename))
    
    input_data += read_encoded_data(filename, N)
    data = torch.tensor(input_data, dtype=torch.long).unsqueeze(-1)

    conditions = torch.Tensor(get_condition(input_data, N)).long()
    data = torch.cat([data, conditions], dim=-1).long()
    n = int(0.9*len(data)) # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]
    # shape: [L, N, 3]
    get_batch = _old_get_get_batch(train_data, val_data, cfg['block_size'], cfg['batch_size'], cfg['device'], adjmat, cfg['with_closed'])
    
    cfg.update({'data_cfg':
        {
            'num_transition': len(data),
            'num_train': len(train_data),
            'num_val': len(val_data),
        }})
    
    return get_batch, data.shape, train_data, val_data

# added by zhouyi
def _closed_get_get_batch(train_data, val_data, block_size, batch_size, device):
    def get_batch(split):
        data, adj_lists, idx_closeds = train_data if split == "train" else val_data
        idx = np.random.choice(len(data), batch_size, replace=False)
        sample_batch = lambda A: [A[i] for i in idx]
        data, adj_lists, idx_closeds = map(sample_batch, (data, adj_lists, idx_closeds))
        breakpoint()
        return data, adj_lists, idx_closeds
    return get_batch
    
# added by zhouyi
def closed_build_dataloader(cfg, graph_dir='graph', traj_file='data_closed.txt', num_closed=15):
    N = cfg['N']
    if N > 1:
        raise NotImplementedError
    adj_lists = pickle.load(open(os.path.join(graph_dir, f'grid_closed_{num_closed}.pkl'), 'rb'))
    idx_closeds = np.load(os.path.join(graph_dir, 'idx_closed.npy'))
    with open('data_closed.txt', 'r') as file:
        input_data = file.readlines()    
        input_data = [[[int(i)] for i in traj.split()] for traj in input_data]
        trajs = [torch.tensor(traj, dtype=torch.long).unsqueeze(-1) for traj in input_data]
        conditions = [torch.tensor(get_condition(a_data, N), dtype=torch.long) for a_data in input_data]
        data = [torch.cat([traj, condition], dim=-1).long() for traj, condition in zip(trajs, conditions)]
    n = int(0.9 * len(data))
    train_data = data[:n], adj_lists[:n], idx_closeds[:n]
    val_data = data[n:], adj_lists[n:], idx_closeds[n:]
    get_batch = _closed_get_get_batch(train_data, val_data, cfg['block_size'], cfg['batch_size'], cfg['device'])
    return get_batch, train_data, val_data

class StreamingDataset(Dataset):
    def __init__(self, folded_data:dict, block_size:int):
        # data format : { 
        #     'folded_traj':folded_traj, # [T, N]
        #     'agent_mask':agent_mask, 
        #     'cond':cond, # [T, N, 2]
        #     'reagent_mask':reagent_mask # [T, N]
        #     'adjmask':adjmask # [T,N,V]
        # }
        self.data = folded_data
        self.block_size = block_size
        
        self.N = self.data['folded_traj'].shape[-1]
        self.T = self.data['folded_traj'].shape[0]
        # self.use_agent_mask = use_agent_mask
        
    def __len__(self):
        return self.T - self.block_size # not +1, because the Y is the shifted version of X, which needs one more step
    
    def __getitem__(self, idx):
        # X,Y,C,M,A
        return (self.data['folded_traj'][idx:idx+self.block_size], 
                self.data['folded_traj'][idx+1:idx+self.block_size+1], 
                self.data['cond'][idx:idx+self.block_size],    
                self.data['reagent_mask'][idx:idx+self.block_size], # if self.use_agent_mask else None , 
                self.data['adjmask'][idx:idx+self.block_size]
                )       
        # return self.data[idx:idx+self.block_size]

def cycleiter(iterable):
    # Original itertools.cycle is bad
        # which will save all element val in the first iteration epoch
        # and in the next epoch, it will yield the same val in the same order 

    while True:
        for ele in iterable:
            yield ele

def build_dataloader(cfg,adjmat,filename='../Boston/folded_traj.pkl'):
    # { 
    #     'folded_traj':folded_traj, # [N, T]
    #     'agent_mask':agent_mask, # [N, T]
    #     'max_cur_agent':max_cur_agent,
    #     'max_timestep':max_timestep,
    #     'cond':cond, # [N, T, 2]
    #     'reagent_mask':reagent_mask # [N, T]
    # }
    
    folded_data = pickle.load(open(filename,'rb'))
    for k,v in folded_data.items():
        if k in ['folded_traj','cond']:
            folded_data[k] = torch.tensor(v, dtype=torch.long).transpose(0,1)
        elif k in ['agent_mask','reagent_mask']:
            folded_data[k] = torch.tensor(v, dtype=torch.bool).transpose(0,1)
            # [T,N,...]
    
    # build adjmask
    folded_data['adjmask'] = get_adjmask(folded_data['folded_traj'], adjmat)
        # it's comped and stored in cpu. it's huge
    
    T = folded_data['max_timestep']
    N = folded_data['max_cur_agent']
    cfg['N'] = N
    device = cfg['device']
    
    folded_data.pop('max_timestep')
    folded_data.pop('max_cur_agent')
    
    
    train_data = {k:v[:int(0.9*T)] for k,v in folded_data.items()}
    val_data = {k:v[int(0.9*T):] for k,v in folded_data.items()}
    train_dataset = StreamingDataset(train_data, cfg['block_size']) #, cfg['use_agent_mask'])
    val_dataset = StreamingDataset(val_data, cfg['block_size']) #, cfg['use_agent_mask'])
    
    train_dataloader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=8)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg['batch_size'], shuffle=False)
    
    train_cycleiter = cycleiter(train_dataloader)
    val_cycleiter = cycleiter(val_dataloader)
    # train_cycleiter = iter(train_dataloader)
    # val_cycleiter = iter(val_dataloader)
    
    def _get_batch(split:str)->Tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
        # return train_dataloader if split=='train' else val_dataloader
        ret = next(train_cycleiter) if split=='train' else next(val_cycleiter)
        return tuple(x.to(device) for x in ret)
    
    return _get_batch, (T,N), train_data, val_data
    
    
def comp_traj_weight(traj,weight_dict):
    return sum([weight_dict[int(x)]  for agents in traj for x in agents])
    
def get_adjmask(X, graph_adjmat:torch.Tensor):
    # X: [B, T, N] or [T,N]
    # adjmat: [V, V]
    s = X.shape
    adj_mask = graph_adjmat[X.reshape(-1)].reshape(*s,-1)
    
    return adj_mask

def get_graph(cfg):
    # load adj list from somewhere
    if cfg['setting']=='boston':
        adjlist = pickle.load(open('./graph/boston.pkl','rb'))
    elif cfg['setting']=='grid':
        if cfg['with_closed']:
            adjlist = pickle.load(open('./graph/grid_closed_10.pkl','rb'))
        else:
            adjlist = pickle.load(open('./graph/grid_10.pkl','rb'))
    else:
        raise NotImplementedError
    
    V = len(adjlist)+1
    print("Vocab Size: ",V)
    cfg['vocab_size'] = V
    adjmat = torch.zeros(V,V)


    indices = [[node, neighbor] for node, neighbors in adjlist.items() for neighbor in neighbors]
    indices = torch.tensor(indices).t()  # Transpose to match the input requirement of sparse_coo_tensor
    values = torch.ones(len(indices.t()))  # All adjacency values are 1

    # Create a sparse adjacency matrix
    adjmat = torch.sparse_coo_tensor(indices, values, (V, V))#.to_dense()
    adjmat = torch.eye(V)+adjmat
    max_degree = int(adjmat.sum(dim=-1).max().item())
    print("Max degree: ",max_degree)
    adjmat[0]=1
    adjmat[:,0]=1
    
    return adjlist, adjmat, V

# def get_model(cfg, load_from:Optional[str] = None):
#     N = cfg['N']
#     device = cfg['device']
#     block_size, n_embd, n_head, n_layer, dropout = cfg['block_size'], cfg['n_embd'], cfg['n_head'], cfg['n_layer'], cfg['dropout']
#     n_hidden = cfg['n_hidden']
#     n_embed_adj = cfg['n_embed_adj']
#     vocab_size = cfg['vocab_size']
#     use_ne = cfg['use_ne']
#     use_ge = cfg['use_ge']
#     use_adaLN = cfg['use_adaLN']
#     use_adjembed = cfg['use_adjembed']
#     postprocess = cfg['postprocess']
#     window_size = cfg['window_size']

#     use_model = cfg['use_model']
#     graph_embedding_mode = cfg['graph_embedding_mode']
#     if use_model=="sd":
#         model = SpatialTemporalCrossMultiAgentModel(vocab_size, 
#                                                     n_embd, 
#                                                     n_hidden, 
#                                                     n_layer, 
#                                                     n_head, 
#                                                     block_size,
#                                                     n_embed_adj,
#                                                     window_size=window_size,
#                                                     dropout=dropout, 
#                                                     use_ne=use_ne, 
#                                                     use_ge=use_ge, 
#                                                     device=device,
#                                                     postprocess=postprocess,
#                                                     use_adjembed=use_adjembed,
#                                                     graph_embedding_mode=graph_embedding_mode
#                                                     )
#     else:
#         raise NotImplementedError
#     # elif use_model=="naive":
#     #     model = NaiveMultiAgentLanguageModel(N,vocab_size, n_embd, n_layer,n_head,block_size, dropout,device=device)
#     # else:    
#     #     model = MultiAgentBigramLanguageModelWithoutAttention(N,vocab_size)

#     model = model.to(device)
    
#     if load_from is not None:
#         model.load_state_dict(torch.load(load_from, map_location=device),strict=True)
    
#     return model

def count_non_embedding_params(model):
    non_embedding_params = 0
    
    for name, layer in model.named_modules():
        if not isinstance(layer, nn.Embedding):
            for param in layer.parameters(recurse=False):
                non_embedding_params += param.numel()
    
    return non_embedding_params

def get_cfg(args: Optional[List[str]]=None)->dict:
    str2bool = lambda x: x.lower() in ['true', '1', 't','y','yes']
    
    parser = argparse.ArgumentParser()
    # Defining Scenario
    parser.add_argument('--expname', type=str, default='n', help='Name of the experiment')
    parser.add_argument('--debug', type=str2bool, default=False)
    parser.add_argument('--setting', type=str, default='boston', choices=['boston', 'grid', 'paris', 'porto', 'beijing', 'jinan'])
    parser.add_argument('--new_dataloader', type=str2bool, default=True)
    parser.add_argument('--N', type=int, default=1)
    parser.add_argument('--grid_size', type=int, default=10)
    parser.add_argument('--enable_interaction', type=str2bool, default=True)
    parser.add_argument('--random_od', type=str2bool, default=True)
    
    parser.add_argument('--use_wandb', type=str2bool, default=True)
    
    # For data preprocess
    parser.add_argument('--vocab_size', type=int, default=101)
    parser.add_argument('--num_processes','-np', type=int, default=100) # not used by far
    parser.add_argument('--total_trajectories', type=int, default=50000)
    parser.add_argument('--with_closed', type=str2bool, default=False, help='Grid with closed points')
    parser.add_argument('--hop', type=int, default=2)
    parser.add_argument('--root_path', type=str, default='.', help='root path')
    parser.add_argument('--graph_path', type=str, default='graph/boston_100.pkl', help='graph path')
    parser.add_argument('--data_path', type=str, default='data/data_100.npy', help='data path')
    parser.add_argument('--length_path', type=str, default='data/valid_length_100.npy', help='valid length path')
    parser.add_argument('--distance_path', type=str, default='data/distance.npy', help='distance path')
    parser.add_argument('--od_per_graph', type=int, default=1000)
    parser.add_argument('--num_file', type=int, default=100)

    # For model
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--use_agent_mask', type=str2bool, default=True)
    parser.add_argument('--use_adj_mask', type=str2bool, default=False)
    parser.add_argument('--use_model', type=str, default='sd', choices=['sd', 'naive', 'noattention'])
    parser.add_argument('--use_ne', type=str2bool, default=True, help='Use Normalized Toekn Embedding')
    parser.add_argument('--use_ge', type=str2bool, default=False, help='Use Geolocation Embedding')
    parser.add_argument('--use_adaLN', type=str2bool, default=True, help='Use Adaptation Layernorm')
    parser.add_argument('--use_adjembed', type=str2bool, default=True, help='Adj embed from sratch')
    parser.add_argument('--postprocess', type=str2bool, default=False, help='Mul adj before softmax')
    parser.add_argument('--window_size', type=int, default=4, help='prefix length')

    parser.add_argument('--norm_position', type=str, default='prenorm', choices=['prenorm', 'postnorm'])
    parser.add_argument('--block_size', type=int, default=4)
    parser.add_argument('--n_embd', type=int, default=64)
    parser.add_argument('--n_embed_adj', type=int, default=16)
    parser.add_argument('--n_hidden', type=int, default=64)
    #parser.add_argument('--d_cross', type=int, default=64)
    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--n_layer', type=int, default=6)
    parser.add_argument('--dropout', type=float, default=0.0)
    
    # for training
    parser.add_argument('--save_model', type=str2bool, default=True)
    parser.add_argument('--use_dp', type=str2bool, default=False)
    parser.add_argument('--ddp_device_ids', type=str, default=None,help="Usage: --dp_device_ids='0,1,2,3'")
    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--max_iters', type=int, default=3000)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--learning_rate_s2r', type=float, default=1e-5)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--graph_embedding_mode', type=str, default='adaLN', choices=['adaLN', 'none', 'add', 'cross'])
    parser.add_argument('--iter_per_epoch', type=int, default=100)
    parser.add_argument('--finetune_load_from', type=str, default=None)

    # grad clip
    parser.add_argument('--use_ucbgc', type=str2bool, default=True)
    parser.add_argument('--max_grad_norm', type=float, default=0.5)
    parser.add_argument('--ucbgc_alpha', type=float, default=0.99)
    parser.add_argument('--ucbgc_beta', type=float, default=1.0)
    
    parser.add_argument('--eval_iters', type=int, default=100)
    parser.add_argument('--eval_interval', type=int, default=500)
    parser.add_argument('--save_interval', type=int, default=5000)
    
    # for eval
    parser.add_argument('--eval_load_from', type=str, default=None)
    
    cfg = vars(parser.parse_args(args))
    if cfg['use_dp']:
        assert cfg['dp_device_ids'] is not None, "Please specify the device ids for Data Parallel"
        cfg['dp_device_ids'] = list(map(int,cfg['dp_device_ids'].split(',')))
        cfg['batch_size'] = cfg['batch_size'] * len(cfg['dp_device_ids'])
        cfg['device'] = 'cuda'
    # cfg['vocab_size'] = 101
    # cfg['max_value'] = cfg['grid_size']
    
    return cfg


# transfer node, wrighted_adj to graph
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

def transfer_graph_(adj_table):
    # adj_table: B x N x V x 4 x 2, 1-indexing, 0 is special token, 0 is not exist, 1 is exist
    #! G is 0-indexing
    G = nx.DiGraph()
    for i in range(len(adj_table)):
        G.add_node(i)
    for i in range(len(adj_table)):
        for j in range(len(adj_table[i])):
            if adj_table[i,j,1] != 0:
                G.add_edge(i,int(adj_table[i,j,0]),weight=adj_table[i,j,1]) #adj_table is 1-indexing, G is 0-indexing
    return G