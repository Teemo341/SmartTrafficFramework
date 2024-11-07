from operator import contains
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any, Callable, Union, Optional
import sys
import os
import contextlib 
import time
from multiprocessing import Pool
from multiprocess import Pool as pPool 
# pyright: reportAttributeAccessIssue=false

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from collections import defaultdict, deque
from functools import partial,wraps
import itertools
import fire
import yaml
import pickle
from copy import deepcopy
import pdb
import argparse
# from numba import
from torch.utils.data import RandomSampler
from task1.ma_model import MAMODEL,SpatialTemporalMultiAgentModel, DPWrapperOut, dp_wrap_model
from task1.data_type_task1 import Batch

GET_BATCH = Callable[[str],Batch]

def set_seed(seed:int, deterministic:bool=False):
    """
    Set the random seed for reproducibility.
    
    Parameters:
    seed (int): The seed value to use for random number generators.
    deterministic (bool): If True, sets the CuDNN backend to deterministic mode.
                          This may impact performance.
    """
    import random; random.seed(seed) 
        ## Our code doesn't use random lib, but it's good to set it
        ## Since some lib may use random lib
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    


def color_print(*args, color:str='red', **kwargs):
    print(color_print.color_dict[color], end='')
    print(*args, **kwargs)
    print(color_print.color_dict['end'], end='')

color_print.color_dict = {
    'black': '\033[30m',
    'red': '\033[91m',
    'green': '\033[92m',
    'yellow': '\033[93m',
    'blue': '\033[94m',
    'purple': '\033[95m',
    'cyan': '\033[96m',
    'white': '\033[97m',
    'bold': '\033[1m',
    'underline': '\033[4m',
    'bg_black': '\033[40m',
    'bg_red': '\033[41m',
    'bg_green': '\033[42m',
    'bg_yellow': '\033[43m',
    'bg_blue': '\033[44m',
    'bg_purple': '\033[45m',
    'bg_cyan': '\033[46m',
    'bg_white': '\033[47m',
    'bright_black': '\033[90m',
    'bright_red': '\033[91m',
    'bright_green': '\033[92m',
    'bright_yellow': '\033[93m',
    'bright_blue': '\033[94m',
    'bright_purple': '\033[95m',
    'bright_cyan': '\033[96m',
    'bright_white': '\033[97m',
    'end': '\033[0m'
}

DEBUG = lambda *args, **kwargs: color_print("DEBUG: ",*args, color='yellow', **kwargs)

@contextlib.contextmanager
def timeit(name:str):
    start_time = time.time()
    yield
    end_time = time.time()
    print(f">>{name} with time: {end_time-start_time:.2f} ")

def pdb_decorator(fn):
    if True: return fn
    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            print(">>BUG: ",e)
            import pdb;pdb.post_mortem()
    return wrapper

def dev_decorator(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        print(">>DEV:",fn.__name__)
        print(">>DEV: This Func is under development.")
        return fn(*args, **kwargs)
    return wrapper

def timer_decorator(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        print(f">>TIMER: Enter {fn.__name__}")
        ret = fn(*args, **kwargs)
        end_time = time.time()
        print(f">>TIMER: Of {fn.__name__}: {end_time-start_time:.2f}")
        return ret
    return wrapper

def deprecate_decorator(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        print(">>DEPRECATE:",fn.__name__)
        print(">>DEPRECATE: This Func is deprecated.")
        return fn(*args, **kwargs)
    return wrapper


class SmartRunner:

    def __init__(self,cmd_dict: Union['dict[ str, Callable]', Callable[[List[str], Any], Any]], 
              **kwargs):
        self.cmd_dict = cmd_dict
        self.color = kwargs.get('color', 'green')
        if 'log_dir' in kwargs:
            self.log_path = os.path.join(kwargs['log_dir'],'log.txt')
        else:
            self.log_path = None
        self.use_fire = kwargs.get('fire', True)
        assert self.use_fire or not isinstance(cmd_dict, dict), "cmd_dict should be a callable if fire is False"
        
        self.run()
        
    @pdb_decorator
    def run(self):
        with self.log_and_print():
            args = sys.argv[1:]
            join_args = 'python '+sys.argv[0]+' '+ ' '.join(args)
            color_print("-----"*20,color = self.color)
            color_print("Current Time: ", time.strftime('%Y-%m-%d %H:%M:%S')," Time Zone: ",time.tzname[0],color = self.color)
            color_print("Args: \t", args,color = self.color)
            color_print("Join Args: \t",join_args,color = self.color)
            color_print("Log at: \t", self.log_path,color = self.color)
            color_print("-----"*20,color = self.color)
            start_time = time.time()
            if self.use_fire:
                ret= fire.Fire(self.cmd_dict)
            else:
                assert callable(self.cmd_dict), "cmd_dict should be a callable if fire is False"
                ret = self.cmd_dict(args,self)
            end_time = time.time()
            color_print("-----"*20,color = self.color)
            color_print("Result: \t",ret,color = self.color)
            color_print("Args: \t", args,color = self.color)
            color_print("Join Args: \t", join_args,color = self.color)
            color_print("Log at: \t", self.log_path,color = self.color)
            color_print("Program Running time: ", end_time-start_time,color = self.color)
            color_print("Current Time: ", time.strftime('%Y-%m-%d %H:%M:%S')," Time Zone: ",time.tzname[0],color = self.color)

    @contextlib.contextmanager
    def log_and_print(self):
        log_path = self.log_path
        if log_path is None:
            yield
            return
        
        self._lap_old_print = sys.stdout.write
        if not os.path.exists(os.path.dirname(log_path)):
            os.makedirs(os.path.dirname(log_path))
        self._lap_log_handle = open(log_path, 'a')
        def _log_print(*args, **kwargs):
            self._lap_log_handle.write(*args, **kwargs)
            self._lap_old_print(*args, **kwargs)
            # sys.stdout.flush()
            self._lap_log_handle.flush()
        sys.stdout.write = _log_print
        try:
            yield
        finally:
            sys.stdout.write = self._lap_old_print
            self._lap_log_handle.close()
            del self._lap_log_handle
            
    def mov_log(self, new_log_path:str):
        assert self.log_path is not None, "No log_path is set, can't move log file"
        import shutil
        
        color_print("|   Move log file to ",new_log_path, " from ", self.log_path, "   |",color = self.color)
        if not os.path.exists(os.path.dirname(new_log_path)):
            os.makedirs(os.path.dirname(new_log_path))
            
        self._lap_log_handle.close()
        
        # use append to keep the original content in 'new log'
        
        with open(new_log_path, 'a') as f:
            with open(self.log_path, 'r') as f2:
                f.write(f2.read())
                
                
        os.remove(self.log_path)
        
        self.log_path = new_log_path
        
        self._lap_log_handle = open(self.log_path, 'a')
        
        

def lazy_readlines(filepath):
    with open(filepath,'r') as f:
        while True:
            line =  f.readline()
            if not line:
                break
            yield line

def __edit_distance(arr1:np.ndarray, arr2:np.ndarray)->int:
    
    len1 = len(arr1)
    len2 = len(arr2)
    
    # 创建一个 (len1+1) x (len2+1) 的矩阵
    dp = np.zeros((len1 + 1, len2 + 1), dtype=int)
    
    # 初始化第一行和第一列
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j
    
    # 填充动态规划矩阵
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if arr1[i - 1] == arr2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]  # 元素相同，不需要额外操作
            else:
                dp[i][j] = min(dp[i - 1][j] + 1,    # 删除
                               dp[i][j - 1] + 1,    # 插入
                               dp[i - 1][j - 1] + 1)  # 替换
    
    return dp[len1][len2]


# Data process

def decompose_traj_adjlevel(val_traj, adjlist, V, verbose=False):
    """
    Decomposes trajectory adjacency levels.

    Args:
        val_traj: (L,)
        adjlist: Adjacency list of the graph
        adjmat: Adjacency matrix of the graph
        verbose: If True, prints the levels and their counts

    Returns:
        level_cnt_list: A dictionary with level counts
    """
    if isinstance(val_traj, torch.Tensor):
        val_traj = val_traj.cpu().numpy()
        
    candidates = defaultdict(int)
    # V = adjmat.shape[0]

    def bfs(src, dst):
        queue = deque([(src, (int)(0))])
        visited = np.zeros(V, dtype=bool)
        visited[src] = True
        while queue:
            current, level = queue.popleft()
            if current == dst:
                return level
            for neighbor in adjlist.get(current, []):
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append((neighbor, level + 1))
        return None
    
    level_cnt_list = defaultdict(int)

    # Count the occurrences of each transition in the trajectory
    for i in range(1, len(val_traj)):
        candidates[(val_traj[i-1], val_traj[i])] += 1
    
    # Compute the level for each candidate transition
    for (src, dst), count in candidates.items():
        level = bfs(src, dst)
        level = -1 if level is None else level 
        if level==-1:
            print(f"Transition {src} -> {dst} has level {level}")
        level_cnt_list[str(level)] += count
        
    if verbose:
        for level in sorted(level_cnt_list.keys()):
            print(f"level {level} cnt: {level_cnt_list[level]}/{len(val_traj)-1}")
  
    return level_cnt_list

def level_batch_to_arr(level_batch,max_level_=None):
    """
    Level Count: Batch(
        0: array(243431),
        1: array(12187),
        2: array(254),
    )
    key is a string, in [-1, ...]; value is a np array, shape (1)
    
    wanted: [0,243431,12187,254,0,...]
        # right shift 1 position, for -1
    """
    max_level = max(map(int, level_batch.keys()))
    if max_level_ is not None:
        assert max_level_>=max_level, "max_level_ should be larger than the max level in the level_batch"
        max_level = max_level_
    
    # Initialize the output array with zeros, shifted by one position for the -1 key
    output_arr = torch.zeros(max_level + 2, dtype=torch.int)
    
    # Populate the output array with values from the level_batch
    for key, value in level_batch.items():
        level = int(key) + 1  # shift right by 1 position for -1
        output_arr[level] = value.item()
    
    return output_arr
    
    

# Data loading
class StreamingDataset(Dataset):
    def __init__(self, streamdata:Batch, block_size:int):
        # data format : { 
        #     'traj':folded_traj, # [T, N]
        #     'ratio':ratio, # [T, N] # Optional
        #     'cond':cond, # [T, N, 2]
        #     'agent_mask':agent_mask, # Optional
        #     'reagent_mask':reagent_mask # [T, N]# Optional
        #     'adjmask':adjmask # [T,N,V] # Optional
        # }
        self.data = streamdata
        self.block_size = block_size
        
        self.T,self.N = self.data.shape
        # self.use_agent_mask = use_agent_mask
        
    def __len__(self)->int:
        return self.T - self.block_size # not +1, because the Y is the shifted version of X, which needs one more step
    
    def __getitem__(self, idx)->Batch:
        # X,Y,C,M,A
        # return (self.data['traj'][idx:idx+self.block_size], 
        #         self.data['traj'][idx+1:idx+self.block_size+1], 
        #         self.data['cond'][idx:idx+self.block_size],    
        #         self.data['reagent_mask'][idx:idx+self.block_size], # if self.use_agent_mask else None , 
        #         self.data['adjmask'][idx:idx+self.block_size]
        #         )       
        minibatch = self.data[idx:idx+self.block_size]
        minibatch.traj_targ = self.data['traj'][idx+1:idx+self.block_size+1]
        if 'ratio' in self.data:
            minibatch.ratio_targ = self.data['ratio'][idx+1:idx+self.block_size+1]
        return minibatch
        # return self.data[idx:idx+self.block_size]

class JuxDataset(Dataset):
    def __init__(self, data:Batch):
        # data format : { 
        #     'traj':, # [B,T,1]
        #     'cond':cond, # [B, T,1, 2]
        #     'len':len, # [B] #TODO:
        #     'start_time':start_time, # [B]#TODO:
        #     'ratio':ratio, # [B,T] Optional
        #     
        # }
        self.data = data
        
        self.B,self.T,_ = self.data.traj.shape
        # self.use_agent_mask = use_agent_mask
        
    def __len__(self)->int:
        return self.B
    
    def __getitem__(self, idx)->Batch:    
        minibatch = self.data[idx][:-1]
        minibatch.traj_targ = self.data['traj'][idx][1:]
        if 'ratio' in self.data:
            minibatch.ratio_targ = self.data['ratio'][idx][1:]
        return minibatch

class BlockJuxDataset(Dataset):
    def __init__(self, data:Batch, trajlen:torch.Tensor, block_size:int):
        # data format : { 
        #     'traj':, # [B,T,1]
        #     'cond':cond, # [B, T,1]
        #     'len':len, # [B] #TODO:
        #     'start_time':start_time, # [B]#TODO:
        #     'ratio':ratio, # [B,T] Optional
        #     
        # }
        
        # 
        self.data = data
        self.B = self._make_blocks_data(trajlen, block_size)
        self.block_size = block_size
        # DEBUG(f"{(self.B,self.block_size,self.data.shape)=}")
        # self.B = len(self.data)
        # 
        # self.B,self.T,_ = self.data.traj.shape
        # self.use_agent_mask = use_agent_mask
        
    # @dev_decorator
    # @timer_decorator
    def _make_blocks_data(self,trajlen:torch.Tensor,block_size:int)->int:
        """
            For each traj, we make blocks of it
            s.t. the first block is [0:block_size], the second block is [1:block_size+1]
            the num of blocks is len(cur_traj)-block_size
            note that each traj generate a list of blocks, different traj generate different number of blocks
            
            We return a new Batch, with the same keys, but the values are list of blocks
            [Batch,Block,1]
        """
        B,T,_ = self.data.traj.shape 
        modified_len = torch.clamp(trajlen+1,1,T-block_size)
        self._prefixsum = torch.cumsum(modified_len,0)
        datalen = int(self._prefixsum[-1].item())
        self._prefixsum = torch.cat([torch.tensor([0]),self._prefixsum[:-1]])
        
        DEBUG(f"{datalen=}",f"{self._prefixsum=}")
        # breakpoint()
        return datalen
        
        ...
        
    def __len__(self)->int:
        return self.B
        ...
    def __getitem__(self, idx:int)->Batch:    
        batch_idx = int(torch.searchsorted(self._prefixsum,idx, right=True).item())-1
        batch_shift = int(idx-self._prefixsum[batch_idx].item())
        # return self.data[batch_idx][batch_shift:batch_shift+self.block_size]
        minibatch = self.data[batch_idx][batch_shift:batch_shift+self.block_size]
        
        minibatch.traj_targ = self.data['traj'][batch_idx][batch_shift+1:batch_shift+self.block_size+1]
        if 'ratio' in self.data:
            minibatch.ratio_targ = self.data['ratio'][batch_idx][batch_shift+1:batch_shift+self.block_size+1]
        # if batch_shift == 0:
        #     minibatch.time_weight = torch.ones_like(minibatch.traj
        # )
        minibatch.is_start = torch.tensor([batch_shift==0],dtype=torch.float32)    
        
        return minibatch
        # return self.data[idx]
        ...



def cycleiter(iterable):
    # Original itertools.cycle is bad
        # which will save all element val in the first iteration epoch
        # and in the next epoch, it will yield the same val in the same order 

    while True:
        for ele in iterable:
            yield ele

def _batch_collate_fn(batches:List[Batch])->Batch:
    # batch: [B, T, N]
    return Batch.stack(batches)

def build_streamdataloader(cfg,verbose=True)->Tuple[GET_BATCH, List[int], Batch, Batch]:
    # { 
    #     'traj':folded_traj, # [T,N]
    #     'ratio':ratio, # [T,N]
    #     'cond':cond, # [T,N, 2]
    #     'agent_mask':agent_mask, # [T,N]
    #     'reagent_mask':reagent_mask # [T,N]
    #     
    # }
    
    device = cfg['device']
    
    num_few_shots = cfg['num_few_shots']
    
    fileList = cfg['__datapath_list']
    # data_batch_list = []
    train_data_list = []
    val_data_list = []
    
    for filename in fileList:
        databatch:Batch = pickle.load(open(filename,'rb'))        

        # if not cfg['use_agent_mask']:
        #     for key in ['agent_mask','reagent_mask']:
        #         if key in databatch:
        #             databatch.pop(key)

        if not cfg['use_len_ratio']:
            for key in ['ratio']:
                if key in databatch:
                    databatch.pop(key)
                
        
        local_T = databatch.shape[0]
        # data_batch_list.append(databatch)
        assert num_few_shots<local_T*0.9, "num_few_shots should be less than the length of the train trajectory"
        train_batch = databatch[:int(0.9*local_T)] if num_few_shots<0 else databatch[:num_few_shots]
        train_data_list.append(train_batch)
        val_data_list.append(databatch[int(0.9*local_T):int(0.95*local_T)])
            # reserve 5% for final test
    # T,N = data.shape
    #     # N checking will happen in Batch.cat, save our attention
    #     
    train_data = Batch.cat(train_data_list)
    val_data = Batch.cat(val_data_list)
    T = train_data.shape[0]+val_data.shape[0]
    N = train_data.shape[1]
    
    train_dataset = StreamingDataset(train_data, cfg['block_size']) 
    val_dataset = StreamingDataset(val_data, cfg['block_size']) 
    
    
    sampler = RandomSampler(train_dataset, replacement=True, generator=torch.Generator().manual_seed(0))
    train_dataloader = DataLoader(train_dataset, 
                                  sampler=sampler,
                                  batch_size=cfg['batch_size'], 
                                  collate_fn=_batch_collate_fn,
                                  shuffle=False, 
                                  num_workers=cfg['tdl_num_workers'],
                                  prefetch_factor=cfg['tdl_prefetch_factor'],
                                  pin_memory=True, 
                                  
                                  )
    val_dataloader = DataLoader(val_dataset, batch_size=cfg['batch_size'], collate_fn=_batch_collate_fn,
                                shuffle=False)
    
    train_cycleiter = cycleiter(train_dataloader)
    val_cycleiter = cycleiter(val_dataloader)
    
    def _get_batch(split:str)->Batch:
        ret:Batch = next(train_cycleiter) if split=='train' else next(val_cycleiter)
        ret.to_torch(device=device)
        return ret
    
    if verbose:
        b=_get_batch('val')

        print("Num Agents: ",N)
        print("Data shape :",[T,N])
        print("Train Shape:",train_data.shape,"Val Shape:",val_data.shape)
        print("Batch Shape:",[x.shape for x in b.values()])
        
    cfg.update({'data_cfg':
        {
            'num_total_steps': T,
            'num_train': len(train_data),
            'num_val': len(val_data),
        },
        'N':N
        })
    
    return _get_batch, [T,N], train_data, val_data
    
def make_jux_get_batch_fn(train_data:Batch,
                      val_data:Batch,
                      device:str,
                      seed:int,
                      
                      batch_size:int,
                      num_workers:int,
                      prefetch_factor:int,
                      
                      block_size:int=-1,
                      use_blockjux:bool=False,
                        train_trajlen:Optional[torch.Tensor]=None,
                        val_trajlen:Optional[torch.Tensor]=None
                      )->GET_BATCH:
    # train_dataset = StreamingDataset(train_data, cfg['block_size']) 
    # val_dataset = StreamingDataset(val_data, cfg['block_size']) 
    if use_blockjux:
        assert block_size>0, "block_size should be larger than 0"
        train_dataset = BlockJuxDataset(train_data, train_trajlen, block_size) #type:ignore
        val_dataset = BlockJuxDataset(val_data, val_trajlen, block_size)  #type:ignore
    else:
        train_dataset = JuxDataset(train_data)
        val_dataset = JuxDataset(val_data)
    
    
    # sampler = 
    train_dataloader = DataLoader(train_dataset, 
                                sampler=RandomSampler(train_dataset, replacement=True, generator=torch.Generator().manual_seed(seed)),
                                batch_size=batch_size, 
                                collate_fn=_batch_collate_fn,
                                shuffle=False, 
                                num_workers= num_workers,
                                prefetch_factor=prefetch_factor,
                                pin_memory=True, 
                                
                                )
    val_dataloader = DataLoader(val_dataset, 
                                sampler=RandomSampler(val_dataset, replacement=True, generator=torch.Generator().manual_seed(seed)),
                                batch_size=batch_size, collate_fn=_batch_collate_fn,
                                shuffle=False)
    
    train_cycleiter = cycleiter(train_dataloader)
    val_cycleiter = cycleiter(val_dataloader)
    
    print("Train Dataset Size: ",len(train_dataset))
    print("Val Dataset Size: ",len(val_dataset))
    
    def _get_batch(split:str)->Batch:
        ret:Batch = next(train_cycleiter) if split=='train' else next(val_cycleiter)
        ret.to_torch(device=device)
        return ret
    return _get_batch
    
def build_juxdataloader(cfg,verbose=True)->Tuple[GET_BATCH, List[int], dict]:
    # Loaded Data: { 
    #     'traj':, # [B,T,1]
    #     'cond':cond, # [B, T,1]
    #     'len':len, # [B]
    #     'start_time':start_time, # [B]
    #     'ratio':ratio, # [B,T] Optional
    #     
    # }
    # Output:
    #    _get_batch: Callable[[str],Batch]
    #       each call will return a Batch with shape [B,T,N]
    #   data_shape: List[int], [B,T,N]
    #   dict:
        #   train_data: Batch, [B,T,N]
        #   val_data: Batch, [B,T,N]
        #   train_trajlen, 
        #   val_trajlen
    
    device = cfg['device']
    
    num_few_shots = cfg['num_few_shots']
    
    fileList = cfg['__datapath_list']
    # data_batch_list = []
    train_data_list = []
    val_data_list = []
    
    _split_tvs = cfg['_split_tvs']
    
    for filename in fileList:
        databatch:Batch = pickle.load(open(filename,'rb'))  
        if not cfg['__use_blockjux']:
            databatch.pop('len')
        databatch.pop('start_time')      

        if not cfg['use_len_ratio']:
            for key in ['ratio']:
                if key in databatch:
                    databatch.pop(key)
        
        if _split_tvs:
            local_B = databatch.shape[0]
            assert num_few_shots<local_B*0.9, "num_few_shots should be less than the num of the train trajectory"
            train_batch = databatch[:int(0.9*local_B)] if num_few_shots<0 else databatch[:num_few_shots]
            train_data_list.append(train_batch)
            val_data_list.append(databatch[int(0.9*local_B):int(0.95*local_B)])
        else:
            if 'val' in filename or 'test' in filename:
                val_data_list.append(databatch)
            else:
                train_data_list.append(databatch if num_few_shots<0 else databatch[:num_few_shots])
                # val_data_list.append(databatch)
    train_data = Batch.cat(train_data_list)
    val_data = Batch.cat(val_data_list)
    
    if not train_data_list :
        color_print("Warning: No train data loaded",color='red')
        train_data = val_data[:1]
    
    B = train_data.shape[0]+(val_data.shape[0] if _split_tvs else 0)
    T = train_data.traj.shape[1]
    
    if T>cfg['block_size'] and not cfg['__use_blockjux']:
        print("Warning: block_size is smaller than the traj length")
        print("Warning: block_size is set to the traj length-1")
        cfg['block_size'] = T-1
    
    if cfg['__use_blockjux']:
        train_trajlen = train_data.pop('len')
        val_trajlen = val_data.pop('len')
    else:
        train_trajlen = None
        val_trajlen = None
    
    _get_batch = make_jux_get_batch_fn(train_data, val_data, device, 
                                        cfg['seed'], 
                                        cfg['batch_size'], 
                                        cfg['tdl_num_workers'], 
                                        cfg['tdl_prefetch_factor'], 
                                        cfg['block_size'], 
                                        cfg['__use_blockjux'],
                                        train_trajlen, 
                                        val_trajlen
                                   )

    
    if verbose:
        b=_get_batch('val')

        print("Raw Data shape :",[B,T])
        print("Train Shape:",train_data.shape,"Val Shape:",val_data.shape)
        print("Batch Shape:",[x.shape for x in b.values()])
        
    cfg.update({'data_cfg':
        {
            'num_trajs': B,
            'num_train': len(train_data),
            'num_val': len(val_data),
        },
        'N':1,
        'traj_max_len':T,
        })

    raw_data = {
        'train_data':train_data,
        'val_data':val_data,
        'train_trajlen':train_trajlen,
        'val_trajlen':val_trajlen,   
    }
    return _get_batch, train_data.shape, raw_data
                # train_data, val_data
    ...
    
    
def build_dataloader(cfg,verbose=True):
    if True:
        cfg['type_dataloader']='jux'
        return build_juxdataloader(cfg,verbose)
    else:
        cfg['type_dataloader']='stream'
        return build_streamdataloader(cfg,verbose)
    
    
    
# Meta function
def collect_args(**kwargs):
    
    args = []
    for k,v in kwargs.items():
        args.append(f'--{k}')
        args.append(str(v))
    return args
    
def comp_traj_weight(traj,weight_dict):
    return sum([weight_dict[int(x)]  for agents in traj for x in agents])
    

def get_graph(cfg)->Tuple[Dict[int,List[int]],None,int]:
    # load adj list from somewhere
    # if cfg['setting']=='boston':
    #     res = pickle.load(open('./graph/boston.pkl','rb'))
    #     cfg['vocab_size'] = res['V']
    #     return res['adj_list'],None,res['V']
    # elif cfg['setting']=='porto':
    res = pickle.load(open(f'./graph/{cfg["setting"]}.pkl','rb'))
    cfg['vocab_size'] = res['V']
    # return res['adj_list'],res['adjmat'],res['V']
    return res['adj_list'],None,res['V']
    # else:
    #     raise NotImplementedError
    
    raise NotImplementedError("Please rethinking the 1-based or 0-based indexing problem, and adjmat problem for boston data")

    ...

def get_model(cfg:dict, load_from:Optional[str] = None)->SpatialTemporalMultiAgentModel:

    device = cfg['device']
    block_size, n_embd, n_head, n_layer, dropout = cfg['block_size'], cfg['n_embd'], cfg['n_head'], cfg['n_layer'], cfg['dropout']
    n_hidden = cfg['n_hidden']
    window_size = cfg['ta_sliding_window']
    vocab_size = cfg['vocab_size']
    use_ne = cfg['use_ne']
    use_model = cfg['use_model']
    
    if use_model=="sd":
        model = SpatialTemporalMultiAgentModel(vocab_size, 
                                               n_embd,
                                               n_hidden, 
                                               n_layer,
                                               n_head,
                                               block_size=block_size,
                                               window_size=window_size, 
                                               dropout=dropout,
                                               use_ne=use_ne,
                                               use_agent_mask=cfg['use_agent_mask'],
                                               use_len_ratio=cfg['use_len_ratio'],
                                               use_twl=cfg['use_twl'],
                                               time_weighted_loss=cfg['time_weight_loss'], 
                                               norm_position=cfg['norm_position'],
                                               use_enchead_ver=cfg['use_enchead_ver'],
                                               use_pe=cfg['use_pe'],
                                               )

    model = model.to(device)
    
    if load_from is not None:
        print("Load model from: ",load_from)
        model.load_state_dict(torch.load(load_from, map_location=device),strict=True)
    
    return model



def get_cfg(args: Optional[List[str]]=None)->dict:
    str2bool = lambda x: x.lower() in ['true', '1', 't','y','yes']
    
    parser = argparse.ArgumentParser()
    # Defining Scenario
    parser.add_argument('--expname', type=str, default='n', help='Name of the experiment')
    parser.add_argument('--short_runname','-s',type=str2bool, default=False, help='Use runname==expname')
    parser.add_argument('--expcomm', type=str, default='n', help='Comment of the experiment, Useless, Just for record')
    
    parser.add_argument('--debug', type=str2bool, default=False)
    parser.add_argument('--setting', type=str, default='porto', choices=['boston','jinan','shenzhen','porto', 'grid','small_porto','line_20','line_500'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--__use_blockjux', type=str2bool, default=True)
    parser.add_argument('--new_dataloader', type=str2bool, default=True)
    parser.add_argument('--N', type=int, default=1)
    parser.add_argument('--enable_interaction', type=str2bool, default=True)
    parser.add_argument('--random_od', type=str2bool, default=True)
    
    parser.add_argument('--use_wandb', type=str2bool, default=True)
    
    # For data preprocess & data loader)
    parser.add_argument('--datapath','-d', type=str, default='')
    parser.add_argument('--num_few_shots', type=int, default=-1)
    parser.add_argument('--_split_tvs', type=str2bool, default=True, help='whether to split the data into train, val, test (90:5:5)')
    
    parser.add_argument('--num_processes','-np', type=int, default=64) 
    parser.add_argument('--total_trajectories', type=int, default=50000)
    parser.add_argument('--tdl_num_workers', type=int, default=2, help='Number of workers for train dataloader')
    parser.add_argument('--tdl_prefetch_factor', type=int, default=2, help='Number of workers for train dataloader')
    
    # For model
    parser.add_argument('--use_agent_mask', type=str2bool, default=False)
    parser.add_argument('--use_adj_mask', type=str2bool, default=False)
    parser.add_argument('--use_len_ratio', type=str2bool, default=False)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--use_model', type=str, default='sd', choices=['sd', 'naive', 'noattention'])
    parser.add_argument('--use_ne', type=str2bool, default=True, help='Use Normalized Toekn Embedding')
    parser.add_argument('--use_pe', type=str, default="abs",choices=["abs","alibi"], help='What type of Positional Encoding to use')
    parser.add_argument('--use_enchead_ver', type=int, default=1)
    parser.add_argument('--norm_position', type=str, default='prenorm', choices=['prenorm', 'postnorm'])
    parser.add_argument('--block_size', type=int, default=4)
    parser.add_argument('--ta_sliding_window', type=int, default=-1)
    parser.add_argument('--n_embd', type=int, default=64)
    parser.add_argument('--n_hidden', type=int, default=64)
    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--n_layer', type=int, default=6)
    parser.add_argument('--dropout', type=float, default=0.0)
    
    #我添加的
    parser.add_argument('--vocab_size', type=int, default=100)
    #
    
    # for training
    parser.add_argument('--save_model', type=str2bool, default=True)
    parser.add_argument('--use_dp', type=str2bool, default=False)
    parser.add_argument('--ddp_world_size', type=int, default=1)
    # parser.add_argument('--dp_device_ids', type=str, default=None,help="Usage: --dp_device_ids='0,1,2,3'")
    parser.add_argument('--grad_accumulation',type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_iters', type=int, default=3000)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--lr_min', type=float, default=0.0)
    parser.add_argument('--lr_warmup', type=int, default=-1, help='Warmup steps for learning rate, -1 means no warmup. Typically 1000')
    parser.add_argument('--lamb_ratio', type=float, default=1.0)
    parser.add_argument('--use_twl',type=bool,default=False)
    parser.add_argument('--time_weight_loss','-twl', type=str, default="none")
    
        # grad clip
    parser.add_argument('--use_ucbgc', type=str2bool, default=True)
    parser.add_argument('--max_grad_norm', type=float, default=0.5)
    parser.add_argument('--ucbgc_alpha', type=float, default=1-1e-2)
    parser.add_argument('--ucbgc_beta', type=float, default=1.5)
    
        # training time eval
    parser.add_argument('--eval_iters', type=int, default=50)
    parser.add_argument('--eval_interval', type=int, default=1000)
    parser.add_argument('--save_interval', type=int, default=10000)
    
    
    # for eval & finetune
    parser.add_argument('--model_load_from','-mlf', type=str, default=None, help='Load model from this path, used by eval and finetune')
    parser.add_argument('--load_model_cfg','-lmc', type=bool, default=False)
    parser.add_argument('--_eval_all', type=str2bool, default=False)
    
    # for finetune
    parser.add_argument('--is_ft', type=str2bool, default=False,help="Determines it's pretraining or finetuning")
    parser.add_argument('--ft_type',type=str, default='full', choices=['full','lora'])
    parser.add_argument('--lora_rank', type=int, default=8)
    parser.add_argument('--use_kl_reg', type=str2bool, default=False)
    parser.add_argument('--kl_reg_factor', type=float, default=0.1)
    
    
    
    cfg = vars(parser.parse_args(args))
    
    # print(">>Configurations: ",end=' ')
    # for k,v in cfg.items():
    #     print(f"--{k}={v}",end=' ')
    
    set_seed(cfg['seed'])
    
    color_print('<<<',cfg['expname'],'>>>',color='purple')
    
    assert cfg['new_dataloader']
    if cfg['use_dp']:
        assert cfg['ddp_world_size']>=1
        # assert cfg['dp_device_ids'] is not None, "Please specify the device ids for Data Parallel"
        # cfg['dp_device_ids'] = list(map(int,cfg['dp_device_ids'].split(',')))
        # cfg['batch_size'] = cfg['batch_size'] * len(cfg['dp_device_ids'])
        # cfg['batch_size'] = cfg['batch_size'] * cfg['ddp_world_size']
        cfg['device'] = 'cuda'
    else:
        cfg['ddp_world_size'] = 1
    
    if cfg['use_adj_mask']:
        raise NotImplementedError("Adj Mask is removed")
    
    if cfg['load_model_cfg']:
        assert cfg['model_load_from'] is not None, "Please specify the model to load the config"
        
        pret_cfg_path = os.path.join(os.path.dirname(cfg['model_load_from']), 'cfg.yaml')
        pret_cfg = read_cfg(pret_cfg_path)
        keys_model = ["setting","__use_blockjux","use_agent_mask","use_adj_mask","use_len_ratio","time_weight_loss","device","use_model","use_ne","use_pe","use_enchead_ver","norm_position","block_size","ta_sliding_window","n_embd","n_hidden","n_head","n_layer","dropout"]
        for k in keys_model:
            if k in pret_cfg:
                cfg[k] = pret_cfg[k]
                
        if not cfg['datapath']:
            cfg['datapath'] = pret_cfg['datapath']
            cfg['_split_tvs'] = pret_cfg['_split_tvs']
                
    if cfg['use_twl']:
        assert cfg['time_weight_loss'] =='none', "time_weight_loss should be none if use_twl is True"
                
    filename = cfg['datapath']
    try:
        fileList = eval(filename)
    except Exception:
        if ',' in filename:
            fileList = filename.split(',')
        else:
            fileList = [str(filename)]

    cfg['__datapath_list'] = fileList
    print("Data File List: ",fileList)
    
    if cfg['expcomm'] != 'n':
        color_print(">>Exp Comment: ",cfg['expcomm'],color='red')
    
    cfg['__time_idx']=time.strftime('%m%d%H%M%S')
    
    return cfg

    
def read_cfg(filename:str)->dict:
    with open(filename, 'r') as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)
    return cfg
    
def write_cfg(cfg:dict, filename:str):
    # # Save as yaml
    # # we wish it can be easily reload
    # with open(filename, 'w') as file:
    #     for k,v in cfg.items():
    #         file.write(f'{k}: {v}\n')
    yaml.dump(cfg, open(filename, 'w'))
    
def get_runname(cfg:dict)->str:
    if cfg['short_runname']:
        return cfg['expname']
    
    setting = cfg['setting']
    expname = cfg['expname']
    use_dp = cfg['use_dp']
    max_iters = cfg['max_iters']
    
    if not cfg['is_ft']:
        
        run_name = ("Debug_" if cfg['debug'] else ""   )+f"{expname}_{setting}_{'ddp' if use_dp else ''}_it{max_iters}_t{cfg['__time_idx']}"
    else:
        
        ori_run_name =  cfg['model_load_from'].split('/')[-2] 
        run_name = (
                    ("Debug_" if cfg['debug'] else ""   ) + 
                    f"{expname}_it{max_iters}_t{cfg['__time_idx']}_ft[{ori_run_name}]"
                    )
    return run_name