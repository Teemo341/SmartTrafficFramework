from util import *
import matplotlib.pyplot as plt
from functools import partial
from copy import deepcopy
from pprint import pprint
from itertools import chain

# from numba import jit

def vis_len_dist(traj_len_list):
    max_len = max(traj_len_list)
    plt.hist(traj_len_list,bins=100)
    plt.title('Trajectory Length Distribution')
    plt.xlabel('Trajectory Length')
    plt.ylabel('Number of Trajectories')
    plt.savefig('traj_len_dist.png')
    return max_len


def juxtapose(filepath = "../Porto/oneday_data.csv", 
            outpath = "../Porto/dev_out.pkl",
            with_ratio = True,
            downsample=1,
            max_len=1e6,
            **kwargs):
    args = []
    for k,v in kwargs.items():
        args.append(f'--{k}')
        args.append(str(v))
    cfg = get_cfg(args)
    num_proc = cfg['num_processes']
    chunk_size = None
    print("Fold Traj from file: ",filepath)
    print("Num of Processes: ",num_proc)
    

    raw_data =[]
    def _trans(x):
        tmp=eval(x)
        return (tmp[0],tmp[1]) if not with_ratio else (tmp[0],tmp[1],tmp[2])
    
    def _transline(line:str):
        first = line.split(' ')[1::downsample] 
        return list(map(_trans,first))
        
        
    
    def _padding_zero(traj,max_len):
        # return traj + [0]*(max_len-len(traj))
        return np.concatenate([traj,np.zeros((max_len-len(traj),))],axis=0)
    
    def _padding_zero_ratio(traj,max_len):
        # return traj + [0]*(max_len-len(traj))
        return np.concatenate([traj,np.zeros((max_len-len(traj),2))],axis=0)
    
    def _get_cond(traj,max_len):
        end = traj[-1]
        cond = np.tile(np.array([end]),(len(traj)))
        cond = np.concatenate([cond,np.zeros((max_len-len(traj)))],axis=0)
        return cond
        
    
    with timeit("Read Raw Data"):
        with pPool(num_proc) as p:
            raw_data = (p.map(_transline,lazy_readlines(filepath),chunk_size))
            # raw_data = [_transline(line) for line in lazy_readlines(filepath)]
                # (B,T,*)
    
    # # DEBUG:
    # raw_data = raw_data[:2000]
    raw_data = list(filter(lambda x:len(x)<=max_len,raw_data))
    
    print(raw_data[:3])
    print("Num of trajs: ", len(raw_data) )
    max_timestep = max([x[-1][0] for x in raw_data])
    print("Max timestep: ", max_timestep)
    
    if not with_ratio:
        len_list, start_time_list,raw_traj = zip(*[(len(x),x[0][0],np.array(x)[:,1]) for x in raw_data])
    else:
        len_list, start_time_list,raw_traj = zip(*[(len(x),x[0][0],np.array(x)[:,1:]) for x in raw_data])
    # len_list, start_time_list,raw_traj = zip(*p.map(lambda x:(len(x),x[0][0],np.array(x)[:,1]),raw_data,chunk_size)    )
    
    vis_len_dist(len_list)
    max_len = max(len_list)+1
    print("Max Traj Len: ",max_len)
    
    if not with_ratio:
        max_edge_id = max([(traj.max()) for traj in raw_traj])
    else:
        max_edge_id = max([(traj[:,0].max()) for traj in raw_traj])
    print("Max Edge ID: ",max_edge_id)
    
    with timeit("Pad Zero & Get Cond"):
        with pPool(num_proc) as p:
            # traj_list,cond = zip(*[(_padding_zero(traj,max_len),_get_cond(traj,max_len)) for traj in raw_traj])
            if not with_ratio:
                traj_list,cond = zip(*p.map(lambda x:(_padding_zero(x,max_len),_get_cond(x,max_len)),raw_traj,chunk_size))
            else:
                traj_list,cond = zip(*p.map(lambda x:(_padding_zero_ratio(x,max_len),_get_cond(x[...,0],max_len)),raw_traj,chunk_size))
    B = len(raw_data)
    
    traj_arr_ = np.array(traj_list) # (B,T,?)
    if not with_ratio:
        traj_arr = traj_arr_.astype(np.int64 ).reshape(B,max_len,1)
    else:
        traj_arr = traj_arr_[...,0].astype(np.int64).reshape(B,max_len,1)
        ratio_arr = traj_arr_[...,1].astype(np.float32).reshape(B,max_len,1)
        
    buf = Batch(
        {
        'traj':traj_arr,
        'cond':np.array(cond,dtype=np.int64).reshape(B,max_len,1),
        'len':np.array(len_list,dtype=np.int64),
        'start_time':np.array(start_time_list,dtype=np.int64),
        'reagent_mask': np.ones((B,max_len,1),dtype=np.int64),
    }
    )
    if with_ratio:
        buf.update({
            'ratio':ratio_arr
        })
    # print(f"{buf[0]=}")
    color_print(f"{np.sum(buf.len)=}")
    color_print(f"{buf['traj'].shape=}")
    color_print(f"{buf.shape=}")
        
    # Target:
    #     Batch({
    #         'traj': (B,T,1), where T is padding to max_len
    #         'cond': (B,T,1), where T is padding to max_len
    #         'len': (B,)
    #         'start_time': (B,)
    #     })
    
    
    buf.to_torch()
    pickle.dump(buf,open(outpath,'wb'))
        

def juxcat(filelist:str,outpath = "../Porto/dev_out.pkl",**kwargs):
    """
        Usage: python traj_generator.py juxcat --filelist="../Porto/100m5s_0.pkl,../Porto/100m5s_1.pkl" --outpath=../Porto/2x14w5s.pkl
    """
    buf_list = []
    max_len = 0
    for filepath in filelist.split(','):
        cur_batch = pickle.load(open(filepath,'rb'))
        cur_len = cur_batch.traj.shape[1]
        # assert use_ratio == ('ratio' in cur_batch), "Inconsistent use of ratio" 
        print(f"Cur Len of {filepath}: ",cur_len)
        max_len = max(max_len,cur_len)
        buf_list.append(cur_batch)
    print("Max Len: ",max_len)
    
    use_ratio = 'ratio' in buf_list[0]
    
    
    for buf in buf_list:
        shape = buf.cond.shape
        print("Cur Shape: ",shape)
        # print(f"Dtype: {buf.traj.dtype=}, {buf.cond.dtype=}, {buf.reagent_mask.dtype=}")
        buf.traj = torch.concatenate([buf.traj,torch.zeros((shape[0],max_len-shape[1],shape[2]),dtype=buf.traj.dtype)],dim=1)
        if use_ratio:
            buf.ratio = torch.concatenate([buf.ratio,torch.zeros((shape[0],max_len-shape[1],shape[2]),dtype=buf.ratio.dtype)],dim=1)
        buf.cond = torch.concatenate([buf.cond,torch.zeros((shape[0],max_len-shape[1],shape[2]),dtype=buf.cond.dtype)],dim=1)
        buf.reagent_mask = torch.concatenate([buf.reagent_mask,torch.ones((shape[0],max_len-shape[1],shape[2]),dtype=buf.reagent_mask.dtype)],dim=1)
    
    tot_buf = Batch.cat(buf_list)
    tot_num = tot_buf.shape[0]
    print("Total Number of Traj: ",tot_num)
    color_print(f"{tot_buf.len.sum()=}")
    color_print(f"{tot_buf['traj'].shape=}")
    color_print(f"{tot_buf.shape=}")
    pickle.dump(tot_buf,open(outpath,'wb'))
    

if __name__=="__main__":
    SmartRunner({
        'jux':juxtapose,
        'juxcat':juxcat,
        # 'dev':dev,
        },fire=True)