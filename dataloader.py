from operator import is_
from random import choice
from torch.utils.data import Dataset,DataLoader
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
from process_data import read_traj,edge_node_trans,repeat_traj
from task1.test1 import train as train1
from task2.process_task2 import train as train2
from task3.train_mae import train as train3
import time

def _padding_zero(traj,max_len):
        # return traj + [0]*(max_len-len(traj))
        return np.concatenate([traj,np.zeros((max_len-len(traj),),dtype=np.int64)],axis=0)

def simulation2adj(file_path='data/simulation/edge_node_10*10.csv'):    
    
    u_nodes = []
    v_nodes = []
    lengths = []
    data = pd.read_csv(file_path)
    
    for i in range(len(data)):
        u_nodes.append(data.loc[i, 'Origin'])
        v_nodes.append(data.loc[i, 'Destination'])
        lengths.append(data.loc[i, 'Length'])

    num_of_nodes = np.max((np.max(u_nodes), np.max(v_nodes)))
    adj_matrix = np.zeros((num_of_nodes + 1, num_of_nodes + 1))
    for u, v, length in zip(u_nodes, v_nodes, lengths):
        adj_matrix[u][v] = length
        adj_matrix[v][u] = length
    
    return adj_matrix

def adj2sparse_adjmatrix_weighted(adj):
    num_nodes = len(adj)
    indices = []
    values = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj[i][j] != 0:
                indices.append([i, j])
                values.append(adj[i][j])
    return np.array(indices).T, np.array(values, dtype=np.float32)

def read_node_type(file_path='data/simulation/node_type_10*10.csv'):
    data = pd.read_csv(file_path)
    node_type = data['Type'].tolist()
    # for i in range(len(data)):
    #     node_type.append(data.loc[i, 'Type'])
    return node_type

def adj_m2adj_l(adj_matrix,max_connection=4):
    adj_list = torch.zeros([adj_matrix.shape[0],max_connection,2])
    adj_list = torch.FloatTensor(adj_list)
    for i in range(adj_matrix.shape[0]):
        #获取节点i的邻接节点
        adj_nodes = np.nonzero(adj_matrix[i])[0]
 
        if len(adj_nodes) > max_connection:
            raise ValueError('Error: Max connection is wrong')
 
        for j in range(len(adj_nodes)):
  
            adj_list[i,j,0] = adj_nodes[j]
            adj_list[i,j,1] = adj_matrix[i,adj_nodes[j]]
    return adj_list
def task4data_process(trajs,max_len,map_path='data/simulation/edge_node_10*10.csv',
                      state_path='data/simulation/edge_node_10*10_state.csv',
                      node_type_path='data/simulation/node_type_10*10.csv'):
        node_type = read_node_type(node_type_path)
        state = pd.read_csv(state_path)
        traj_ = _padding_zero(trajs,max_len)
        traj_ = torch.tensor(traj_,dtype=torch.int64)
        result = torch.zeros([max_len,101,7])
        for i in range(len(traj_)-1):
            if traj_[i]==traj_[i+1]:
                pass
            elif traj_[i+1]==0:
                pass
            else:
                #print(traj_[i],traj_[i+1])
                state1 = state[state['EdgeID_x']==int(traj_[i])]
                #print(state1)
                state2 = state1[state1['EdgeID_y']==int(traj_[i+1])]
                #print(state2)
                value = state2['state'].values[0]
                #print('value:',value)
                if value == 0:
                    pass
                else:
                    a = [0]*7
                    a[value-1]=1
                    node1 = state['Origin_y'].values[0]
                    result[i,int(node1)]+=torch.tensor(a)
        for i in range(1,101):
            if node_type[i-1]=="T":
                result[:,i,0:4]=-1
            elif node_type[i-1]=="C":
                result[:,i,5:]=-1
            else:
                result[:,i,:]=-1

        return result

class SmartTrafficDataset(Dataset):
    # trajs: list of list of int
    # map_: np.array([[edge,o,d,length],...])
    # T: int


    def __init__(self,trajs,map_path='data/simulation/edge_node_10*10.csv',
                 node_type_path = 'data/simulation/node_type_10*10.csv',
                 state_path = 'data/simulation/edge_node_10*10_state.csv',
                 task4_path = 'data/simulation/task4_traj_data.npy',
                 T=None,window_size=1,mode = "task1",max_len=None,is_edge=True,task4_num=2000,vocab_size=100+1):
        super(SmartTrafficDataset,self).__init__()
        self.mode = mode
        self.is_edge = is_edge
        self.vocab_size = vocab_size
        
        if self.mode == "task1":
        
        #task1
            self.trajs = trajs
            self.window_size = window_size
            self.max_len = max_len if max_len else max([len(traj) for traj in trajs])
            self.T = T if T else self.max_len//2
        #task2
        elif self.mode == "task2":
            self.max_len = max([len(traj) for traj in trajs])
            self.trajs = trajs
            self.window_size = window_size
            self.map = pd.read_csv(map_path)
            self.adjacent= simulation2adj(map_path)
            self.indices , self.values =adj2sparse_adjmatrix_weighted(self.adjacent)
            self.T = T if T else self.max_len//2
        #task3
        elif self.mode == "task3":
            self.max_len = max([len(traj) for traj in trajs])
            self.trajs = trajs
            self.window_size = window_size
            self.max_len = max_len if max_len else max([len(traj) for traj in trajs])
            self.T = T if T else self.max_len//2
            self.adj_l = adj_m2adj_l(self.adjacent)
        #task4
        elif self.mode == "task4":
            #self.node_type = read_node_type(node_type_path)
            #self.state = pd.read_csv(state_path)
            self.task4_num = task4_num
            self.task4_path = task4_path
            self.trajs = np.load(self.task4_path)
            self.max_len = self.trajs.shape[1]
            

    def __len__(self):
        return len(self.trajs)
    
    def __getitem__(self,idx):
        
        if self.mode == 'task4':
            idx = np.arange(0,len(self.trajs))
            choice  = np.random.choice(idx,size = self.task4_num)
            #results =np.sum(self.trajs[choice],axis=0)
            results = np.sum(self.trajs[choice],axis=0)
            return torch.tensor(results, dtype=torch.int)
        
        else:
            if self.T + self.window_size > self.max_len:
                raise ValueError('Error: T + window_size is too large')
            traj = _padding_zero(self.trajs[idx],self.max_len)
            traj_ = traj[0:self.T]
            traj_targ = traj[self.window_size:self.T+self.window_size]
            reagent_mask = [ 1 if x!=0 else 0 for x in traj_ ]
            reagent_mask = torch.tensor(reagent_mask,dtype=torch.int64).view(-1,1)

            valid_length = torch.tensor([torch.sum(reagent_mask)],dtype=torch.int64)
            od = torch.tensor([traj[0],traj[valid_length[0]-1]],dtype=torch.int64)
            od = od[None,None,:].repeat(self.T,1,1) 
            
            traj_ = torch.tensor(traj_,dtype=torch.int64)
            traj_ = traj_.view(-1,1)
            traj_targ = torch.tensor(traj_targ,dtype=torch.int64)
            traj_targ = traj_targ.view(-1,1)

            if self.mode == "task1":
                #traj:[T,1]
                #od:[T,1,2]
                #traj_targ:[T,1]
                #reagent_mask:[T,1]
                
                return {'traj':traj_,'cond':od,'traj_targ':traj_targ,'reagent_mask':reagent_mask}
            
            if self.mode == "task2":
                # x and y: (1, T, 1)  
                # x_valid: (1, 1), valid length for each trajectory
                # condition: (1, T, 1, 2) 
                
                return traj_ ,valid_length,od,traj_targ,self.indices,self.values
            
            if self.mode == "task3":
                #time_step(not uesd) is set [0]
                traj_ = traj_[:,0]
                traj_ = traj_[None,:]
                reagent_mask = reagent_mask[:,0]
                reagent_mask = reagent_mask[None,:]

                return traj_  , [0], reagent_mask , self.adj_l[None,:,:,:]

    
    def __getattribute__(self, name: str) -> torch.Any:
        return super().__getattribute__(name)

class SmartTrafficDataloader(DataLoader):
    def __init__(self,dataset,batch_size=1,shuffle=False):
        super(SmartTrafficDataloader,self).__init__(dataset,batch_size=batch_size,shuffle=shuffle)
        self.max_len = dataset.max_len
        self.vocab_size = dataset.vocab_size
    def get_max_len(self):
        return self.max_len
    
    def randomize_condition(self, observe_prob=0.5):
        self.observe_list = np.random.choice(
            (self.vocab_size), int(self.vocab_size*observe_prob), replace=False)+1
    
    def filter_condition(self, traj_batch):
        unobserved = torch.ones(traj_batch.shape, dtype=torch.int32, device = traj_batch.device)
        for i in range(len(self.observe_list)):
            unobserved *= (traj_batch != self.observe_list[i])
        observed = 1 - unobserved
        traj_batch = traj_batch * observed
        return traj_batch


if __name__ == '__main__':

    # trajs = read_traj('data/simulation/trajectories_10*10.csv')
    # print(trajs[0:1])
    # map_ = pd.read_csv('data/simulation/edge_node_10*10.csv')
    # map_ = np.array(map_)
    # new_trajs = repeat_traj(trajs,map_)
    # trajectories_str = ["_".join(map(str, traj)) for traj in new_trajs]
    # new_traj = pd.DataFrame(trajectories_str,columns=['Trajectory'])
    # new_traj.to_csv('data/simulation/trajectories_10*10_repeat.csv',index=False)
    # trajs = read_traj('data/simulation/trajectories_10*10_repeat.csv')
    # print(trajs[0:1])
    # trajs = [edge_node_trans(map_,traj,is_edge=True) for traj in trajs]
    # trajectories_str = ["_".join(map(str,[int(x) for x in traj])) for traj in trajs]
    # new_traj = pd.DataFrame(trajectories_str,columns=['Trajectory'])
    # new_traj.to_csv('data/simulation/trajectories_10*10_repeat_node.csv',index=False)

    trajs = read_traj('data/simulation/trajectories_10*10_repeat_node.csv')
    trajs_edge = read_traj('data/simulation/trajectories_10*10_repeat.csv')
    print(trajs[0])
    max_len = max([len(traj) for traj in trajs_edge])
    print(max_len)
    #results = list(tqdm(map(lambda traj_edge: task4data_process(traj_edge, max_len), trajs_edge), total=len(trajs_edge), desc="Processing trajectories"))
    #results = [task4data_process(traj_edge,max_len) for traj_edge in trajs_edge]
    #results = np.array(results)
    

    dataset1 = SmartTrafficDataset(trajs_edge , mode="task1",is_edge=False)
    dataset2 = SmartTrafficDataset(trajs ,mode="task2",is_edge=False)
    # dataset3 = SmartTrafficDataset(trajs,mode="task3",is_edge=False)
    time0 = time.time()
    #dataset4 = SmartTrafficDataset(None,mode="task4",is_edge=True)
    # print(dataset4[0])
    data_loader1 = SmartTrafficDataloader(dataset1,batch_size=32,shuffle=False)
    data_loader2 = SmartTrafficDataloader(dataset2,batch_size=1,shuffle=False)
    # data_loader3 = SmartTrafficDataloader(dataset3,batch_size=2,shuffle=False)
    time0 = time.time()
    #data_loader4 = SmartTrafficDataloader(dataset4,batch_size=32,shuffle=False)
    print('初始化',time.time()-time0)
    time0 = time.time()
    # for data in data_loader4:
    #     print(time.time()-time0)
    #     time0 = time.time()
    #     print(data.shape)
    #     break
    cfg1 = {
            "vocab_size": 180+1,
            "device": "cuda:0",
            "block_size": dataset1.max_len //2,
            "n_embd": 64,
            "n_head": 4,
            "n_layer": 2,
            "dropout": 0.1,
            "n_hidden": 64,
            "use_agent_mask": True,
            'model_save_path': "weights/model_task1.pth",
            'ta_sliding_window':1,
            'use_model':'sd',
            'use_ne':True
            }
    cfg2 = {
        'device':'cuda',
        'block_size':dataset2.max_len //2, # max length of trajectory
        'n_embd':10,
        'n_head':1,
        'n_layer':1,
        'dropout':0.1,
        'n_hidden':10,
        'n_embed_adj':10,
        'vocab_size':100+1,
        'window_size':2,
        'max_iters':10,
        'learning_rate':0.001,
        'batch_size':1,
        'model_save_path': "weights/model_task2.pth"
        }

    # cfg3 = {
    #     'vocab_size':101,
    #     'n_embd' : 64,
    #     'n_head' : 4,
    #     'n_layer' : 2,
    #     'dropout' : 0.1,
    #     'device' :"cuda",
    #     "block_size":dataset3.max_len //2,
    #     'weight_quantization_scale': 20,
    #     'use_adj_table':True,
    #     'learning_rate':0.0001,
    #     'max_epochs':10,
    #     'observe_ratio':0.5,
    #     'special_mask_value':0.0001
    # }
    # x,_,z,w = dataset3[0]
    # print(x)
    # print(x.shape)
    # print(z)
    # print(z.shape)
    # print(w)
    # print(w.shape)
    # print(type(w))



    #train3(cfg3,data_loader3)


    #i = 0
    #state = pd.read_csv('data/simulation/edge_node_10*10_state.csv')
    #print(state[state['EdgeID_x']==46][state['EdgeID_y']==64]['state'])
    # for data in data_loader4:
    #     i+=1
    #     print(data[0,10 ,9,:])
    #     print(data)
    #     print(torch.sum(data))
    #     print(torch.sum(data,dim = -1)[0,0,:])
    #     if i>1:
    #         break
    train1(cfg1,data_loader1)
    #train2(cfg2,data_loader2)