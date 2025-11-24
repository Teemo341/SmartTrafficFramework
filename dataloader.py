from torch.utils.data import Dataset,DataLoader,Subset
import torch
import numpy as np
import pandas as pd
from utils import padding_zero
import os 
from utils import remove_consecutive_duplicates,node2edge,adj_m2adj_l

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

def read_node_type(file_path='data/simulation/node_type_10*10.csv'):
    data = pd.read_csv(file_path)
    node_type = data['Type'].tolist()
    return node_type

class SmartTrafficDataset(Dataset):
    # trajs: list of list of int
    # map_: np.array([[edge,o,d,length],...])
    # T: int


    def __init__(self,trajs,map_path='data/simulation/edge_node_10*10.csv',
                 #task4_path = 'data/simulation/task4_traj_data.npy',
                 trajs_path = '',
                 time_step_path = '',
                 adjcent_path = None,
                 weight_quantization_scale = 1,
                 max_history = 5,
                 T=None,window_size=1,mode = "task1",max_len=None,task4_num=2000,vocab_size=8909):
        super(SmartTrafficDataset,self).__init__()
       
        self.mode = mode
        self.max_history = max_history
        self.vocab_size = vocab_size
        self.traj_path = trajs_path
        self.time_step_path = time_step_path
        self.weight_quantization_scale = weight_quantization_scale

        if self.mode == "task1":
        #task1
            self.trajs = trajs
            self.window_size = window_size
            self.max_len = max_len if max_len else max([len(traj) for traj in trajs])
            self.T = T if T else self.max_len//2
            self.map = pd.read_csv(map_path)
            self.map = np.array(self.map[['EdgeID','Origin','Destination']])
        #task2
        elif self.mode == "task2":
            self.max_len = max_len if max_len else max([len(traj) for traj in trajs])
            self.trajs = trajs
            self.window_size = window_size
            
            self.adjacent= np.load(adjcent_path) if adjcent_path else simulation2adj(map_path)
            #self.indices , self.values =adj2sparse_adjmatrix_weighted(self.adjacent)
            if len(self.adjacent.shape) == 2:
                self.adjacent = adj_m2adj_l(self.adjacent)
            else:
                self.adjacent = torch.tensor(self.adjacent,dtype=torch.float)

            self.indices , self.values =self.adjacent[:,:,0],self.adjacent[:,:,1]
            if self.weight_quantization_scale is not None:
                self.values = torch.ceil(self.values/self.values.max()*self.weight_quantization_scale)
            
            #self.indices = torch.tensor(self.indices.clone().detach(),dtype=torch.int64)
            self.indices = self.indices.clone().detach().to(torch.int64)

            self.T = T if T else self.max_len-window_size
        #task3
        elif self.mode == "task3":
            self.max_len = max_len if max_len else max([len(traj) for traj in trajs])
            self.trajs = trajs
            self.window_size = window_size
            self.T = T if T else self.max_len//2
            self.adjacent= np.load(adjcent_path) if adjcent_path else simulation2adj(map_path)
            self.adj_l = adj_m2adj_l(self.adjacent)
            self.adj_l = torch.tensor(self.adj_l,dtype=torch.float)
            if self.weight_quantization_scale is not None:
                self.adj_l[:,:,1] = torch.ceil(self.adj_l[:,:,1]/self.adj_l[:,:,1].max()*self.weight_quantization_scale)
        #task4
        elif self.mode == "task4":

            self.task4_num = task4_num
            self.task4_path = trajs_path
            self.trajs = trajs
            # if self.trajs is not None:
            #     self.max_len = self.trajs.shape[1]
         

    def __len__(self):
        if self.mode == "task4":
            # return 4*256*100
            if self.trajs is not None and isinstance(self.trajs,str):
                return len(os.listdir(self.trajs))
            else:
                return len(self.trajs) if self.trajs is not None else len(os.listdir(self.task4_path))
            # return len(self.trajs)//10 if self.trajs is not None else len(os.listdir(self.traj_path))
        return len(self.trajs) if self.trajs else len(os.listdir(self.traj_path))
    
    def __getitem__(self,idx):
        
        if self.mode == 'task4':
            # data:[T,V,7]
            if self.trajs is not None:
                if isinstance(self.trajs,str):
                    # print(self.trajs)
                    path = os.listdir(self.trajs)
                    choice = np.random.choice(len(path),size = self.task4_num)
                    choice_path = [os.path.join(self.trajs,path[i]) for i in choice]
                    for i in range(len(choice_path)):
                        trajs = np.load(choice_path[i])
                        if i == 0:
                            results = trajs
                        else:
                            results += trajs
                    results = torch.tensor(results, dtype=torch.int)
                    return results 
                else:
                    idx = np.arange(0,len(self.trajs))#.astype(np.int)
                    choice  = np.random.choice(idx,size = self.task4_num)
                    selected_trajs = [self.trajs[i] for i in choice]
                    results = np.sum(selected_trajs,axis=0)
                    return torch.tensor(results, dtype=torch.int)
            if self.trajs is None:
                trajs = np.load(self.traj_path+str(idx+1)+'.npy')
                return torch.tensor(trajs, dtype=torch.int)        
        else:
            if self.T + self.window_size > self.max_len:
                raise ValueError('Error: T + window_size is too large')
            if self.trajs is None:
                traj = np.load(self.traj_path+str(idx+1)+'.npy')
                traj_o = np.load(self.traj_path+str(idx+1)+'.npy')
            else:
                traj = self.trajs[idx]
        
        if self.mode != 'task3':
            # if len(traj.shape) == 1:
            #     traj = traj[None,:]
            #     traj_o = traj_o[None,:] 
            # index = np.random.randint(0,len(traj))  
            # traj = traj[:,index]
            # traj_o = traj_o[:,index]
    
            traj = padding_zero(traj,self.max_len)
            traj_ = traj[0:self.T]
            traj_targ = traj[self.window_size:self.T+self.window_size]
            reagent_mask_ = [ 1 if x!=0 else 0 for x in traj ]
            reagent_mask_ = torch.tensor(reagent_mask_,dtype=torch.int64).view(-1,1)

            reagent_mask = [ 1 if x!=0 else 0 for x in traj_ ]
            reagent_mask = torch.tensor(reagent_mask,dtype=torch.int64).view(-1,1)

            valid_length = torch.tensor([torch.sum(reagent_mask)],dtype=torch.int64)
            od = torch.tensor([traj_o[0],traj_o[-1]],dtype=torch.int64)
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
                # incides and values: (2, num_edges)
            
                o = od[0,0,0].item() 
                d = od[0,0,1].item()
                
                od[:,:,0]=d
                od[:,:,1]=o
                #print('dataset',od)
                return traj_ ,valid_length,od,traj_targ,self.indices,self.values,reagent_mask
        
        if self.mode == "task3":
            # traj: (1, T, 1)
            #time_step(not uesd) is set [0]
            #reagent_mask: (1, T, 1)
            #adj_l: (1, V , max_connection , 2)
            # time_step = np.load(self.time_step_path+str(idx+1)+'.npy')
            time_step=  np.array([0]) 
            traj_list = []
            traj_targ_list = []
            traj_mask_list = []
            time_step_list = []
            if len(traj.shape) ==1:
                traj = traj[None,:]
            for j in range(self.max_history):
                i = j
                if i > traj.shape[0]-1:
                    i = np.random.randint(0,traj.shape[0])
                traj_=padding_zero(traj[i,:],self.max_len)
                traj_ = traj_[0:self.T]
                traj_list.append(traj_[0:self.T])
                traj_targ_list.append(traj_[self.window_size:self.T+self.window_size])
                reagent_mask_ = [ 1 if x!=0 else 0 for x in traj_]
                traj_mask_list.append(reagent_mask_)
                time_step_list.append(time_step[0])
                

            traj_ = torch.tensor(np.array(traj_list),dtype=torch.int64)
            traj_targ = torch.tensor(np.array(traj_targ_list),dtype=torch.int64)
            reagent_mask = torch.tensor(np.array(traj_mask_list),dtype=torch.int64)
            time_step = torch.tensor(np.array(time_step_list),dtype=torch.int64)
            return traj_, time_step, reagent_mask , self.adj_l.unsqueeze(0).repeat(traj_.shape[0],1,1,1)

    
    def __getattribute__(self, name: str):
        return super().__getattribute__(name)

class SmartTrafficDataloader(DataLoader):
    def __init__(self,dataset,batch_size=1,max_len=203,vocab_size=23313,shuffle=False,x=1000000,y=1000000 ,**kwargs):
        super(SmartTrafficDataloader,self).__init__(dataset,batch_size=batch_size,shuffle=shuffle, **kwargs)
        self.max_len = max_len
        self.vocab_size = vocab_size
        x = min(x,dataset.__len__()*8//10)
        y = min(y,dataset.__len__()*8//10)
        self.train_dataset = [i for i in range(x)]
        self.train_dataset = Subset(dataset,self.train_dataset)
        self.test_dataset = [i for i in range(y,dataset.__len__())]
        self.test_dataset = Subset(dataset,self.test_dataset)

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
    
    def get_train_data(self):
        return SmartTrafficDataloader(self.train_dataset,batch_size=self.batch_size,shuffle=False)
    
    def get_test_data(self):
        return SmartTrafficDataloader(self.test_dataset,batch_size=self.batch_size,shuffle=False)
    


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

    #results = list(tqdm(map(lambda traj_edge: task4data_process(traj_edge, max_len), trajs_edge), total=len(trajs_edge), desc="Processing trajectories"))
    #results = [task4data_process(traj_edge,max_len) for traj_edge in trajs_edge]
    #results = np.array(results)
    
    #load data
    # trajs = read_traj('data/simulation/trajectories_10*10_repeat_node.csv') #trajs of nodes
    # trajs_edge = read_traj('data/simulation/trajectories_10*10_repeat.csv') #trajs of edges
    # trajs_not_repeat = read_traj('data/simulation/trajectories_10*10.csv')

    # dataset1 = SmartTrafficDataset(trajs_edge , mode="task1")
    # dataset2 = SmartTrafficDataset(trajs_not_repeat ,mode="task2")
    # dataset3 = SmartTrafficDataset(trajs,mode="task3")

    # data_loader1 = SmartTrafficDataloader(dataset1,batch_size=32,shuffle=False)
    # data_loader2 = SmartTrafficDataloader(dataset2,batch_size=256,shuffle=False)
    # data_loader3 = SmartTrafficDataloader(dataset3,batch_size=32,shuffle=False)

    #configs
    # cfg1 = {
    #         "vocab_size": 180+1,
    #         "device": get_local_device,
    #         "block_size": dataset1.max_len //2,
    #         "n_embd": 64,
    #         "n_head": 4,
    #         "n_layer": 2,
    #         "dropout": 0.1,
    #         "n_hidden": 64,
    #         'model_save_path': "weights/model_task1.pth",
    #         }
    # cfg2 = {
    #     'device':get_local_device,
    #     'block_size':dataset2.max_len //2, # max length of trajectory
    #     'n_embd':64,
    #     'n_head':4,
    #     'n_layer':2,
    #     'dropout':0.1,
    #     'n_hidden':64,
    #     'n_embed_adj':64,
    #     'vocab_size':100+1,
    #     'window_size':1,
    #     'max_epochs':1000,
    #     'learning_rate':0.001,
    #     'batch_size':256,
    #     'model_save_path': "weights/model_task2.pth"
    #     }
    # cfg2 = {
    #     'device':get_local_device,
    #     'block_size':dataset2.max_len-1, # max length of trajectory
    #     'n_embd':20,
    #     'n_head':4,
    #     'n_layer':8,
    #     'dropout':0.1,
    #     'n_hidden':16,
    #     'n_embed_adj':20,
    #     'vocab_size':100+1,
    #     'window_size':1,
    #     'max_epochs':20,
    #     'learning_rate':0.001,
    #     'batch_size':256,
    #     'model_read_path': None,
    #     'model_save_path': "weights/model_task2_1.pth"
    #     }

    # cfg3 = {
    #     'vocab_size':101,
    #     'n_embd' : 64,
    #     'n_head' : 4,
    #     'n_layer' : 2,
    #     'dropout' : 0.1,
    #     'device' :get_local_device,
    #     "block_size":dataset3.max_len //2,
    #     'weight_quantization_scale': 20,
    #     'use_adj_table':True,
    #     'learning_rate':0.001,
    #     'max_epochs':100,
    #     'observe_ratio':0.5,
    #     'special_mask_value':0.0001,
    #     'model_save_path': "weights/model_task3.pth"
    # }

    #train1(cfg1,data_loader1)

    #print('max_len:',dataset2.max_len)
    #train2(cfg2,data_loader2)
    
    #train3(cfg3,data_loader3)
    #print(dataset2[0][2].shape)
    dataset1 = SmartTrafficDataset(None,mode="task1",T=60,max_len=63,
                                    trajs_path='data/jinan/edge_traj_repeat_one_by_one/')
    print(dataset1[0])
    dataloader1 = SmartTrafficDataloader(dataset1,batch_size=32,shuffle=False)
    import time
    time0 = time.time()
    for data in dataloader1:
        print(data['traj'].shape)
        print(data['cond'].shape)
        print(data['traj_targ'].shape)
        print(data['reagent_mask'].shape)
        print(time.time()-time0)
        break
        
    # adjcent = np.load('data/jinan/adjcent.npy')
    # print(adjcent.shape)
    # print(adjcent)
    # num = 1
    # for i in range(len(adjcent)):
    #     for j in range(i,len(adjcent)):
    #         if adjcent[i][j] != 0:
    #             num+=1
    # print(num)
    # dataset3 = SmartTrafficDataset( None,mode="task3",T=60,max_len=63,
    #                                 time_step_path ='data/jinan/jinan_time_step/', 
    #                                 trajs_path='data/jinan/node_traj_repeat_one_by_one/',
    #                                 adjcent_path='data/jinan/adjcent.npy')
    # for i in range(100):
    #     print(dataset3[i][0].shape)
    #     print(dataset3[i][1].shape)
    #     print(dataset3[i][2].shape)
    #     print(dataset3[i][3].shape)
    #     break
    # dataloader3 = SmartTrafficDataloader(dataset3,batch_size=32,shuffle=False)
    # for data in dataloader3:
    #     print(data[0].shape)
    #     print(data[2].shape)
    #     print(data[3].shape)

    # map = pd.read_csv('data/jinan/edge_node_jinan.csv')
    # print(map['EdgeID','Origin','Destination'].values)
    # path = 'data/jinan/edge_traj_repeat_one_by_one/'
    # files = os.listdir(path)
    # print(len(files))
    # path =  '/home/shenshiyu/SmartTrafficFramework/weights/jinan/task2/best_model_0.0880.pth'
    # print(path[-10:-4])
#     dataset = SmartTrafficDataset(trajs = None,mode="task3",T=60,max_len=120,adjcent_path='data/jinan/adjcent_class.npy',trajs_path='data/jinan/node_traj_test/')
# #dataset = dataset[:100]
#     print(dataset[0])   
#     dataloader = SmartTrafficDataloader(dataset,batch_size=1,shuffle=False, num_workers=4)
#     dataloader.randomize_condition()
#     for condition, time_step, special_mask, adj_table in dataloader:
#         print(condition.shape)
#         print(time_step)
#         print(special_mask.shape)
#         print(adj_table.shape)
#         break
    # dataset = SmartTrafficDataset(trajs = None,mode="task1",trajs_path='data/jinan/edge_traj_new/',need_repeat=True,max_len=203,T=200)
    # dataloader = SmartTrafficDataloader(dataset,batch_size=1,max_len=203,shuffle=False, num_workers=4,vocab_size=23313)
    # train_dataloader = dataloader.get_train_data()
    # test_dataloader = dataloader.get_test_data()
    # for data in train_dataloader:
    #     print(data['traj'])
    #     print(data['cond'])
    #     print(data['traj_targ'])
    #     print(data['reagent_mask'])
    #     break
 
        