import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.sparse.csgraph import dijkstra
from torch.utils.data import Dataset, DataLoader
from task2.ma_model import SpatialTemporalCrossMultiAgentModel
import torch
import random
import os

# node_path = 'data/jinan/node_jinan.csv'
# edge_path = 'data/jinan/edge_jinan.csv'


# node = pd.read_csv(node_path)
# edge = pd.read_csv(edge_path)
# edge_node = []
# print(node.columns)
# print(edge.columns)
# for i in range(len(edge)):
#     edge_node.append([i+1,edge.loc[i, 'Origin'], edge.loc[i, 'Destination'], edge.loc[i, 'Length'], edge.loc[i, 'Class'],\
#                       node[node['NodeID']==edge.loc[i, 'Origin']]['Longitude'], node[node['NodeID']==edge.loc[i, 'Origin']]['Latitude'],\
#                         node[node['NodeID']==edge.loc[i, 'Destination']]['Longitude'], node[node['NodeID']==edge.loc[i, 'Destination']]['Latitude']])
    
# edge_node = pd.DataFrame(edge_node, columns=['EdgeID', 'Orgin', 'Destination', 'Length', 'Class', 'Orgin_Longitude', 'Orgin_Latitude', 'Destination_Longitude', 'Destination_Latitude'])
# edge_node.to_csv('data/jinan/edge_node_jinan.csv', index=False)

def get_model(cfg):

    device = cfg['device']
    block_size, n_embd, n_head, n_layer, dropout = cfg['block_size'], cfg['n_embd'], cfg['n_head'], cfg['n_layer'], cfg['dropout']
    n_hidden = cfg['n_hidden']
    n_embed_adj = n_embd
    vocab_size = cfg['vocab_size']
    window_size = cfg['window_size']
    
    model = SpatialTemporalCrossMultiAgentModel(vocab_size, 
                                                    n_embd, 
                                                    n_hidden, 
                                                    n_layer, 
                                                    n_head, 
                                                    block_size,
                                                    n_embed_adj,
                                                    window_size=window_size,
                                                    dropout=dropout, 
                                                    device=device,
                                                    )
    return model.to(device)
    
def boston2adj(file_path='data/jinan/edge_node_jinan.csv', output_csv_file='data/jinan/edges_mapping.csv', output_adj_file='data/jinan/adjacent.npy'):
    
    u_nodes = []
    v_nodes = []
    lengths = []
    data = pd.read_csv(file_path)
    for i in range(len(data)):
        u_nodes.append(data.loc[i, 'Orgin'])
        v_nodes.append(data.loc[i, 'Destination'])
        lengths.append(data.loc[i, 'Length'])

    num_of_nodes = np.max((np.max(u_nodes), np.max(v_nodes)))
    adj_matrix = np.zeros((num_of_nodes + 1, num_of_nodes + 1))
    for u, v, length in zip(u_nodes, v_nodes, lengths):
        adj_matrix[u][v] = length
    np.save(output_adj_file, adj_matrix)


def temp_generate_trajectory(n_traj):
    
    adjacent = np.load('data/jinan/adjacent.npy')
    trajs = []

    for _ in range(n_traj):
        traj = None
        while traj is None:
            # in adjacent, index start from 0, but we want start from 1, so when appending index, we add 1
            ori, des = np.random.randint(0, adjacent.shape[0], (2))
            _, predecessors = dijkstra(csgraph=adjacent, indices=ori, return_predecessors=True)
            if predecessors[des] == -9999:
                continue
            traj = [des+1]
            while traj[0] != ori+1:
                pre = predecessors[des]
                traj.insert(0, pre+1)
                des = pre
        trajs.append(traj)
    return trajs

def zero_padding(traj,max_len):
    return traj + [0] * (max_len - len(traj))

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

class Dijkstra_Dataset(Dataset):
    def __init__(self,adjcent_path = 'data/jinan/adjacent.npy',traj_path ='data/jinan/traj.npy',vaild_length_path ='data/jinan/valid_length.npy'):
        self.adjacent = np.load(adjcent_path)
        self.trajs = np.load(traj_path)
        self.vaild_length = np.load(vaild_length_path)
        self.indices , self.values = adj2sparse_adjmatrix_weighted(self.adjacent)
        self.data_x = self.trajs[:, :-1,None]
        self.data_y = self.trajs[:, 1:,None]
        od_condation = (self.trajs[np.arange(self.trajs.shape[0]), self.vaild_length - 1])[:,None]
        self.od_condation = np.repeat(od_condation[:,None,:],self.data_x.shape[1],axis=-2)[:,:,None,:]

        
    def __getitem__(self, index):
        return self.data_x[index],self.vaild_length[index],self.od_condation[index],self.data_y[index],self.indices,self.values
    
    def __len__(self):
        return len(self.trajs)
    
def data_loader(batch_size,mode = 'train'):
    data = Dijkstra_Dataset()
    data_loader = DataLoader(data, 
                             batch_size=batch_size, 
                             shuffle=True,
                             drop_last=True if mode == 'train' else False
                             )
    return data_loader

def estimate_loss(model, train_iter, 
                  val_iter, device='cuda'):
    out = {}
    loader = ['train', 'val']
    device = model.module.device if isinstance(model,torch.nn.DataParallel) else model.device
    model.eval()
    for i, dataloader in enumerate([train_iter, val_iter]):
        losses = 0
        iter_count = 0
        for x, x_valid, od_condition, y, adj_indices, adj_values in dataloader:
            iter_count += 1
            x, x_valid, od_condition, y, adj_indices, adj_values = x.to(device), x_valid.to(device), od_condition.to(device),\
                                                                y.to(device), adj_indices[0].unsqueeze(0).to(device), adj_values.to(device)
            y = y.long()
            logits, loss = model(x, x_valid, y, condition=od_condition, adj=(adj_indices, adj_values))

            loss = loss.mean().item()

            losses += loss
            #torch.cuda.empty_cache()
        out[loader[i]] = losses / iter_count
        #torch.cuda.empty_cache()
    model.train()
    
    return out


def train(cfg , data_loader):

    random.seed(1337)
    np.random.seed(1337)
    torch.manual_seed(1337)
    max_epochs = cfg['epochs']
    learning_rate = cfg['learning_rate']
    device = cfg['device']
    train_iter = data_loader

    model = get_model(cfg)
    old_path = None
    if cfg['model_read_path']:
        model.load_state_dict(torch.load(cfg['model_read_path']))
        if 'best_model' in cfg['model_read_path'] or 'last_model' in cfg['model_read_path']:
            try:
                last_loss = float(cfg['model_read_path'][-10:-4])
                old_path = cfg['model_read_path']
            except:
                last_loss = 10000
    else:
        last_loss = 10000
    if cfg['model_read_path']:
        model.load_state_dict(torch.load(cfg['model_read_path']))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    lr_sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.99)
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    for it in range(max_epochs):
        iter_count = 0
        mean_loss = 0

        for x, x_valid, od_condition, y, adj_indices, adj_values,mask in tqdm(train_iter, desc=f"Epoch {it + 1}/{max_epochs}"):
            iter_count += 1
            x, x_valid, od_condition, y, adj_indices, adj_values,mask = x.to(device), x_valid.to(device), od_condition.to(device),\
                                                                y.to(device), adj_indices[0].unsqueeze(0).to(device), adj_values.to(device),mask.to(device)
            y = y.long()
            logits, _ = model(x, x_valid, y, condition=od_condition, adj=(adj_indices, adj_values))
            target = y
            logits = logits.squeeze(2)
            target = target.squeeze(-1).squeeze(-1)
            loss = criterion(logits.view(-1,logits.shape[-1]), target.view(-1))
            mask = mask.squeeze(-1).float()
            loss = loss.view(mask.shape)
            loss = loss * mask
            loss = loss.sum() / mask.sum()
            
            # print(logits)
            # print(logits.shape)   
            #loss = loss.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            #a = mean_loss
            mean_loss += loss.item()
            #print(mean_loss - a)
            optimizer.step()
            lr_sched.step()

        mean_loss /= iter_count
        print(f'epoch {it+1}, Loss: {mean_loss}, LR: {lr_sched.get_last_lr()[0]}')
        if os.path.isdir(cfg['model_save_path']):
            path = os.path.join(cfg['model_save_path'],f"best_model_{mean_loss:.4f}.pth")
            if mean_loss < last_loss:
                if old_path:
                    os.remove(old_path)
                last_loss = mean_loss
                torch.save(model.state_dict(), path)
                old_path = path

    if os.path.isdir(cfg['model_save_path']):
        path = os.path.join(cfg['model_save_path'],f"last_model_{mean_loss:.4f}.pth")
        torch.save(model.state_dict(), path)

    

    # generate adjacent matrix
    # trajs = temp_generate_trajectory(10)
    # max_len = max([len(traj) for traj in trajs])
    # vaild_length = [len(traj) for traj in trajs]
    # trajs = [zero_padding(traj, max_len) for traj in trajs]
    # np.save('data/jinan/traj.npy', trajs)
    # np.save('data/jinan/valid_length.npy', vaild_length)
    # test Dijkstra_Dataset
    # data = Dijkstra_Dataset()
    # print(data[0])
    #test dataloader
    # data = data_loader(1)
    # for x, x_valid, od_condition, y, adj_indices, adj_values in data:
    #     print(x.shape)
    #     print(x_valid.shape)
    #     print(od_condition.shape)
    #     print(y.shape)
    #     print(adj_indices.shape)
    #     print(adj_values.shape)
    #     break
if __name__=='__main__':
    
    cfg = {
        'device':'cuda',
        'block_size':95, # max length of trajectory
        'n_embd':10,
        'n_head':1,
        'n_layer':1,
        'dropout':0.1,
        'n_hidden':10,
        'n_embed_adj':10,
        'vocab_size':8909,
        'window_size':10,
        'max_iters':10,
        'learning_rate':0.001,
        'batch_size':1
        }
    #train(cfg)
    