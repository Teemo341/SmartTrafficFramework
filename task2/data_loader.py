import os
import numpy as np
import torch
import pickle
import os
from torch.utils.data import Dataset, DataLoader
from torch.distributed import get_rank

class RealWorld_Dataset(Dataset):
    def __init__(self, root_path='./real_data', graph_path='jinan/adj.pkl', flag='train', 
                 data_path='jinan/data.npy', length_path='jinan/valid_length.npy',
                 distance_path='jinan/distance.npy', use_ddp=False):
        # Initialize
        assert flag in ['train', 'val', 'test']
        self.type_map = {'train': 0, 'val': 1, 'test': 2}
        self.flag_map = {0: 'train', 1: 'val', 2: 'test'}
        self.set_type = self.type_map[flag]
        self.hop = 0
        self.root_path = root_path
        self.graph_path = os.path.join(root_path, graph_path)
        self.data_path = os.path.join(root_path, data_path)
        self.length_path = os.path.join(root_path, length_path)
        self.distance_path = os.path.join(root_path, distance_path)

        self.use_ddp = use_ddp
        data = np.load(self.data_path)

        num_vali = int(len(data) * 0.1)
        num_test = int(len(data) * 0.2)
        num_train = len(data) - num_vali - num_test

        border1s = [0, num_train, len(data) - num_test]
        border2s = [num_train, num_train + num_vali, len(data)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        valid_length = np.load(self.length_path)
        distance = np.load(self.distance_path)
        # od_condition: [batch_size, 1]
        od_condition = (data[np.arange(data.shape[0]), valid_length - 1])[:, None]
        
        self.data_x = data[border1:border2, :-1][:, : ,None]
        self.data_y = data[border1:border2, 1:][:, :, None]
        self.valid_length = valid_length[border1:border2][:, None]
        self.distance = distance[border1:border2]
        # od_condition: [batch_size, 1]
        self.od_condition = od_condition[border1:border2]
        self.od_condition = np.repeat(self.od_condition[:, None, :], self.data_x.shape[1], axis=-2)[:, :, None, :]

        # adj_matrix: [num_graphs, V, V]
        adj_list = pickle.load(open(self.graph_path, 'rb'))

        self.adj_values_max = 0.0
        for node, neighbors in adj_list[0].items():
            if neighbors:
                weights = []
                for weight in neighbors.values():
                    weights.append(weight['weight'])
                max_weight = max(weights)
                if max_weight > self.adj_values_max:
                    self.adj_values_max = max_weight

        self.adj_indices, self.adj_values = adjlist2sparse_adjmatrix_weighted(adj_list, self.hop, 
                                                                              use_ddp=self.use_ddp,
                                                                              padding_value=self.adj_values_max)
        self.adj_values = self.adj_values[0]
        #self.adj_values /= self.adj_values_max

        self.len = len(self.data_x)
        print('Num of %s trajs: %d' % (self.flag_map[self.set_type], self.len))

    def __getitem__(self, index):
        if self.flag_map[self.set_type] in ['train', 'val']:
            return self.data_x[index], self.valid_length[index], self.od_condition[index], \
                self.data_y[index], self.adj_indices, self.adj_values / self.adj_values_max
        else:
            return self.data_x[index, 0][None, :], self.valid_length[index], \
                self.od_condition[index, 0][None, :, :], self.data_y[index], \
                self.adj_indices, self.adj_values, self.distance[index]
        
    def __len__(self):
        return self.len

class Dijkstra_Dataset(Dataset):
    def __init__(self, root_path='.', graph_path='graph/boston_100.pkl', flag='train', 
                 data_path='data/data_100', length_path='data/valid_length_100', 
                 od_per_graph=1000, hop=1, use_ddp=False, mode='train', num_file=100):
        # Initialize
        assert flag in ['train', 'val']
        type_map = {'train': 0, 'val': 1}
        self.set_type = type_map[flag]

        self.root_path = root_path
        self.graph_path = os.path.join(root_path, graph_path)
        self.data_path = os.path.join(root_path, data_path)
        self.length_path = os.path.join(root_path, length_path)
        self.hop = hop
        self.od_per_graph = od_per_graph
        self.use_ddp = use_ddp
        self.mode = mode
        self.num_of_file = num_file
        self.graph_per_file = 100
        self.num_of_graph = num_file * self.graph_per_file
        self.num_train_file = int(0.9 * num_file)
        self.num_val_file = num_file - self.num_train_file

        self.data_x = []
        self.data_y = []
        self.adj_values = []
        self.valid_length = []
        self.od_condition = []

        if self.set_type:
            start = self.num_train_file
            end = self.num_train_file + self.num_val_file
        else:
            start = 0
            end = self.num_train_file

        if self.mode == 'test':
            self.graph_per_file = 10
            self.traj_per_file = self.od_per_graph * self.graph_per_file
            a_data = np.load(os.path.join(self.data_path, "test.npy"))
            valid_length = np.load(os.path.join(self.length_path, "test.npy"))
            self.data_x.append(a_data[:, :-1, None])
            self.data_y.append(a_data[:, 1:, None])
            self.valid_length.append(valid_length[:, None])
            
            od_condition = (a_data[np.arange(a_data.shape[0]), valid_length - 1])[:, None]
            self.od_condition.append(np.repeat(od_condition[:, None, :], self.data_x[-1].shape[1], axis=-2)[:, :, None, :])

            adj_list = pickle.load(open(os.path.join(self.graph_path, "test.pkl"), 'rb'))
            adj_indices, adj_values = adjlist2sparse_adjmatrix_weighted(adj_list, self.hop, use_ddp=self.use_ddp)
            self.adj_indices = adj_indices
            self.adj_values.append(adj_values)
            self.len = int(self.graph_per_file * self.od_per_graph)
            print(f"Num of test graph: {self.graph_per_file}, Num of test trajs: {self.traj_per_file}")
        else:
            self.traj_per_file = self.od_per_graph * self.graph_per_file
            for file_id in range(start, end):
                a_data = np.load(os.path.join(self.data_path, f"{file_id}.npy"))
                valid_length = np.load(os.path.join(self.length_path, f"{file_id}.npy"))
                self.data_x.append(a_data[:, :-1, None])
                self.data_y.append(a_data[:, 1:, None])
                self.valid_length.append(valid_length[:, None])
                
                od_condition = (a_data[np.arange(a_data.shape[0]), valid_length - 1])[:, None]
                self.od_condition.append(np.repeat(od_condition[:, None, :], self.data_x[-1].shape[1], axis=-2)[:, :, None, :])

                adj_list = pickle.load(open(os.path.join(self.graph_path, f"{file_id}.pkl"), 'rb'))
                adj_indices, adj_values = adjlist2sparse_adjmatrix_weighted(adj_list, self.hop, use_ddp=self.use_ddp)
                self.adj_indices = adj_indices
                self.adj_values.append(adj_values)

            if self.set_type:
                self.len = int(self.num_val_file * self.graph_per_file * self.od_per_graph)
                print(f"Num of val graph: {self.num_val_file * 100}, Num of val trajs: {self.num_val_file * self.traj_per_file}")
            else:
                self.len = int(self.num_train_file * self.graph_per_file * self.od_per_graph)
                print(f"Num of train graph: {self.num_train_file * 100}, Num of train trajs: {self.num_train_file * self.traj_per_file}")

    def __getitem__(self, index):
        file_id, traj_id = index // self.traj_per_file, index % self.traj_per_file

        if self.mode == 'train':
            return (self.data_x[file_id][traj_id], self.valid_length[file_id][traj_id], 
                    self.od_condition[file_id][traj_id], self.data_y[file_id][traj_id], 
                    self.adj_indices, self.adj_values[file_id][traj_id // self.od_per_graph])
        elif self.mode == 'test':
            return (self.data_x[file_id][traj_id, 0][None, :], self.valid_length[file_id][traj_id], 
                    self.od_condition[file_id][traj_id, 0][None, :, :], self.data_y[file_id][traj_id], 
                    self.adj_indices, self.adj_values[file_id][traj_id // self.od_per_graph])
    
    def __len__(self):
        return self.len

class Simulator_Dataset(Dataset):
    def __init__(self, root_path='.', graph_path='graph/boston_100.pkl', flag='train', 
                 data_path='data/data_100.npy', length_path='data/valid_length_100.npy', 
                 od_per_graph=100000, hop=1, use_ddp=False, mode='train'):
        # init
        assert flag in ['train', 'val']
        type_map = {'train': 0, 'val': 1}
        self.set_type = type_map[flag]

        self.root_path = root_path
        self.graph_path = graph_path
        self.data_path = data_path
        self.length_path = length_path
        self.hop = hop
        self.od_per_graph = od_per_graph
        self.use_ddp = use_ddp
        self.mode = mode
        self.__read_data__()

    def __read_data__(self):
        # data: [batch_size, max_token_length]
        data = np.load(os.path.join(self.root_path, self.data_path))
        self.max_token_length = data.shape[-1]

        num_train = int(len(data) * 0.9)
        num_vali = len(data) - num_train

        border1s = [0, len(data) - num_vali]
        border2s = [num_train, len(data)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # adj_matrix: [num_graphs, V, V]
        adj_list = pickle.load(open(os.path.join(self.root_path, \
                    self.graph_path), 'rb'))[border1//self.od_per_graph:border2//self.od_per_graph]

        self.adj_indices, self.adj_values = adjlist2sparse_adjmatrix_weighted(adj_list, self.hop, use_ddp=self.use_ddp)

        # valid_length: [batch_size,]
        valid_length = np.load(os.path.join(self.root_path, self.length_path))
        
        # od_condition: [batch_size, 1]
        od_condition = (data[np.arange(data.shape[0]), valid_length - 1])[:, None]
        
        self.data_x = data[border1:border2, :-1][:, : ,None]
        if not self.use_ddp or self.use_ddp and get_rank() == 0:
            print("Num traj: ", len(self.data_x))

        self.data_y = data[border1:border2, 1:][:, :, None]
        self.valid_length = valid_length[border1:border2][:, None]

        # od_condition: [batch_size, 1]
        self.od_condition = od_condition[border1:border2]
        self.od_condition = np.repeat(self.od_condition[:, None, :], self.data_x.shape[1], axis=-2)[:, :, None, :]
        
    def __getitem__(self, index):
        if self.mode == 'train':
            return self.data_x[index], self.valid_length[index], self.od_condition[index], \
                self.data_y[index], self.adj_indices, self.adj_values[index//self.od_per_graph]
        elif self.mode == 'test':
            return self.data_x[index, 0][None, :], self.valid_length[index], \
                self.od_condition[index, 0][None, :, :], self.data_y[index], \
                self.adj_indices, self.adj_values[index//self.od_per_graph]

    def __len__(self):
        return len(self.data_x)

def get_realworld_dataloader(batch_size=1, root_path='./real_data', graph_path='jinan/adj.pkl',
                             data_path='jinan/data.npy', length_path='jinan/valid_length.npy',
                             distance_path='jinan/distance.npy', flag='train', shuffle_flag=True, 
                             num_workers=10, use_ddp=False, pin_memory=True):
    data_set = RealWorld_Dataset(
        root_path=root_path,
        graph_path=graph_path,
        data_path=data_path,
        length_path=length_path,
        distance_path=distance_path,
        flag=flag,
        use_ddp=use_ddp,
        )

    if use_ddp:
        sampler = torch.utils.data.distributed.DistributedSampler(data_set)
        shuffle = False
    else:
        sampler = None
        shuffle = shuffle_flag

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=pin_memory,
        sampler=sampler
    )

    return (data_loader, data_set.adj_values_max) if flag=='test' else data_loader

def get_dijkstra_dataloader(batch_size=1, root_path='.', graph_path='./data/paris/adj', 
                            data_path='./data/paris/data', length_path='./data/paris/valid_length', 
                            flag='train', shuffle_flag=True, num_workers=10,
                            drop_last=True, hop=1, use_ddp=False, pin_memory=True,
                            od_per_graph=1000, mode='train', num_file=100):
    data_set = Dijkstra_Dataset(
        root_path=root_path,
        graph_path=graph_path,
        data_path=data_path,
        length_path=length_path,
        flag=flag, hop=hop, 
        od_per_graph=od_per_graph, 
        use_ddp=use_ddp,
        mode=mode,
        num_file=num_file,
        )

    if use_ddp:
        sampler = torch.utils.data.distributed.DistributedSampler(data_set)
        shuffle = False
    else:
        sampler = None
        shuffle = shuffle_flag

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True if flag=='train' else False,
        pin_memory=pin_memory,
        sampler=sampler
    )

    return data_loader

def get_simulator_dataloader(batch_size=1, root_path='.', graph_path='./data/paris/weighted_adj.pkl', 
                            data_path='./data/paris/data_weighted.npy', length_path='./data/paris/valid_length_weighted.npy', 
                            flag='train', shuffle_flag=True, num_workers=10,
                            drop_last=True, hop=1, use_ddp=False, pin_memory=True,
                            od_per_graph=1000, mode='train'):
    data_set = Simulator_Dataset(
        root_path=root_path,
        graph_path=graph_path,
        data_path=data_path,
        length_path=length_path,
        flag=flag, hop=hop, 
        od_per_graph=od_per_graph, 
        use_ddp=use_ddp,
        mode=mode,
        )

    if use_ddp:
        sampler = torch.utils.data.distributed.DistributedSampler(data_set)
        shuffle = False
    else:
        sampler = None
        shuffle = shuffle_flag

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True if flag=='train' else False,
        pin_memory=pin_memory,
        sampler=sampler
    )

    return data_loader

def adjlist2sparse_adjmatrix_weighted(adj_list, hop, use_ddp, padding_value=1):
    if isinstance(adj_list, list):
        V = len(adj_list[0]) + 1
        num_graphs = len(adj_list)
    else:
        V = len(adj_list) + 1
        adj_list = [adj_list]
        num_graphs = 1
    #if not use_ddp or use_ddp and get_rank() == 0:
    #    print("Vocab Size: ", V, " Num Graphs: ", num_graphs)

    max_degree = 0
    for i in range(V-1):
        max_degree = max(max_degree, len(adj_list[0][i+1]))

    indices = np.zeros((V, max_degree+2), dtype=np.int32)
    values = np.zeros((len(adj_list), V, max_degree+2), dtype=np.float32)

    for i in range(num_graphs):
        count = np.zeros(V, dtype=np.int32)
        for j, neighbor in adj_list[i].items():
            for k, v in neighbor.items():
                # new neighbor and weight
                indices[j][count[j]] = k
                values[i][j][count[j]] = v['weight']
                count[j] += 1
            # self loop
            indices[j][count[j]] = j
            values[i][j][count[j]] = padding_value
            count[j] += 1
            # edge to padding
            values[i][j][count[j]] = padding_value
        
        values[i][0][0] = padding_value

    return indices, values

def normalize_sparse_adj(sparse_matrix, hop=1):
    D = torch.sparse.sum(sparse_matrix, dim=1).to_dense()
    D = torch.pow(D, -0.5)
    D = torch.diag(D)
    
    sparse_matrix = torch.sparse.mm(D, sparse_matrix)
    sparse_matrix = torch.sparse.mm(sparse_matrix, D)
    for _ in range(hop-1):
        sparse_matrix = torch.sparse.mm(sparse_matrix, sparse_matrix)

    return sparse_matrix

def normalize_adj(adj_matrix, hop=1):
    D = np.sum(adj_matrix, axis=-1) ** -0.5
    identity_matrix = np.eye(D.shape[-1])
    D = D[:, :, None] * identity_matrix[None, :, :]

    adj_matrix = np.einsum('ijk,ikl->ijl', D, adj_matrix)
    adj_matrix = np.einsum('ijk,ikl->ijl', adj_matrix, D)
    for _ in range(hop-1):
        adj_matrix = np.einsum('ijk,ikl->ijl', adj_matrix, adj_matrix)

    return adj_matrix

if __name__ == '__main__':
    adj = pickle.load(open('data/paris/adj/0.pkl', 'rb') )
    print(adj)
    exit()
    dataloader = get_simulator_dataloader(flag='val')
    
    for x, x_valid, od_condition, adj, y in dataloader:
        print(adj)
        exit()
        indices = adj.nonzero(as_tuple=False).t()
        values = adj[indices[0], indices[1], indices[2]]
        adj_sparse = torch.sparse_coo_tensor(indices, values, adj.size())
        print(adj_sparse.shape)
        exit()
        perm = torch.randperm(1)
        x_perm = x[perm]
        x = torch.concat((x, x_perm), dim=-1)
        x_valid_perm = x_valid[perm]
        x_valid = torch.concat((x_valid, x_valid_perm), dim=-1)
        B, T, N = x.shape
        #print(x.shape, x_valid.shape, od_condition.shape, y.shape, adj.shape)
        mask = torch.ones((T, T), dtype=torch.bool, device=x.device).triu(1)
        mask = mask.unsqueeze(0)
        
        mask_padding = torch.arange((T), dtype=torch.float32, device=x.device)[:, None]
        # mask: (T, N) -> (1, T, N) -> (B, T, N)
        #print(torch.repeat_interleave(mask_padding, N, axis=-1))
        mask_padding = torch.repeat_interleave(mask_padding, N, axis=-1)[None, :, :] > x_valid[:, None, :]
        #print(mask_padding[0][x_valid[0][0]+1])
        #exit()
        mask_padding = mask_padding.transpose(1,2).reshape(B*N, -1, T)
        #print(mask_padding.expand(-1, 29, -1))
        #print(mask.shape)
        #print(mask.expand(32, -1, -1))
        mask = mask | mask_padding
        print(mask.shape)    
        #exit()
    