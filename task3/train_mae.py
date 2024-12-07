import pickle
import os
import argparse
from numpy.core.fromnumeric import size
from numpy.core.numeric import indices
import torch
import numpy as np
import time
import random
from tqdm import tqdm
from task3.model_mae import no_diffusion_model_cross_attention_parallel as no_diffusion_model

def define_model(cfg):

    vocab_size = cfg['vocab_size']
    n_embd = cfg['n_embd']
    n_head = cfg['n_head']
    n_layer = cfg['n_layer']
    dropout = cfg['dropout']
    device = cfg['device']
    block_size = cfg['block_size']
    weight_quantization_scale = cfg['weight_quantization_scale']
    use_adj_table = cfg['use_adj_table']

    model= no_diffusion_model(vocab_size,
                               n_embd, 
                               n_embd,
                               n_layer, 
                               n_head,
                               block_size, 
                               dropout,
                               weight_quantization_scale = weight_quantization_scale,
                               use_adj_table=use_adj_table, 
                               use_ne=True, 
                               use_ge=True,
                               use_agent_mask=False, 
                               norm_position='prenorm')

    model = model.to(device)
    model.load_state_dict(torch.load(cfg['model_read_path']))
    return model

def train(cfg,dataloader):


    vocab_size = cfg['vocab_size']
    n_embd = cfg['n_embd']
    n_head = cfg['n_head']
    n_layer = cfg['n_layer']
    dropout = cfg['dropout']
    device = cfg['device']
    block_size = cfg['block_size']
    weight_quantization_scale = cfg['weight_quantization_scale']
    use_adj_table = cfg['use_adj_table']
    learning_rate = cfg['learning_rate']
    max_epochs = cfg['epochs']
    load_dir_id = None
    observe_ratio = cfg['observe_ratio']
    special_mask_value = 0.0001
        # make loggers
    logger_train_loss = []
    logger_train_acc = []
    logger_train_acc_inner = []

    model= no_diffusion_model(vocab_size,
                               n_embd, 
                               n_embd,
                               n_layer, 
                               n_head,
                               block_size, 
                               dropout,
                               weight_quantization_scale = weight_quantization_scale,
                               use_adj_table=use_adj_table, 
                               use_ne=True, 
                               use_ge=True,
                               use_agent_mask=False, 
                               norm_position='prenorm')
    print(vocab_size,n_embd,n_embd,n_layer, n_head,block_size, dropout,weight_quantization_scale,use_adj_table, True, True,False, 'prenorm')

    model = model.to(device)
    old_path = None
    if cfg['model_read_path']:
        model.load_state_dict(torch.load(cfg['model_read_path']))
        if 'best_model' in cfg['model_read_path'] or 'last_model' in cfg['model_read_path']:
            last_loss = float(cfg['model_read_path'][-10:-4])
            old_path = cfg['model_read_path']
    else:
        last_loss = 10000
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_epochs, eta_min=0, last_epoch=-1 if not load_dir_id else load_dir_id)
   
    dataloader.randomize_condition(observe_ratio)
    start = 0
    for i in range (start+1, max_epochs+1):
        model.train()
        # dataloader.randomize_condition(observe_ratio)

        epoch_time = time.time()
        load_data_time = 0
        preprocess_data_time = 0
        forward_time = 0
        backward_time = 0
        a = time.time()
        for condition, time_step, special_mask, adj_table in tqdm(dataloader, desc=f'Train epoch {i:>6}/{max_epochs:<6}'):
     
            loss1 = []
            load_data_time += time.time()-epoch_time
            epoch_time = time.time()
            # return trajectory: [B x N x T], time_step: [B x N], special_mask: [B x N x T], adj_table: [B x N x V x 4 x 2]

            # random choice a traj as input, the rest as condition
            shuffled_indices = torch.randperm(condition.size(1))
            condition = condition[:,shuffled_indices,:]
            #time_step = time_step[:,shuffled_indices]
            special_mask = special_mask[:,shuffled_indices,:]

            # get y, filter trajecotry into condition and get x
            condition = condition.to(device)
            y = condition[:,0,:] # [B x T]
            y = y.long()
            # todo try another filter method
            condition_ = dataloader.filter_condition(condition) # remove unboservable nodes

            x = condition_[:,0,:] # [B x T]
            condition = condition[:,1:,:] # [B x N-1 x T]
            # condition = None

            if use_adj_table:
                if isinstance(adj_table, torch.FloatTensor):
                    #print(shuffled_indices[0])
                    adj_table = adj_table[:,shuffled_indices[0],:,:,:].to(device) # [B x V x 4 x 2]
                    adj_table = [adj_table[...,0],adj_table[...,1]] # [B x V x 4], [B x V x 4]
                elif isinstance(adj_table, torch.sparse.FloatTensor):
                    adj_table = adj_table.to_dense()[:,shuffled_indices[0],:,:,:] # [B x V x 4 x 2]
                    adj_table = [adj_table[...,0].to(device),adj_table[...,1].to(device)] # [B x V x 4], [B x V x 4]
                else:
                    raise ValueError('No adj matrix type should be torch.FloatTensor or torch.sparse.FloatTensor')
            else:
                raise ValueError('No adj matrix in current version, please use adj table')
            
            #time_step = time_step.to(device)
            special_mask = special_mask[:,0,:].to(device)
            special_mask_ = (special_mask+special_mask_value).clamp(0,1).float()

            preprocess_data_time += time.time()-epoch_time
            epoch_time = time.time()
       
            logits, loss = model(x, condition, adj_table, y, None , None, special_mask_)
    
            loss = torch.mean(loss)
            loss1.append(loss)

            forward_time += time.time()-epoch_time
            epoch_time = time.time()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.zero_grad()
            loss.backward()

            logger_train_loss.append(loss.item())
            logger_train_acc.append((torch.argmax(logits, dim=-1) == y).float().mean().item())
            logger_train_acc_inner.append((((torch.argmax(logits, dim=-1) == y).float()*special_mask).sum()/special_mask.sum()).item())
            optimizer.step()

            avg_loss = sum(loss1)/len(loss1)
            
            backward_time += time.time()-epoch_time
            epoch_time = time.time()
       
        if os.path.isdir(cfg['model_save_path']):
            path = os.path.join(cfg['model_save_path'],f"best_model_{avg_loss:.4f}.pth")
            if avg_loss < last_loss:
                if old_path:
                    os.remove(old_path)
                last_loss = avg_loss
                torch.save(model.state_dict(), path)
                old_path = path

        lr_sched.step()
        print(f'Train epoch {i:>6}/{max_epochs:<6}|  Loss: {loss.item():<10.8f}  |  Acc: {logger_train_acc[-1]:<7.2%}  |  Acc_inner: {logger_train_acc_inner[-1]:<7.2%}  |  LR: {lr_sched.get_last_lr()[0]:<10.8f}  | Load data time: {load_data_time/60:.<7.2f}m  |  Preprocess data time: {preprocess_data_time/60:<7.2f}m  |  Forward time: {forward_time/60:<7.2f}m  |  Backward time: {backward_time/60:<7.2f}m  |  Total time: {(load_data_time + preprocess_data_time + forward_time + backward_time)/60:<7.2f}m')
        epoch_time = time.time()

    if os.path.isdir(cfg['model_save_path']):
        path = os.path.join(cfg['model_save_path'],f"last_model_{avg_loss:.4f}.pth")
        torch.save(model.state_dict(), path)

    # if cfg['model_save_path']:
    #     torch.save(model.state_dict(), cfg['model_save_path'])
        # if i % eval_epochs == 0:
        #     if dataloader.test_loader is None:
        #         print('No test data, skip evaluation')
        #         continue
        #     model.eval()
        #     test_loss = []
        #     test_acc = []
        #     test_acc_inner = []
        #     real_true_rate = []
        #     with torch.no_grad():
        #         for(j, (condition, time_step, special_mask, adj_table)) in enumerate(dataloader.test_loader):
        #             # return trajectory: [B x N x T], special_mask: [B x N x T], adj_table: [B x N x V x 4 x 2]

        #             # random choice a traj as input, the rest as condition
        #             shuffled_indices = torch.randperm(condition.size(1))
        #             condition = condition[:,shuffled_indices,:]
        #             time_step = time_step[:,shuffled_indices]
        #             special_mask = special_mask[:,shuffled_indices,:]
        #             adj_table = adj_table[:,shuffled_indices,:,:,:]

        #             # get y, filter trajecotry into condition and get x
        #             condition = condition.to(device)
        #             y = condition[:,0,:] # [B x T]
        #             y = y.long()
        #             # todo try another filter method
        #             condition_ = dataloader.filter_condition(condition) # remove unboservable nodes
        #             x = condition_[:,0,:] # [B x T]
        #             condition = condition[:,1:,:] # [B x N-1 x T]
        #             # condition = None

        #             if use_adj_table:
        #                 if isinstance(adj_table, torch.FloatTensor):
        #                     adj_table = adj_table[:,shuffled_indices[0],:,:,:].to(device) # [B x V x 4 x 2]
        #                     adj_table = [adj_table[...,0],adj_table[...,1]] # [B x V x 4], [B x V x 4]
        #                 elif isinstance(adj_table, torch.sparse.FloatTensor):
        #                     adj_table = adj_table.to_dense()[:,shuffled_indices[0],:,:,:].to(device) # [B x V x 4 x 2]
        #                     adj_table = [adj_table[...,0].to_sparse(),adj_table[...,1].to_sparse()] # [B x V x 4], [B x V x 4]
        #                 else:
        #                     raise ValueError('No adj matrix type should be torch.FloatTensor or torch.sparse.FloatTensor')
        #             else:
        #                 raise ValueError('No adj matrix in current version, please use adj table')
                    
        #             time_step = time_step.to(device)
        #             special_mask = special_mask[:,0,:].to(device)
        #             special_mask_ = (special_mask+special_mask_value).clamp(0,1).float()

        #             logits, loss = model(x, condition, adj_table, y, time_step, None, special_mask_)
        #             loss = torch.mean(loss)

        #             acc = (torch.argmax(logits, dim=-1) == y).float().mean()
        #             acc_inner = ((torch.argmax(logits, dim=-1) == y).float()*special_mask).sum()/special_mask.sum()
        #             real_true = 0
        #             for j in range(y.shape[0]):
        #                 if (torch.argmax(logits[j], dim = -1) == y[j]).all():
        #                     real_true += 1/y.shape[0]
                    
        #             test_loss.append(loss.item())
        #             test_acc.append(acc.item())
        #             test_acc_inner.append(acc_inner.item())
        #             real_true_rate.append(real_true)
            
        #     loss = np.mean(test_loss)
        #     acc = np.mean(test_acc)
        #     acc_inner = np.mean(test_acc_inner)
        #     real_true_rate = np.mean(real_true_rate)

        #     print(f'Test epoch {i//eval_epochs:>6}/{max_epochs//eval_epochs:<6}|  Loss: {loss:<10.8f}  |  Acc: {acc:<7.2%}  |  Acc_inner: {acc_inner:<7.2%}  |  Real True Rate: {real_true_rate:<7.2%}  |  Time: {(time.time()-epoch_time)/60:<7.2f}m\n')
   



if __name__ == '__main__':

    # ------------------------------------------------------------------------------------------------------------------------------------------
    # Hyperparameters

    parser = argparse.ArgumentParser()

    # about dataset
    parser.add_argument('--city', type=str, default='boston')
    parser.add_argument('--simulation_num', type=int, default=500000)
    parser.add_argument('--test_simulation_num', type=int, default=10000)
    parser.add_argument('--use_given_data', type=bool, default=True)
    parser.add_argument('--condition_num', type=int, default=5)
    parser.add_argument('--capacity_scale', type=int, default=10)
    parser.add_argument('--weight_quantization_scale', type=int, default=None)
    parser.add_argument('--max_connection', type=int, default=4)
    parser.add_argument('--train_data_dir', type=str, default='./data')
    parser.add_argument('--test_data_dir', type=str, default='./data_test')

    # about dataloader
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--block_size', type=int, default=50)
    parser.add_argument('--observe_ratio', type=float, default=0.5)
    parser.add_argument('--special_mask_value', type=float, default=0.0001)

    # about model
    parser.add_argument('--n_embd', type=int, default=64)
    parser.add_argument('--n_head', type=int, default=16)
    parser.add_argument('--n_layer', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--use_adj_table', type=bool, default=True)

    # about training
    parser.add_argument('--max_epochs', type=int, default=1000)
    parser.add_argument('--learning_rate', type=float, default=1e-1)
    parser.add_argument('--lr_drop_rate', type=float, default=0.5)
    parser.add_argument('--eval_freq', type=int, default=100)
    parser.add_argument('--save_freq', type=int, default=5)
    parser.add_argument('--device_ids', type=int, nargs= '+', default=[0])
    parser.add_argument('--seed', type=int, default=0)

    # about record and resume training
    parser.add_argument('--load_dir', type=str, default='./checkpoints_mae/')
    parser.add_argument('--load_dir_id', type=int, default=None)

    args = parser.parse_args()
    if not os.path.exists(args.load_dir):
        os.makedirs(args.load_dir)
    with open(f'{args.load_dir}args.pkl', 'wb') as f:
        pickle.dump(args, f)
    for arg in vars(args):
        print(f"{arg:<30}: {getattr(args, arg)}")


    # about dataset
    city = args.city
    simulation_num = args.simulation_num
    test_simulation_num = args.test_simulation_num
    use_given_data = args.use_given_data
    condition_num = args.condition_num # on each weighted graph, give how many condition?, same as total_trajectories in dataset
    if simulation_num == 0:
        use_given_data = False
    capacity_scale = args.capacity_scale
    weight_quantization_scale = args.weight_quantization_scale
    max_connection = args.max_connection
    data_dir = args.train_data_dir
    test_data_dir = args.test_data_dir

    # about dataloader
    batch_size = args.batch_size # how many independent djkastra graph will we process in parallel?
    if torch.cuda.is_available() and len(args.device_ids) > 1:
        batch_size *= len(args.device_ids)
    block_size = args.block_size # The max length of all shortest path
    special_mask_value = args.special_mask_value # The value of special mask
    observe_ratio = args.observe_ratio # The ratio of the observed trajectory

    # about model
    n_embd = args.n_embd
    n_head = args.n_head
    n_layer = args.n_layer
    dropout = args.dropout

    # about training
    max_epochs = args.max_epochs
    learning_rate = args.learning_rate
    lr_drop_rate = args.lr_drop_rate
    eval_epochs = int(max_epochs/args.eval_freq)
    # eval_iters = 1
    save_epochs = max_epochs//args.save_freq
    max_epochs = save_epochs*args.save_freq

    if torch.cuda.is_available():
        device_ids = args.device_ids
        device = f'cuda:{args.device_ids[0]}'
    else:
        device = 'cpu'

    use_adj_table = args.use_adj_table

    # about record and resume training
    load_dir_id = args.load_dir_id
    load_dir = args.load_dir
    # load_dir_id = 10000

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


    # ------------------------------------------------------------------------------------------------------------------------------------------
    # Load data
    start_time = time.time()

    dataloader = traj_dataloader(city, data_dir, test_data_dir, simulation_num, test_simulation_num, condition_num, block_size, capacity_scale, weight_quantization_scale, max_connection, batch_size, shuffle=True, num_workers=8)

    vocab_size = dataloader.vocab_size
    print(f'{city} has {vocab_size -1} nodes, add 0 for sepcial token, now vocab size is {vocab_size}')

    print('Data loaded in', time.time()-start_time, 'seconds')


    # ------------------------------------------------------------------------------------------------------------------------------------------
    # Load model
    start_time = time.time()

    model= no_diffusion_model(vocab_size, n_embd, n_embd, n_layer, n_head, block_size, dropout, weight_quantization_scale = weight_quantization_scale, use_adj_table=use_adj_table, use_ne=True, use_ge=True, use_agent_mask=False, norm_position='prenorm')

    if load_dir_id:
        model = torch.load(f'{load_dir}complete_model_{load_dir_id}.pth')
        print('Model loaded from', f'{load_dir}complete_model_{load_dir_id}.pth')

    model = model.to(device)
    if torch.cuda.is_available() and len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    print('Model loaded in', time.time()-start_time, 'seconds')


    # ------------------------------------------------------------------------------------------------------------------------------------------
    # set optimizer and lr scheduler

    # learning_rate *= lr_drop_rate**int(load_dir_id//save_epochs)

    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=1e-15)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, eps=1e-15)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)

    # lr_sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=save_iters, gamma=lr_drop_rate)

    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_epochs, eta_min=0, last_epoch=-1 if not load_dir_id else load_dir_id)


    # ------------------------------------------------------------------------------------------------------------------------------------------
    # train

    # make loggers
    logger_train_loss = []
    logger_train_acc = []
    logger_train_acc_inner = []
    logger_test_loss = []
    logger_test_acc = []
    logger_test_acc_inner = []
    logger_test_real_true_rate = []
    best_test_acc = 0

    if load_dir_id:
        with open(f'{load_dir}logger_train_loss.pkl', 'rb') as f:
            logger_train_loss = pickle.load(f)
        with open(f'{load_dir}logger_train_acc.pkl', 'rb') as f:
            logger_train_acc = pickle.load(f)
        with open(f'{load_dir}logger_train_acc_inner.pkl', 'rb') as f:
            logger_train_acc_inner = pickle.load(f)
        with open(f'{load_dir}logger_test_loss.pkl', 'rb') as f:
            logger_test_loss = pickle.load(f)
        with open(f'{load_dir}logger_test_acc.pkl', 'rb') as f:
            logger_test_acc = pickle.load(f)
        with open(f'{load_dir}logger_test_acc_inner.pkl', 'rb') as f:
            logger_test_acc_inner = pickle.load(f)
        with open(f'{load_dir}logger_test_real_true_rate.pkl', 'rb') as f:
            logger_test_real_true_rate = pickle.load(f)


    # training epochs
    start_time = time.time()
    start = 0
    if load_dir_id:
        start = load_dir_id
    print('start training from iteration', start)

    dataloader.randomize_condition(observe_ratio)

    for i in range (start+1, max_epochs+1):
        model.train()
        # dataloader.randomize_condition(observe_ratio)

        epoch_time = time.time()
        load_data_time = 0
        preprocess_data_time = 0
        forward_time = 0
        backward_time = 0
        for(j, (condition, time_step, special_mask, adj_table)) in tqdm(enumerate(dataloader.train_loader)):
            load_data_time += time.time()-epoch_time
            epoch_time = time.time()
            # return trajectory: [B x N x T], time_step: [B x N], special_mask: [B x N x T], adj_table: [B x N x V x 4 x 2]

            # random choice a traj as input, the rest as condition
            shuffled_indices = torch.randperm(condition.size(1))
            condition = condition[:,shuffled_indices,:]
            time_step = time_step[:,shuffled_indices]
            special_mask = special_mask[:,shuffled_indices,:]

            # get y, filter trajecotry into condition and get x
            condition = condition.to(device)
            y = condition[:,0,:] # [B x T]
            y = y.long()
            # todo try another filter method
            condition_ = dataloader.filter_condition(condition) # remove unboservable nodes

            x = condition_[:,0,:] # [B x T]
            condition = condition[:,1:,:] # [B x N-1 x T]
            # condition = None

            if use_adj_table:
                if isinstance(adj_table, torch.FloatTensor):
                    adj_table = adj_table[:,shuffled_indices[0],:,:,:].to(device) # [B x V x 4 x 2]
                    adj_table = [adj_table[...,0],adj_table[...,1]] # [B x V x 4], [B x V x 4]
                elif isinstance(adj_table, torch.sparse.FloatTensor):
                    adj_table = adj_table.to_dense()[:,shuffled_indices[0],:,:,:] # [B x V x 4 x 2]
                    adj_table = [adj_table[...,0].to(device),adj_table[...,1].to(device)] # [B x V x 4], [B x V x 4]
                else:
                    raise ValueError('No adj matrix type should be torch.FloatTensor or torch.sparse.FloatTensor')
            else:
                raise ValueError('No adj matrix in current version, please use adj table')
            
            time_step = time_step.to(device)
            special_mask = special_mask[:,0,:].to(device)
            special_mask_ = (special_mask+special_mask_value).clamp(0,1).float()

            preprocess_data_time += time.time()-epoch_time
            epoch_time = time.time()

            logits, loss = model(x, condition, adj_table, y, time_step, None, special_mask_)
            loss = torch.mean(loss)

            forward_time += time.time()-epoch_time
            epoch_time = time.time()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.zero_grad()
            loss.backward()

            logger_train_loss.append(loss.item())
            logger_train_acc.append((torch.argmax(logits, dim=-1) == y).float().mean().item())
            logger_train_acc_inner.append((((torch.argmax(logits, dim=-1) == y).float()*special_mask).sum()/special_mask.sum()).item())
            optimizer.step()

            backward_time += time.time()-epoch_time
            epoch_time = time.time()
        
        lr_sched.step()
        print(f'Train epoch {i:>6}/{max_epochs:<6}|  Loss: {loss.item():<10.8f}  |  Acc: {logger_train_acc[-1]:<7.2%}  |  Acc_inner: {logger_train_acc_inner[-1]:<7.2%}  |  LR: {lr_sched.get_last_lr()[0]:<10.8f}  | Load data time: {load_data_time/60:.<7.2f}m  |  Preprocess data time: {preprocess_data_time/60:<7.2f}m  |  Forward time: {forward_time/60:<7.2f}m  |  Backward time: {backward_time/60:<7.2f}m  |  Total time: {(load_data_time + preprocess_data_time + forward_time + backward_time)/60:<7.2f}m')
        epoch_time = time.time()

        if i % eval_epochs == 0:
            if dataloader.test_loader is None:
                print('No test data, skip evaluation')
                continue
            model.eval()
            test_loss = []
            test_acc = []
            test_acc_inner = []
            real_true_rate = []
            with torch.no_grad():
                for(j, (condition, time_step, special_mask, adj_table)) in enumerate(dataloader.test_loader):
                    # return trajectory: [B x N x T], special_mask: [B x N x T], adj_table: [B x N x V x 4 x 2]

                    # random choice a traj as input, the rest as condition
                    shuffled_indices = torch.randperm(condition.size(1))
                    condition = condition[:,shuffled_indices,:]
                    time_step = time_step[:,shuffled_indices]
                    special_mask = special_mask[:,shuffled_indices,:]
                    adj_table = adj_table[:,shuffled_indices,:,:,:]

                    # get y, filter trajecotry into condition and get x
                    condition = condition.to(device)
                    y = condition[:,0,:] # [B x T]
                    y = y.long()
                    # todo try another filter method
                    condition_ = dataloader.filter_condition(condition) # remove unboservable nodes
                    x = condition_[:,0,:] # [B x T]
                    condition = condition[:,1:,:] # [B x N-1 x T]
                    # condition = None

                    if use_adj_table:
                        if isinstance(adj_table, torch.FloatTensor):
                            adj_table = adj_table[:,shuffled_indices[0],:,:,:].to(device) # [B x V x 4 x 2]
                            adj_table = [adj_table[...,0],adj_table[...,1]] # [B x V x 4], [B x V x 4]
                        elif isinstance(adj_table, torch.sparse.FloatTensor):
                            adj_table = adj_table.to_dense()[:,shuffled_indices[0],:,:,:].to(device) # [B x V x 4 x 2]
                            adj_table = [adj_table[...,0].to_sparse(),adj_table[...,1].to_sparse()] # [B x V x 4], [B x V x 4]
                        else:
                            raise ValueError('No adj matrix type should be torch.FloatTensor or torch.sparse.FloatTensor')
                    else:
                        raise ValueError('No adj matrix in current version, please use adj table')
                    
                    time_step = time_step.to(device)
                    special_mask = special_mask[:,0,:].to(device)
                    special_mask_ = (special_mask+special_mask_value).clamp(0,1).float()

                    logits, loss = model(x, condition, adj_table, y, time_step, None, special_mask_)
                    loss = torch.mean(loss)

                    acc = (torch.argmax(logits, dim=-1) == y).float().mean()
                    acc_inner = ((torch.argmax(logits, dim=-1) == y).float()*special_mask).sum()/special_mask.sum()
                    real_true = 0
                    for j in range(y.shape[0]):
                        if (torch.argmax(logits[j], dim = -1) == y[j]).all():
                            real_true += 1/y.shape[0]
                    
                    test_loss.append(loss.item())
                    test_acc.append(acc.item())
                    test_acc_inner.append(acc_inner.item())
                    real_true_rate.append(real_true)
            
            loss = np.mean(test_loss)
            acc = np.mean(test_acc)
            acc_inner = np.mean(test_acc_inner)
            real_true_rate = np.mean(real_true_rate)
            logger_test_loss.append(loss.item())
            logger_test_acc.append(acc.item())
            logger_test_acc_inner.append(acc_inner)
            logger_test_real_true_rate.append(real_true_rate)
            print(f'Test epoch {i//eval_epochs:>6}/{max_epochs//eval_epochs:<6}|  Loss: {loss:<10.8f}  |  Acc: {acc:<7.2%}  |  Acc_inner: {acc_inner:<7.2%}  |  Real True Rate: {real_true_rate:<7.2%}  |  Time: {(time.time()-epoch_time)/60:<7.2f}m\n')
            epoch_time = time.time()

            # save logger
            with open(f'{load_dir}logger_train_loss.pkl', 'wb') as f:
                pickle.dump(logger_train_loss, f)
            with open(f'{load_dir}logger_train_acc.pkl', 'wb') as f:
                pickle.dump(logger_train_acc, f)
            with open(f'{load_dir}logger_train_acc_inner.pkl', 'wb') as f:
                pickle.dump(logger_train_acc_inner, f)
            with open(f'{load_dir}logger_test_loss.pkl', 'wb') as f:
                pickle.dump(logger_test_loss, f)
            with open(f'{load_dir}logger_test_acc.pkl', 'wb') as f:
                pickle.dump(logger_test_acc, f)
            with open(f'{load_dir}logger_test_acc_inner.pkl', 'wb') as f:
                pickle.dump(logger_test_acc_inner, f)
            with open(f'{load_dir}logger_test_real_true_rate.pkl', 'wb') as f:
                pickle.dump(logger_test_real_true_rate, f)

            
        if i % save_epochs == 0:
            epoch_time = time.time()
            # torch.save(model.state_dict(), f'./checkpoint/model_{i}.pth')
            if len(device_ids) > 1:
                torch.save(model.module, f'{load_dir}complete_model_{i}.pth')
            else:
                torch.save(model, f'{load_dir}complete_model_{i}.pth')
            text = f'Model saved {i//save_epochs:>6}/{max_epochs//eval_epochs:<6} ,at iteration {i}, time: {time.time()-epoch_time}\n'
            if logger_test_acc[-1] > best_test_acc:
                text += f'Best test acc updated: {best_test_acc} -> {logger_test_acc[-1]}\n'
                best_test_acc = logger_test_acc[-1]
                if len(device_ids) > 1:
                    torch.save(model.module, f'{load_dir}best_model.pth')
                else:
                    torch.save(model, f'{load_dir}best_model.pth')
            print(text)
            epoch_time = time.time()

    print('Training finished in', (time.time()-start_time)//3600, 'hours', ((time.time()-start_time)%3600)//60, 'minutes', ((time.time()-start_time)%3600)%60, 'seconds')