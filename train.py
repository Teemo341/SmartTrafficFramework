import numpy as np
import time
import torch
import os
import sys
import pickle
import argparse
import random
from task1.test1 import train as train1
from task2.process_task2 import train as train2
from task3.train_mae import train as train3
from task4.train import train as train4
from torch.utils.data import SequentialSampler
from utils import read_city, get_task4_data,adj_m2adj_l,generate_node_type
from tqdm import tqdm
from device_selection import get_local_device
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' 
sys.path.append('..')
sys.path.append('../data')

from task4.DQN import DQNAgent
from dataloader import SmartTrafficDataset, SmartTrafficDataloader, read_node_type
from process_data import read_traj



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--model_read_path', type=str, default=None)
    parser.add_argument('--model_save_path', type=str, default=None)

    parser.add_argument('--memory_len', type=int, default=2000)
    parser.add_argument('--n_embd', type=int, default=64)
    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--n_layer', type=int, default=6)
    parser.add_argument('--mask_ratio', type=float, default=0.0)
    parser.add_argument('--wait_quantization', type=int, default=15)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--device', type=str, default=get_local_device(0))
    parser.add_argument('--memory_device', type=str, default=get_local_device(0))
    parser.add_argument('--max_len', type=int, default=122)
    parser.add_argument('--vocab_size', type=int, default=181)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--n_hidden', type=int, default=64)
    parser.add_argument('--window_size', type=int, default=1)
    parser.add_argument('--adjcent', type=str, default='data/jinan/adjcent.npy')
    parser.add_argument('--time_step_path', type=str, default='data/jinan/jinan_time_step/')
    parser.add_argument('--traj_num', type=int, default=100000)
    parser.add_argument('--weight_quantization_scale', type=int, default=20, help='task3')
    parser.add_argument('--observe_ratio', type=float, default=0.5, help='task3')
    parser.add_argument('--use_adj_table', type=float, default=True, help='task3')
    parser.add_argument('--special_mask_value', type=float, default=0.0001, help='task3')
    parser.add_argument('--city', type=str, default='jinan', help='task4')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--log_dir', type=str, default='./task4/log')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--task_type', type=int, default=0)
    parser.add_argument('--trajs_path', type=str, default='data/jinan/traj_repeat_one_by_one/')
    parser.add_argument('--T', type=int, default=100)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)




    # dataloader
    if args.task_type == 0:

        cfg = vars(args)
        trajs_edge = None
        dataset = SmartTrafficDataset(trajs_edge,mode="task1",trajs_path=cfg['trajs_path'],T=cfg['T'],max_len=cfg['max_len'])
        data_loader = SmartTrafficDataloader(dataset,batch_size=args.batch_size,max_len=cfg['max_len'],vocab_size=cfg['vocab_size'],shuffle=False, num_workers=4)
        train_dataloader = data_loader.get_train_data()
        cfg['block_size'] = dataset.T
        train1(cfg, data_loader)

    elif args.task_type == 1:
        cfg = vars(args)
        trajs_node_notrepeat = None
        dataset = SmartTrafficDataset(trajs_node_notrepeat,mode="task2",
                                      trajs_path=cfg['trajs_path'],
                                      adjcent_path=cfg['adjcent'],
                                      vocab_size=args.vocab_size,T=args.T,max_len=args.max_len)
        
        data_loader = SmartTrafficDataloader(dataset,batch_size=args.batch_size,shuffle=False,x=cfg['traj_num'],y=1000000,num_workers=4)

        cfg['block_size'] = cfg['T']
        train2(cfg, data_loader)

    elif args.task_type == 2:
        cfg = vars(args)
        trajs_node_repeat = None
        dataset = SmartTrafficDataset(trajs_node_repeat,mode="task3",
                                      trajs_path=cfg['trajs_path'],
                                      time_step_path=cfg['time_step_path'],
                                      adjcent_path=cfg['adjcent'],
                                      weight_quantization_scale=cfg['weight_quantization_scale'],
                                      T=cfg['T'],max_len=cfg['max_len'])
        
        data_loader = SmartTrafficDataloader(dataset,batch_size=args.batch_size,shuffle=False,x=cfg['traj_num'],y=1000000,num_workers=4)
        train_dataloader = data_loader.get_train_data()
        print(len(train_dataloader))
        cfg['block_size'] = cfg['T']
        train3(cfg, train_dataloader)

    elif args.task_type == 3:
        cfg = vars(args)
        dataset4 = SmartTrafficDataset(None,trajs_path=cfg['trajs_path'],mode="task4")
        data_loader4 = SmartTrafficDataloader(dataset4,batch_size=args.batch_size,shuffle=True, num_workers=4)
        agent = DQNAgent(args.device, args.memory_device, args.memory_len, args.n_layer, args.n_embd, args.n_head, args.wait_quantization, args.dropout,lr=args.learning_rate)
        train4(agent, vars(args), data_loader4, epochs = args.epochs, log_dir = args.log_dir)
    else:
        raise ValueError('task_type should be 0, 1, 2, 3')
  
    print('Training finished!')