import numpy as np
import torch
import os
import sys
import pickle
import random
import matplotlib.pyplot as plt
import cv2
from train import train4

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
sys.path.append('..')
sys.path.append('../data')

from task4.DQN import DQNAgent
from dataloader import SmartTrafficDataset, SmartTrafficDataloader, read_node_type
from process_data import read_traj

method = 1
seed = 0
load_dir = './task4/log/best_model.pth'
save_frame_path = './task4/video/frames'
save_video_path = "./task4/video"
batch_size = 1
device = 'cuda:2'
memory_device = 'cuda:2'
memory_len = 2000
n_layer = 6
n_embd = 64
n_head = 4
wait_quantization = 15
mask_ratio = 0.0

cfg = {
    'seed': seed,
    'memory_len': memory_len,
    'n_embd': n_embd,
    'n_head': n_head,
    'n_layer': n_layer,
    'mask_ratio': mask_ratio,
    'wait_quantization': wait_quantization,
    'device': device,
    'memory_device': memory_device,
    'batch_size': batch_size,
    'log_dir': './task4/log',
    'dropout': 0.1,

}

def train_presention(device='cuda:2',epochs=4):

    trajs_edge = read_traj('data/simulation/trajectories_10*10_repeat.csv')
    dataset4 = SmartTrafficDataset(trajs_edge,mode="task4")
    data_loader4 = SmartTrafficDataloader(dataset4,batch_size=1,shuffle=True, num_workers=4)
    agent = DQNAgent(device,memory_device, memory_len, 1, n_layer, n_embd, n_head,wait_quantization, cfg['dropout'])
    train4(agent, cfg, data_loader4, epochs = epochs, log_dir = cfg['log_dir'])


def draw_frame(wait,light):
    #wait: (5 , 7) 5 cross, 7 choices
    #light: (5)
    plt.figure(figsize=(10,10))
    plt.xlim(0, 6)
    plt.ylim(0, 6)

    wait = wait / (np.max(wait)+1e-6)
    print(wait.max(), wait.min())
    wait_colors = plt.cm.RdYlBu(1 - wait)

    def draw_cross(x,y, wait_,light_):
        width1 = 0.05
        width3 = 3*width1
        width5 = 5*width1
        line_width = 9
        alpha = 0.6
        # plt.plot(x,y,'o',color = 'grey',markersize = 20)

        plt.plot([x+width3,x+width3],[y-width5,y-1+width1],color = wait_[0],linewidth = line_width, alpha = alpha)
        plt.plot([x+width3,x+width3],[y+width5,y+1-width1],color = 'grey',linewidth = line_width, alpha = alpha)
        plt.plot([x-width3,x-width3],[y+width5,y+1-width1],color = wait_[0],linewidth = line_width, alpha = alpha)
        plt.plot([x-width3,x-width3],[y-width5,y-1+width1],color = 'grey',linewidth = line_width, alpha = alpha)

        plt.plot([x+width1,x+width1],[y-width5,y-1+width1],color = wait_[1],linewidth = line_width, alpha = alpha)
        plt.plot([x+width1,x+width1],[y+width5,y+1-width1],color = 'grey',linewidth = line_width, alpha = alpha)
        plt.plot([x-width1,x-width1],[y+width5,y+1-width1],color = wait_[1],linewidth = line_width, alpha = alpha)
        plt.plot([x-width1,x-width1],[y-width5,y-1+width1],color = 'grey',linewidth = line_width, alpha = alpha)

        plt.plot([x-1+width1,x-width5],[y-width3,y-width3],color = wait_[2],linewidth = line_width, alpha = alpha)
        plt.plot([x+1-width1,x+width5],[y-width3,y-width3],color = 'grey',linewidth = line_width, alpha = alpha)
        plt.plot([x+1-width1,x+width5],[y+width3,y+width3],color = wait_[2],linewidth = line_width, alpha = alpha)
        plt.plot([x-1+width1,x-width5],[y+width3,y+width3],color = 'grey',linewidth = line_width, alpha = alpha)

        plt.plot([x-1+width1,x-width5],[y-width1,y-width1],color = wait_[3],linewidth = line_width, alpha = alpha)
        plt.plot([x+1-width1,x+width5],[y-width1,y-width1],color = 'grey',linewidth = line_width, alpha = alpha)
        plt.plot([x+1-width1,x+width5],[y+width1,y+width1],color = wait_[3],linewidth = line_width, alpha = alpha)
        plt.plot([x-1+width1,x-width5],[y+width1,y+width1],color = 'grey',linewidth = line_width, alpha = alpha)

        if light_ == 0:
            plt.plot([x+width3,x+width3],[y+0.5-width1,y-0.5+width1],color = 'black',linewidth = line_width/4)
            plt.plot([x-width3,x-width3],[y+0.5-width1,y-0.5+width1],color = 'black',linewidth = line_width/4)
        elif light_ == 1:
            plt.plot([x+width1,x+width1,x-0.5+width1],[y-0.5+width1,y+width1,y+width1],color = 'black',linewidth = line_width/4)
            plt.plot([x-width1,x-width1,x+0.5-width1],[y+0.5-width1,y-width1,y-width1],color = 'black',linewidth = line_width/4)
        elif light_ == 2:
            plt.plot([x-0.5+width1,x+0.5-width1],[y+width3,y+width3],color = 'black',linewidth = line_width/4)
            plt.plot([x-0.5+width1,x+0.5-width1],[y-width3,y-width3],color = 'black',linewidth = line_width/4)
        elif light_ == 3:
            plt.plot([x+width1,x+width1,x-0.5+width1],[y+0.5-width1,y-width1,y-width1],color = 'black',linewidth = line_width/4)
            plt.plot([x-width1,x-width1,x+0.5-width1],[y-0.5+width1,y+width1,y+width1],color = 'black',linewidth = line_width/4)
        else:
            raise ValueError(f'light error, should be crossroad light, but detected Y light {light}')
        

    draw_cross(3,3,wait_colors[0],light[0])
    draw_cross(3,1,wait_colors[1],light[1])
    draw_cross(1,3,wait_colors[2],light[2])
    draw_cross(3,5,wait_colors[3],light[3])
    draw_cross(5,3,wait_colors[4],light[4])

    return plt


def draw_video(wait,light, save_frame_path = './task4/video/frames', save_video_path = "./task4/video"):
    # wait: (T, 5, 7) 5 cross, 7 choices
    # light: (T, 5)
    print(wait.shape, light.shape)
    if save_frame_path is not None:
        os.makedirs(save_frame_path,exist_ok=True)
    if save_video_path is not None:
        os.makedirs(save_video_path,exist_ok=True)

    T = wait.shape[0]
    for t in range(T):
        plt = draw_frame(wait[t],light[t])
        if save_video_path is not None:
            plt.savefig(f'{save_frame_path}/frame_{t}.png', bbox_inches='tight')
        plt.close()
    
    if save_video_path is not None:
        # Create a VideoCapture object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        frame = cv2.imread(f'{save_frame_path}/frame_{0}.png')
        frame_height, frame_width, _ = frame.shape
        out = cv2.VideoWriter(f'{save_video_path}/video.mp4', fourcc, 2, (frame_width, frame_height))

        # Iterate over all the frames
        for i in range(len(os.listdir(save_frame_path))):
            frame = cv2.imread(f'{save_frame_path}/frame_{i}.png')
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)

        # Release the VideoCapture object
        out.release()

#TODO implement the function
def filter():
    # find the 5 crossroads, center, lower, left, upper, right
    # up to you, maybe you can manually set the 5 crossroads for each dataset

    # ids = [34,33,24,35,44]
    ids = [33,32,23,34,43]
    return ids

def test_presention(method=1):
    method = method
    iteration = 1
    seed = 0
    load_dir = './task4/log/best_model.pth'
    save_frame_path = './task4/video/frames'
    save_video_path = "./task4/video"
    batch_size = 64
    device = 'cuda:2'
    memory_device = 'cuda:2'
    memory_len = 2000
    n_layer = 6
    n_embd = 64
    n_head = 4
    wait_quantization = 15
    mask_ratio = 0.0


    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # dataloader
    trajs_edge = read_traj('data/simulation/trajectories_10*10_repeat.csv')
    trajs_edge = np.load('data/simulation/task4.npy')
    dataset4 = SmartTrafficDataset(trajs_edge,mode="task4")
    data_loader4 = SmartTrafficDataloader(dataset4,batch_size=batch_size,shuffle=True, num_workers=4)

    agent = DQNAgent(device, memory_device, memory_len, n_layer, n_embd, n_head, wait_quantization, 0)
    # agent.model.load_state_dict(torch.load(load_dir))
    agent.model = agent.model.to(device)

    ids = filter()

    cross_type = read_node_type('data/simulation/node_type_10*10.csv') # 1,...,100 V
    cross_type = [3 if i == 'T' else 4 for i in cross_type]
    cross_type = torch.tensor(cross_type, dtype=torch.int, device = device) # (V,)
    # print(cross_type[ids])

    if mask_ratio:
        mask_id = np.random.choice(len(cross_type), int(len(cross_type)*mask_ratio), replace=False)+1
        mask = torch.ones(len(cross_type), dtype=torch.int) # (V,)
        mask[mask_id] = False
    else:
        mask = torch.ones(len(cross_type), dtype=torch.int) # (V,)
    mask = mask.to(device)

    with torch.no_grad():
        for i, wait in enumerate(data_loader4):
            if i != iteration:
                continue
            wait = wait[:,:,1:,:] # remove the special token, (B, T, V, 7)
            wait = wait.to(device)
            wait = torch.clamp(wait, -1, wait_quantization) # (B, T, V, 7), all negative values become special token, clamp the wait value max to wait_quantization
            full_wait = wait.int().clone()
            B, T, V, _ = wait.shape
            wait = wait*mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1) - torch.ones_like(wait,dtype=int)*(1-mask).unsqueeze(0).unsqueeze(0).unsqueeze(-1) # change the value of the masked position to -1
            wait= wait.int()

            light = 0
            light_list = []

            for t in range(T):
                if t == 0:
                    light = agent.best_light(full_wait[:,t,:,:]) # (B, V, 7)
                    print(full_wait[0,t,:,:])
                    # print(light[0,:,:])
                    # print(full_wait[0,t,ids,:])
                    # print(light[0,ids,:])
                    light = torch.argmax(light, dim = -1) # (B, V)
                    # print(light[0,ids])
                    light_list.append(light[0,ids]) # (5)
                else:
                    if method == 0:
                        action = torch.ones((B, V), dtype=torch.int, device=device)
                    elif method == 1:
                        state = (wait[:,t,:,:], cross_type, light)
                        action = agent.act(state[0],state[1],state[2],agent.epsilon) # (B, V)
                    else: 
                        raise ValueError(f'{method} method not supported')
                    # next light
                    light = agent.turn_light(cross_type, light, action) # (B, V)
                    light_list.append(light[0,ids]) # (5)

            light_list = torch.stack(light_list, dim = 0) # (T, 5)
            wait_list = wait[0, :, ids, :] # (T, 5, 7)
            light_list = np.array(light_list.detach().cpu())
            wait_list = np.array(wait_list.detach().cpu())
            draw_video(wait_list, light_list, save_frame_path, save_video_path)
            break

    print('Finish') 



if __name__ == '__main__':

    test_presention(method=1)