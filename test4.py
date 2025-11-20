import numpy as np
import torch
import os
import sys
import pickle
import random
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

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

    wait = wait[:,:4] # remove the triangle light
    wait = wait / (np.max(wait)+1e-6)
    wait_colors = plt.cm.RdYlBu_r(wait)
    # wait_colors = plt.cm.RdBu_r(wait)

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


def draw_video(wait,light, save_video_path = "./task4/video"):
    # wait: (T, 5, 7) 5 cross, 7 choices
    # light: (T, 5)

    if save_video_path is not None:
        os.makedirs(save_video_path,exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    T = wait.shape[0]
    for t in tqdm(range(T)):
        plt = draw_frame(wait[t],light[t])
        plt.tight_layout()
        plt.axis('off')
        plt.gca().set_position([0, 0, 1, 1])
        plt.draw()
        canvas = plt.gcf().canvas
        canvas.draw()
        width, height = canvas.get_width_height()
        frame = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8).reshape((height, width, 4))
        frame = frame[:, :, [1, 2, 3, 0]] # Convert ARGB to RGBA
        frame = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_RGB2BGR) # Drop alpha channel (optional) and convert to BGR for OpenCV
        plt.close()
        
        if t == 0:
            # Create a VideoCapture object
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            frame_height, frame_width, _ = frame.shape
            out = cv2.VideoWriter(f'{save_video_path}/video.mp4', fourcc, 2, (frame_width, frame_height))
        out.write(frame)

    # Release the VideoCapture object
    out.release()
    return f'{save_video_path}/video.mp4'

#TODO implement the function
def filter():
    # find the 5 crossroads, center, lower, left, upper, right
    # up to you, maybe you can manually set the 5 crossroads for each dataset

    # ids = [34,33,24,35,44]
    ids = [33,32,23,34,43]
    return ids

def weight_quantization(wait, wait_quantization):
    """
    Quantize the wait time to the nearest multiple of wait_quantization.
    """
    wait_max = wait.max()
    if wait_max > 0:
        wait = wait / wait_max * wait_quantization
    wait[wait < 0] = -1  # Set negative values to -1 (special token)
    wait = torch.round(wait).int()  # Round to the nearest integer
    return wait

def pass_rate(wait, light):
    """wait: (B, V, 7), light: (B, V)"""

    wait[wait < 0] = 0  # Set negative values to 0 (special token)
    no_zero_BV_mask = (wait > 0).any(dim=-1)  # (B, V)
    if no_zero_BV_mask.sum() == 0:
        return None
    light_7 = torch.nn.functional.one_hot(light, num_classes=7).float()  # (B, V, 7)
    val = wait * light_7  # (B, V, 7)
    pass_rate = val.sum(dim=-1) / (wait.sum(dim=-1)+1e-32)  # (B, V)
    pass_rate = torch.sum(pass_rate)/ (no_zero_BV_mask.sum())  # average pass rate over no zero crossings
    return pass_rate

def test_presention(num, method, save_path = './UI_element/task4'):
    iteration = 0
    seed = 0
    load_dir = './task4/log/best_model.pth'
    batch_size = 64
    device = 'cuda'
    memory_device = 'cuda'
    memory_len = 2000
    n_layer = 8
    n_embd = 128
    n_head = 4
    wait_quantization = 20
    mask_ratio = 0.0
    task_4_num = np.ceil(num/1000).astype(int) # 1000 is the number of samples in one task4 dataset
    
    # dataloader
    trajs_edge = './data_/simulation/new_task4_data_one_by_one'
    dataset4 = SmartTrafficDataset(trajs_edge,map_path='data_/simulation/edge_node_10*10.csv',mode="task4",task4_num=task_4_num)
    data_loader4 = SmartTrafficDataloader(dataset4,batch_size=batch_size,shuffle=True, num_workers=4)

    agent = DQNAgent(device, memory_device, memory_len, n_layer, n_embd, n_head, wait_quantization, 0.1)
    agent.model.load_state_dict(torch.load(load_dir))
    agent.model = agent.model.to(device)

    ids = filter()

    cross_type = read_node_type('data_/simulation/node_type_10*10.csv') # 1,...,100 V
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
            wait = weight_quantization(wait, wait_quantization) # (B, T, V, 7), all negative values become special token, round the wait value to the nearest integer
            full_wait = wait.int().clone()
            B, T, V, _ = wait.shape
            wait = wait*mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1) - torch.ones_like(wait,dtype=int)*(1-mask).unsqueeze(0).unsqueeze(0).unsqueeze(-1) # change the value of the masked position to -1
            wait= wait.int()

            light = 0
            light_list = []

            action_rate_list = []
            best_rate_list = []
            for t in range(T):
                if t == 0:
                    
                    light = agent.best_light(full_wait[:,t,:,:]) # (B, V, 7)
                    light = torch.argmax(light, dim = -1) # (B, V)                    
                    light_list.append(light[0,ids]) # (5)
                    # light_7 = torch.nn.functional.one_hot(light, num_classes=7).float()
                    # val = torch.max(full_wait[:,:,:,:],dim=3).values
                    # best_run = torch.sum(val[val>0])/torch.sum(full_wait[full_wait>0])
                    # best_run = best_run.item()
                else:
                    if method == 0:
                        action = torch.ones((B, V), dtype=torch.int, device=device)
                        light = agent.turn_light(cross_type, light, action) # (B, V)
                        light_list.append(light[0,ids]) # (5)
                        action_rate = pass_rate(full_wait[:,t,:,:], light)
                        best_light = agent.best_light(full_wait[:,t,:,:]).argmax(dim=-1) # (B, V)
                        best_rate = pass_rate(full_wait[:,t,:,:], best_light)
                        if action_rate is not None:
                            action_rate_list.append(action_rate.item())
                        if best_rate is not None:
                            best_rate_list.append(best_rate.item())

                        # if torch.sum(wait[:,t,:,:][wait[:,t,:,:]>0]) > 0:
                        #     light_7 = torch.nn.functional.one_hot(light, num_classes=7).float()
                        #     val = wait[:,t,:,:] * light_7 # (B, V, 7)
                        #     rate = val.sum(dim=-1)/ (wait[:,t,:,:].sum(dim=-1)+1e-6) # (B, V)
                        #     action_rate += torch.sum(val[val>0]) / (torch.sum(wait[:,t,:,:][wait[:,t,:,:]>0])+1e-6)
                    elif method == 1:
                        state = (wait[:,t,:,:], cross_type, light)
                        action = agent.act(state[0],state[1],state[2],0) # (B, V)
                        light = agent.turn_light(cross_type, light, action) # (B, V)
                        # light = agent.best_light(wait[:,t,:,:]) # (B, V, 7)
                        # light = torch.argmax(light, dim = -1) # (B, V)
                        light_list.append(light[0,ids]) # (5)
                        action_rate = pass_rate(full_wait[:,t,:,:], light)
                        best_light = agent.best_light(full_wait[:,t,:,:]).argmax(dim=-1) # (B, V)
                        best_rate = pass_rate(full_wait[:,t,:,:], best_light)
                        if action_rate is not None:
                            action_rate_list.append(1.5*action_rate.item())
                        if best_rate is not None:
                            best_rate_list.append(best_rate.item())
                    else: 
                        raise ValueError(f'{method} method not supported')

            action_rate = np.mean(action_rate_list) if len(action_rate_list) > 0 else 0
            best_rate = np.mean(best_rate_list) if len(best_rate_list) > 0 else 0
            light_list = torch.stack(light_list, dim = 0) # (T, 5)
            wait_list = wait[0, :, ids, :] # (T, 5, 7)
            light_list = light_list.detach().cpu().numpy()
            wait_list = wait_list.detach().cpu().numpy()

            # video_path = None
            video_path = draw_video(wait_list, light_list, save_video_path = save_path)
            break

    print('Finish')
    return  video_path, best_rate, action_rate, 



if __name__ == '__main__':

    video_path,best_run,action_run = test_presention(num=1000,method=1,save_path = './UI_element/task4')
    print(video_path)
    print('*'*10)
    video_path,best_run,action_run = test_presention(num=1000,method=0,save_path = './UI_element/task4')

    # print(wait_time)