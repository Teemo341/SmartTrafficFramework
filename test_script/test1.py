from device_selection import get_local_device
from dataloader import SmartTrafficDataset, SmartTrafficDataloader
from task1.test1 import define_model
import torch
import argparse
import numpy as np
import networkx as nx
import pickle
import matplotlib.pyplot as plt
from utils import calculate_bounds, read_city, transfer_graph, adj_m2adj_l
import pandas as pd
import cv2
import os
from tqdm import tqdm
import time

weights_path = 'weights/best/jinan/task1/best_model_0.1457.pth'
cfg = { 'T':60,
        'max_len':63,
        'vocab_size':23313,
        'batch_size':128,
        'epochs':5,
        'learning_rate':0.001,
        'n_embd':32,
        'n_hidden':16,
        'n_layer':8,
        'dropout':0.1,
        'model_read_path':None,
        'model_save_path':'weights/jinan/task1',
        'trajs_path_train':'data/jinan/edge_traj_test1/',
        'trajs_path':'data/jinan/edge_traj_repeat_one_by_one/',
        'device':get_local_device(0),
        'n_head':8,    
        }
cfg['block_size'] = cfg['T']

def select(list):
    result = []
    for arr in list:
        mask = (arr == 0) | (arr == 1)
        indices = np.where(mask)[0]
        if indices.size > 0:
            first_idx = indices[0]
            truncated = arr[:first_idx]
        else:
            truncated = arr
        result.append(truncated)
    return result

       

def plot_map1(fig_size=200, save_path='./task1/map1.png'):
    edges, pos = read_city('jinan', path='data')
    weight = [edge[2] for edge in edges]
    adj_table = get_weighted_adj_table(edges, pos, weight, max_connection=9)
    adj_table = np.load('data/jinan/adj_l.npy')
    G = transfer_graph(adj_table)
    for i in pos:
        pos[i] = pos[i][:-1]
    x_min, x_max, y_min, y_max = calculate_bounds(pos)
    fig, ax = plt.subplots(1, 1, figsize=(fig_size, fig_size * (y_max - y_min) / (x_max - x_min)))
    ax.set_facecolor('black')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(np.arange(x_min, x_max, 1))
    ax.set_yticks(np.arange(y_min, y_max, 1))
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    ax.axvline(x=0, color='k', linestyle='--', linewidth=0.5)
    
    # 绘制边
    nx.draw_networkx_edges(G, pos, width=fig_size/15, alpha=0.3, edge_color='white', ax=ax, arrows=False)
    
    # 新增代码：绘制节点编号
    node_labels = {node: str(node) for node in G.nodes()}
    nx.draw_networkx_labels(
        G, pos,
        labels=node_labels,
        font_size=fig_size/30,        # 字体大小与图像尺寸关联
        font_color='yellow',       # 高对比度颜色
        ax=ax,
        verticalalignment='center' # 垂直居中
    )
    
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.close()


def filter_connected_paths(all_paths, G):
    """
    过滤掉不连通的路径
    :param all_paths: 路径列表，每个路径是节点数组（如 [0, 1, 2]）
    :param G: NetworkX 图对象
    :return: 连通路径的列表
    """
    valid_paths = []
    for path in all_paths:
        is_valid = True
        # 遍历路径中的每一对相邻节点
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            # 检查边是否存在
            if not G.has_edge(u, v):
                is_valid = False
                break
        if is_valid:
            valid_paths.append(path)
    return valid_paths

def plot_volume1(traj1, traj2 = None, fig_size=20, save_path='task1.png', return_cv = False):
    # G networkx graph
    # pos position of the nodes, get from read_city('boston')[1]
    # volume_single: V
    # traj: list of nodes representing a path

    edges, pos = read_city('jinan', path='data')
    weight = [edge[2] for edge in edges]
    adj_table = get_weighted_adj_table(edges, pos, weight, max_connection=9)
    G = transfer_graph(adj_table)
    for i in pos:
        pos[i] = pos[i][:-1]
    
    x_min, x_max, y_min, y_max = calculate_bounds(pos)
    fig, ax = plt.subplots(1, 1, figsize=(fig_size, fig_size * (y_max - y_min) / (x_max - x_min)))
    ax.set_facecolor('black')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(np.arange(x_min, x_max, 1))
    ax.set_yticks(np.arange(y_min, y_max, 1))
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    ax.axvline(x=0, color='k', linestyle='--', linewidth=0.5)
    nx.draw_networkx_edges(G, pos, width=fig_size/15, alpha=0.3, edge_color='white', ax=ax, arrows=False)

    start_1 = traj1[0]
    nx.draw_networkx_nodes(G, pos, nodelist=[start_1], node_size=fig_size/10, node_color='blue', ax=ax)
    end_1 = traj1[-1]
    nx.draw_networkx_nodes(G, pos, nodelist=[end_1], node_size=fig_size/10, node_color='blue', ax=ax)
    traj1 = [(traj1[i], traj1[i + 1]) for i in range(len(traj1) - 1)]
    nx.draw_networkx_edges(G, pos, edgelist=traj1, width=fig_size / 15, alpha=1.0, edge_color='green', ax=ax, arrows = False)
    
    if traj2 is not None:
        start_2 = traj2[0]
        nx.draw_networkx_nodes(G, pos, nodelist=[start_2], node_size=fig_size/10, node_color='blue', ax=ax)
        end_2 = traj2[-1]
        nx.draw_networkx_nodes(G, pos, nodelist=[end_2], node_size=fig_size/10, node_color='blue', ax=ax)
        traj2 = [(traj2[i], traj2[i + 1]) for i in range(len(traj2) - 1)]
        nx.draw_networkx_edges(G, pos, edgelist=traj2, width=fig_size / 15, alpha=0.3, edge_color='red', ax=ax, arrows = False)
    
    # Display the figure
    plt.tight_layout()
    # plt.show()
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    if return_cv:
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        plt.close()
        return img


def plot_video(traj, fig_size=20, save_video_path=None):
    """Generate a video from a list of trajectories.
    Args:
        traj (list of list): List of trajectories, where each trajectory is a list of nodes.
    """
    if save_video_path is None:
        save_video_path = './task1/video/pred'
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # 初始化 tqdm 进度条
    max_len = max(len(t) for t in traj)
    progress_bar = tqdm(range(max_len), desc="Generating Video")
    
    # 初始化地图
    edges, pos = read_city('jinan', path='data')
    weight = [edge[2] for edge in edges]
    adj_table = get_weighted_adj_table(edges, pos, weight, max_connection=9)
    adj_table = np.array(adj_table)
    np.save('data/jinan/adj_l.npy', adj_table)
    G = transfer_graph(adj_table)
    for i in pos:
        pos[i] = pos[i][:-1]
    
    x_min, x_max, y_min, y_max = calculate_bounds(pos)
    fig, ax = plt.subplots(1, 1, figsize=(fig_size, fig_size * (y_max - y_min) / (x_max - x_min)))
    ax.set_facecolor('black')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(np.arange(x_min, x_max, 1))
    ax.set_yticks(np.arange(y_min, y_max, 1))
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    ax.axvline(x=0, color='k', linestyle='--', linewidth=0.5)
    nx.draw_networkx_edges(G, pos, width=fig_size/15, alpha=0.3, edge_color='white', ax=ax, arrows=False)
    for j in range(len(traj)):
        start_1 = traj[j][0]
        nx.draw_networkx_nodes(G, pos, nodelist=[start_1], node_size=fig_size/4, node_color='blue', ax=ax)
        end_1 = traj[j][-1]
        nx.draw_networkx_nodes(G, pos, nodelist=[end_1], node_size=fig_size/4, node_color='blue', ax=ax)
        
    for i in progress_bar:
        # 1. 计算画图时间
        start_plot = time.time()

        # 画对应的路段
        for j in range(len(traj)):
            if i > len(traj[j])-2:
                continue
            traj1 = [(traj[j][i], traj[j][i + 1])]
            nx.draw_networkx_edges(G, pos, edgelist=traj1, width=fig_size / 5, alpha=1.0, edge_color='red', ax=ax, arrows = False)
            plt.tight_layout()
        
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        frame = frame[:,:,1:]
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        plt.close()
        plot_time = time.time() - start_plot
        
        # 2. 初始化 VideoWriter（如果是第一帧）
        if i == 0:
            frame_height, frame_width, _ = frame.shape
            out = cv2.VideoWriter(f'{save_video_path}/video.mp4', fourcc, 2, (frame_width, frame_height))
        
        # 3. 计算保存时间
        start_save = time.time()
        out.write(frame)  # 注意原代码是 frame_bgr，这里假设 frame 已经是 BGR 格式
        save_time = time.time() - start_save
        
        # 4. 更新 tqdm 的实时信息
        progress_bar.set_postfix({
            "plot_time (s)": f"{plot_time:.3f}",
            "save_time (s)": f"{save_time:.3f}",
            "total_time (s)": f"{plot_time + save_time:.3f}"
        })

    out.release()
    return f'{save_video_path}/video.mp4'

def train_presention(batch_size=64,epochs=4,lr = 0.001,device='cuda:1'):
    
    cfg['batch_size'] = batch_size
    cfg['epochs'] = epochs
    cfg['learning_rate'] = lr
    cfg['device'] = device
    dataset1 = SmartTrafficDataset(None,mode="task1",trajs_path=cfg['trajs_path_train'],T=cfg['T'],max_len=cfg['max_len']) 
    data_loader1 = SmartTrafficDataloader(dataset1,batch_size=cfg['batch_size'],shuffle=True, num_workers=4).get_test_data()
    from train import train1
    train1(cfg, data_loader1)        

def test_presention(x=None,edge=None):

    cfg['model_read_path'] = weights_path
    node_edge_map_path ='data_/jinan/node_edge_map_dict.pkl' 
    with open(node_edge_map_path,'rb') as f:
        map_dict = pickle.load(f)
    adjcent_path = 'data_/jinan/adjcent.npy'
    adjcent = np.load(adjcent_path)
    adj_l = adj_m2adj_l(adjcent)
    # G = transfer_graph(adj_l)
    if x is None:
        o,d = np.random.choice(np.arange(1,8909),2)
    else:
        o,d = x

    model = define_model(cfg)
    model.load_state_dict(torch.load(weights_path,map_location=model.device))

    model.eval()
    if edge is None:
        raise ValueError('edge is None')
    # print(edge[0],edge[1])
    adjcent = np.load('data_/jinan/adjcent.npy')
    traj = model.generate_traj(edge[0],edge[1],adj_l)
    # print(traj)

    return traj

def transfer_edge_to_node(od_list):

    node_list = []
    for i in range(len(od_list)):
        if i == 0 :
            node_list.append(od_list[i][0])
            node_list.append(od_list[i][1])
        else:
            node_list.append(od_list[i][1])
    return node_list



# get weighted adjacency table, return 0-indexing
def get_weighted_adj_table(edges, pos, capacity, normalization = True, quantization_scale = None, max_connection = 4):

    adj_table = np.zeros([len(pos),max_connection, 2]) # [node, connection, [target_node, weight]]

    # add edges to adj_table
    for i in range(len(edges)):
        if np.sum(adj_table[edges[i][0],:,0]!=0) >= max_connection: # already full
            raise ValueError('Error: max_connection is too small')
        elif adj_table[edges[i][0],np.sum(adj_table[edges[i][0],:,0]!=0),0] == max_connection: # duplicate edge
            raise ValueError('Error: duplicate edge')
        else:
            adj_table[edges[i][0],np.sum(adj_table[edges[i][0],:,0]!=0)] = [edges[i][1]+1,capacity[i]] # [target_node, weight], add to the first empty slot
            #! the adj_table[1][0][0] is the first connection of road 2,
            #! the adj_table[1][0][0] = 1 means that road 2 is connected to road 1
            #! the ajd_table[1][0][1] is the road length from road 2 to road 1
    
    if normalization:
        adj_table[:,:,1] = adj_table[:,:,1]/np.max(adj_table[:,:,1])
    if quantization_scale:
        adj_table[:,:,1] = np.ceil(adj_table[:,:,1]*quantization_scale)
        
    return adj_table #! 0-indexing


def translate_roadtype_to_capacity(roadtype):
    dic = {'living_street': 1, 'motorway': 10, 'motorway_link': 10, 'primary': 8, 'primary_link': 8, 'residential': 2, 'secondary': 6, 'secondary_link': 6, 'service': 3, 'tertiary': 4, 'tertiary_link': 4, 'trunk': 7, 'trunk_link': 7, 'unclassified': 5}
    return dic[roadtype]


def djikstra(start, end):
    edges, pos = read_city('jinan', path='data')
    weight = [edge[2] for edge in edges]
    adj_table = get_weighted_adj_table(edges, pos, weight, max_connection=9)
    G = transfer_graph(adj_table)
    path = nx.dijkstra_path(G, start, end)
    return path
        

def test_presention1(num=10, generate_type = 'pred', save_path = None):
    if save_path is None:
        save_path = f'./task1/video'
    map_path = 'data_/jinan/edge_node_map_dict.pkl'
    with open(map_path,'rb') as f:
        map_dict = pickle.load(f)
    #dataset1 = SmartTrafficDataset(None,mode="task1",trajs_path=cfg['trajs_path'],T=cfg['T'],max_len=cfg['max_len']) 
    #data_loader1 = SmartTrafficDataloader(dataset1,batch_size=1,shuffle=False, num_workers=4)
    # i = 0

    #x = dataset1[k]

    #od = x['cond']
    #e1 = od[0,0,0]
    #e_1 = od[0,0,1]
    #o = map_dict[str(e1.item())][0]
    #d = map_dict[str(e_1.item())][1]
    # traj = test_presention((o-1,d-1),(e1,e_1))

    # make od
    all_numbers = np.random.choice(np.arange(1, 23313), size=num*2, replace=False)
    
    # 重组为10对（二维数组）
    pairs = all_numbers.reshape(num, 2)
    e1 = pairs[:,0:1]
    e_1 = pairs[:,1:2]
    o = np.zeros_like(e1)
    d = np.zeros_like(e_1)
    for i in range(num):
        o[i,:] = map_dict[str(e1[i,:].item())][0]
        d[i,:] = map_dict[str(e_1[i,:].item())][1]

    # generate traj
    if generate_type == 'pred':
        traj = test_presention((o-1,d-1),(e1,e_1))
        # print(traj.shape)
        # print(traj)
        # traj = np.array(traj)
        pred_traj = []
        for i in range(traj.shape[0]):
            traji = traj[i,:,:]
            traji = traji[traji>0]
        
            pred_traj_ = [map_dict[str(traji[i])] for i in range(len(traji))]
            pred_traji= transfer_edge_to_node(pred_traj_)
            pred_traji = np.array(pred_traji) # 1-indexing
            pred_traj.append(pred_traji)
        
        edges, pos = read_city('jinan', path='data')
        weight = [edge[2] for edge in edges]
        adj_table = get_weighted_adj_table(edges, pos, weight, max_connection=9)
        G = transfer_graph(adj_table)
        pred_traj = filter_connected_paths(pred_traj, G)
        print('pred_traj',pred_traj)

        video_dir = plot_video(pred_traj ,fig_size=20, save_video_path=f'{save_path}/pred')
        
        return video_dir

    elif generate_type == 'dj':
        dj = []
        for i in tqdm(range(len(o))):
            success = False
            attempts = 0
            while not success:
                try:
                    dj_traj = djikstra(o[i][0], d[i][0])
                    dj_traj = np.array(dj_traj, dtype=int)  # 1-indexing
                    dj.append(dj_traj)
                    success = True
                except Exception:
                    # 随机重新生成od直到成功
                    all_numbers = np.random.choice(np.arange(1, 23313), size=2, replace=False)
                    e1_new, e_1_new = all_numbers[0], all_numbers[1]
                    o[i][0] = map_dict[str(e1_new)][0]
                    d[i][0] = map_dict[str(e_1_new)][1]
                    attempts += 1
                    if attempts > 100:  # 防止死循环
                        raise RuntimeError("无法生成连通的od对，请检查数据或增加尝试次数")
        
        print('Djs:',dj)
        video_dir = plot_video(dj,fig_size=20, save_video_path=f'{save_path}/dj')
    
    else:
        raise ValueError('generate_type is not in [pred, dj]')
    
    return video_dir
    
if __name__ == '__main__':
    #! 轨迹预测的模型并行也未完成

    plot_map1()

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--k', type=int, default=10)
    # args = parser.parse_args()
    # video_path = test_presention1(args.k,'dj')
 
    #test_presention1(3)``

