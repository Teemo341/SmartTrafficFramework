import torch
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cv2
import pandas as pd
import os


# transfer adj table to weighted adj matrix
def transfer_table_to_matrix(adj_table):
    weighted_adj_matrix = np.zeros([len(adj_table),len(adj_table)])
    for i in range(len(adj_table)):
        for j in range(len(adj_table[i])):
            if adj_table[i,j,1] != 0:
                weighted_adj_matrix[i,int(adj_table[i,j,0]-1)] = adj_table[i,j,1]
    return weighted_adj_matrix

# test function whether a trajectory skip a node
def test_trajectory_error(pred, weighted_adj, trajectory):
    i = 0
    while i < len(pred)-1 and pred[i] != 0 and pred[i+1] != 0:
        if weighted_adj[pred[i]-1,pred[i+1]-1] == 0: # currently the codebook is 1-indexing
            return True
        i += 1
    j = 0 
    while trajectory[j] != 0 and trajectory[j+1] != 0 and j < len(trajectory)-1:
        j += 1
    if pred[i] != trajectory[j]:
        return True
    else:
        return False

# Calculate the length of a trajectory
def calculate_length(trajectory, weighted_adj):
    length = 0
    for i in range(len(trajectory)-1):
        if trajectory[i+1] != 0:
            length += weighted_adj[trajectory[i]-1,trajectory[i+1]-1] # currently the codebook is 1-indexing
        else:
            break
    return length

def calculate_length_(trajectory, weighted_adj):
    length = 0
    for i in range(len(trajectory)-1):
        length += weighted_adj[trajectory[i],trajectory[i+1]]
    return length

# transfer node, wrighted_adj to graph
def transfer_graph(adj_table):
    # adj_table: B x N x V x 4 x 2, 1-indexing, 0 is special token, 0 is not exist, 1 is exist
    #! G is 0-indexing
    G = nx.DiGraph()
    for i in range(len(adj_table)):
        G.add_node(i)
    for i in range(len(adj_table)):
        for j in range(len(adj_table[i])):
            if adj_table[i,j,1] != 0:
                G.add_edge(i,adj_table[i,j,0]-1,weight=adj_table[i,j,1]) #adj_table is 1-indexing, G is 0-indexing
    return G


def print_trajectory(predict_trajectory, test_trajectory, condition = None):
    for i in range(len(predict_trajectory)):
        print(f"weighted graph {i+1:<3},")
        result_txt_test = f"true: {np.array2string(test_trajectory[i])}"
        print(result_txt_test)
        
        result_txt_predict = f"pred: {np.array2string(predict_trajectory[i])}"
        print(result_txt_predict)

        if condition is not None:
            result_txt_condition = f"cond: {np.array2string(condition[i])}"
            print(result_txt_condition)
        
        print("\n")
        

# remove the special token, input one traj as numpy, return a list without special token
def remove_special_token(trajectory):
    new_trajectory = []
    start=0
    while trajectory[start] == 0:
        start += 1 # move to the first non-special token
    end = len(trajectory)-1
    while trajectory[end] == 0:
        end -= 1 # move to the last non-special token
    new_trajectory = trajectory[start:end+1]
    for i in range(len(new_trajectory)):
        if new_trajectory[i] == 0:
            new_trajectory[i] = new_trajectory[i-1] # if exist special token, move to former index
        else:
            new_trajectory[i] -= 1
    return new_trajectory

# transfer grid into position
def transfer_position(grid_size):
    pos = {}
    for i in range(grid_size):
        for j in range(grid_size):
            pos[i*grid_size+j] = (i,j)
    return pos

# Function to calculate the bounds of the graph
def calculate_bounds(pos):
    x_values = [node[0] for node in pos.values()]
    y_values = [node[1] for node in pos.values()]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    # x_min, x_max = x_min - 0.1 * abs(x_max - x_min), x_max + 0.1 * abs(x_max - x_min)
    # y_min, y_max = y_min - 0.1 * abs(y_max - y_min), y_max + 0.1 * abs(y_max - y_min)

    return x_min, x_max, y_min, y_max


# plot the volume of the graph, only one frame
def plot_volume(G, pos, volume_single, max_volume, figure_size=20, save_path=None):
    # G networkx graph
    # pos position of the nodes, get from  read_city('boston')[1]
    # volume_single: V

    x_min, x_max, y_min, y_max = calculate_bounds(pos)
    fig, ax = plt.subplots(1, 1, figsize=(figure_size, figure_size*(y_max-y_min)/(x_max-x_min)))
    ax.set_facecolor('black')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(np.arange(x_min, x_max, 1))
    ax.set_yticks(np.arange(y_min, y_max, 1))
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    ax.axvline(x=0, color='k', linestyle='--', linewidth=0.5)

    edge_colors = np.array([(volume_single[int(u)-1]+volume_single[int(v)-1])/2 for u, v in G.edges()])
    edge_colors = edge_colors / max_volume
    edge_colors = plt.cm.RdYlBu(1 - edge_colors)
    # edge_colors = plt.cm.rainbow(edge_colors)
    # colors = ["white","blue", "green", "yellow", "red"]
    # cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors)
    # edge_colors = cmap(edge_colors)

    # nx.draw_networkx_nodes(G, pos, node_size=100, node_color='gray',ax=ax)
    nx.draw_networkx_edges(G, pos, width=figure_size/15, alpha=1, edge_color='black', ax=ax, arrows=False)
    nx.draw_networkx_edges(G, pos, width=figure_size/15, alpha=1, edge_color=edge_colors, ax=ax, arrows=False)

    # ax.legend()
    plt.tight_layout()
    plt.show()
    if save_path != None:
        plt.savefig(save_path, bbox_inches='tight',facecolor = 'black')

def make_volume_frames(G, pos, volume_all, max_volume, figure_size=20, save_path = './videos'):
    # G: networkx graph
    # pos: position of the nodes
    # volume_all: (T, V) where T is time and V is volume
    # max_volume: maximum volume for normalization

    # Calculate bounds for the plot
    x_min, x_max, y_min, y_max = calculate_bounds(pos)

    colors = ["white", "blue", "green", "yellow", "red"]
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors)

    for i in range(len(volume_all)):
        volume_single = volume_all[i]
        edge_colors = np.array([(volume_single[int(u)-1] + volume_single[int(v)-1]) / 2 for u, v in G.edges()])
        edge_colors = edge_colors / max_volume
        # edge_colors = cmap(edge_colors)
        edge_colors = plt.cm.RdYlBu(1 - edge_colors)

        # Create a new figure for the current frame
        fig, ax = plt.subplots(figsize=(figure_size, figure_size * (y_max - y_min) / (x_max - x_min)))
        ax.set_facecolor('black')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(np.arange(x_min, x_max, 1))
        ax.set_yticks(np.arange(y_min, y_max, 1))
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
        ax.axvline(x=0, color='k', linestyle='--', linewidth=0.5)

        # Draw nodes and edges
        nx.draw_networkx_edges(G, pos, width=figure_size / 15, alpha=1, edge_color='black', ax=ax, arrows=False)
        nx.draw_networkx_edges(G, pos, width=figure_size / 15, alpha=1, edge_color=edge_colors, ax=ax, arrows=False)

        # Convert the figure to an image and write it to the video
        plt.axis('off')  # Hide axes
        plt.draw()
        plt.tight_layout()
        if save_path != None:
            plt.savefig(f'{save_path}/frame_{i}.png', bbox_inches='tight', facecolor='black')

        plt.close(fig)  # Close the figure

def make_video_from_frames(frames_dir, video_path):
    # Create a VideoCapture object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame = cv2.imread(f'{frames_dir}/frame_{0}.png')
    frame_height, frame_width, _ = frame.shape
    out = cv2.VideoWriter(video_path, fourcc, 2, (frame_width, frame_height))

    # Iterate over all the frames
    for i in range(len(os.listdir(frames_dir))):
        frame = cv2.imread(f'{frames_dir}/frame_{i}.png')
        out.write(frame)

    # Release the VideoCapture object
    out.release()

# Plot the trajectories on the graph
def plot_trajs(ax, G, pos, weighted_adj, traj, traj_ = None, ground_truth=False):

    x_min, x_max, y_min, y_max = calculate_bounds(pos)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(np.arange(x_min, x_max, 1))
    ax.set_yticks(np.arange(y_min, y_max, 1))
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    ax.axvline(x=0, color='k', linestyle='--', linewidth=0.5)

    nx.draw_networkx_nodes(G, pos, node_size=100, node_color='gray',ax=ax)
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, edge_color='gray', ax=ax)
    # nx.draw_networkx_edge_labels(G, pos, edge_labels={(i, j): f'{weighted_adj[i, j]:.2f}' for i, j in G.edges()}, label_pos=0.7, ax=ax)
    

    # Plot trajectories
    x = [pos[node][0]+np.random.normal(0,0.001*(x_max-x_min)) for node in traj]
    y = [pos[node][1]+np.random.normal(0,0.001*(y_max-y_min)) for node in traj]
    traj_len = calculate_length_(traj, weighted_adj)
    ax.plot(x, y, marker='o', linewidth = 1.0, alpha=0.5, markersize=10, label=f'Pred Trajectory {traj_len}', color='blue')
    ax.plot(x[0], y[0], marker='d', markersize=20, color='blue', alpha=0.5)
    ax.plot(x[-1], y[-1], marker='*', markersize=20, color='blue', alpha=0.5)

    if traj_ is not None:
        x = [pos[node][0]+np.random.normal(0,0.001*(x_max-x_min)) for node in traj_]
        y = [pos[node][1]+np.random.normal(0,0.001*(y_max-y_min)) for node in traj_]
        traj_len = calculate_length_(traj_, weighted_adj)
        ax.plot(x, y, marker='o',  linewidth = 1.0, alpha=0.5, markersize=10, label=f'Real Trajectory {traj_len}', color='red')
        ax.plot(x[0], y[0], marker='d', markersize=20, color='red', alpha=0.5)
        ax.plot(x[-1], y[-1], marker='*', markersize=20, color='red', alpha=0.5)

    ax.legend()


# Plot the training and validation losses
def plot_losses(train_losses, val_losses, eval_interval):
    x = range(0, len(train_losses)*eval_interval, eval_interval)
    plt.plot(x, train_losses, label='train')
    plt.plot(x, val_losses, label='val')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


# correct the jump of predictions
def correct_pred(preds, logits, condition, adj_matrix):
    # preds: L
    # logits: L V
    # condition: 2
    # adj_matrix: V V
    new_preds = np.zeros_like(preds)
    new_preds[0] = preds[0]
    for i in range(1, len(preds)):
        if new_preds[i-1] == condition[1]:
            break
        if adj_matrix[new_preds[i-1]-1, preds[i]-1] != 0:
            new_preds[i] = preds[i]
        else:
            connection_filter = np.concatenate((np.array([0]),adj_matrix[new_preds[i-1]-1]))
            new_preds[i] = np.argmax(logits[i]*connection_filter)
    return new_preds

def correct_pred_tensor(preds, logits, condition, adj_matrix):
    # preds: L
    # logits: L V
    # condition: 2
    # adj_matrix: V V
    new_preds = torch.zeros_like(preds, device=preds.device)
    new_preds[0] = preds[0]
    for i in range(1, len(preds)):
        if new_preds[i-1] == condition[1]:
            break
        if adj_matrix[new_preds[i-1]-1, preds[i]-1] != 0:
            new_preds[i] = preds[i]
        else:
            connection_filter = torch.concatenate((torch.tensor([0],device = preds.device),adj_matrix[new_preds[i-1]-1]))
            new_preds[i] = torch.argmax(logits[i]*connection_filter)
    return new_preds

def test_functions(preds, trajectory, adj_table):
        error_num = 0
        real_true_num = 0
        fake_true_num = 0
        len_rate_sum = 0
        len_10_num = 0
        for j in range(trajectory.shape[0]):
            for k in range(trajectory.shape[1]):
                pred = preds[j,k]
                weighted_adj = transfer_table_to_matrix(adj_table[j])
                if test_trajectory_error(pred, weighted_adj, trajectory[j,k]):
                    error_num += 1
                elif calculate_length(trajectory[j,k], weighted_adj) != 0:
                    len_true = calculate_length(trajectory[j,k], weighted_adj)
                    len_pred = calculate_length(pred, weighted_adj)
                    len_rate = len_pred/len_true
                    len_rate_sum += len_rate
                    if len_true == len_pred:
                        if (preds[j,k] == trajectory[j,k]).any():
                            real_true_num += 1
                        else:
                            fake_true_num += 1
                    if len_rate <= 1.1:
                        len_10_num += 1

        error_rate = error_num/(trajectory.shape[0]*trajectory.shape[1])
        real_true_rate = real_true_num/(trajectory.shape[0]*trajectory.shape[1])
        fake_true_rate = fake_true_num/(trajectory.shape[0]*trajectory.shape[1])
        length_rate_avg = len_rate_sum/(trajectory.shape[0]*trajectory.shape[1]-error_num) if error_num != trajectory.shape[0]*trajectory.shape[1] else 0
        len_10_rate = len_10_num/(trajectory.shape[0]*trajectory.shape[1]-error_num) if error_num != trajectory.shape[0]*trajectory.shape[1] else 0

        return error_rate, real_true_rate, fake_true_rate, length_rate_avg, len_10_rate


def calculate_load(traj_probability, method = 'map'):
    # traj_probability: B, T, V

    if method == 'map':
        max_indices = np.argmax(traj_probability, axis=-1)
        one_hot = np.zeros_like(traj_probability)
        B, T, V = traj_probability.shape
        one_hot[np.arange(B)[:, None], np.arange(T), max_indices] = 1
        load = np.sum(one_hot, axis = 0) # T, V
        load = load[:,1:] # remove the special token
    elif method == 'single_prob':
        max_indices = np.argmax(traj_probability, axis=-1)
        one_hot = np.zeros_like(traj_probability)
        B, T, V = traj_probability.shape
        one_hot[np.arange(B)[:, None], np.arange(T), max_indices] = 1
        filtered = traj_probability*one_hot
        load = np.sum(filtered, axis = 0)
        load = load[:,1:]
    elif method == 'all_prob':
        traj_probability = traj_probability[:,:,1:] # remove the special token
        load = np.sum(traj_probability, axis = 0) # T, V-1
    else:
        raise ValueError(f"method should be 'map', 'single_prob' or 'all_prob', but got {method}")
    return load

def preprocess_node(node_dir):
    #! 0-indexing
    pos = pd.read_csv(node_dir)
    pos = { int(nid): (float(lon), float(lat), bool(hasc)) for _, (nid, lon, lat, hasc) in pos.iterrows() }
    return pos

def preprocess_edge(edge_dir):
    #! 0-indexing
    edges = pd.read_csv(edge_dir)
    edges = [(int(o), int(d), float(l), translate_roadtype_to_capacity(c)) for _, (o, d, c, geo, l) in edges.iterrows()
    ]
    return edges

def translate_roadtype_to_capacity(roadtype):
    dic = {'living_street': 1, 'motorway': 10, 'motorway_link': 10, 'primary': 8, 'primary_link': 8, 'residential': 2, 'secondary': 6, 'secondary_link': 6, 'service': 3, 'tertiary': 4, 'tertiary_link': 4, 'trunk': 7, 'trunk_link': 7, 'unclassified': 5}
    return dic[roadtype]

def read_city(city, path='./data'):
    # if city in ['boston', 'paris']:
    #     origin_data = pd.read_csv(path + '/'+ city + '_data.csv').to_dict(orient='list')
    #     edges, pos = preprocess_data_boston(origin_data) #! 0-indexing
    if city in ['jinan']:
        node_dir = f"{path}/{city}/node_{city}.csv"
        edge_dir = f"{path}/{city}/edge_{city}.csv"
        pos = preprocess_node(node_dir) #! 0-indexing
        edges = preprocess_edge(edge_dir) #! 0-indexing

    return edges, pos