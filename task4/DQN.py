import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque


# 定义基础transformer
class blcok(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        # cross
        self.sa_cross = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ln_cross_1 = nn.LayerNorm(embed_dim)
        self.ffwd_cross = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim))
        self.ln_cross_2 = nn.LayerNorm(embed_dim)
        # light
        self.sa_light = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ln_light_1 = nn.LayerNorm(embed_dim)
        self.ffwd_light = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim))
        self.ln_light_2 = nn.LayerNorm(embed_dim)
        # wait
        self.sa_wait = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ln_wait_1 = nn.LayerNorm(embed_dim)
        self.ca_wait_corss = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ca_wait_light = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ln_wait_2 = nn.LayerNorm(embed_dim)
        self.ffwd_wait = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim))
        self.ln_wait_3 = nn.LayerNorm(embed_dim)

    def forward(self, wait, cross, light):
        # cross (V, hidden_size)
        cross = self.ln_cross_1(cross)
        cross = cross+self.sa_cross(cross, cross, cross)[0]
        cross = self.ln_cross_2(cross+self.ffwd_cross(cross))

        # light (B, V, hidden_size)
        light = self.ln_light_1(light)
        light = light+self.sa_light(light, light, light)[0]
        light = self.ln_light_2(light+self.ffwd_light(light))

        # wait (B, V, hidden_size)
        wait = self.ln_wait_1(wait)
        wait = wait+self.sa_wait(wait, wait, wait)[0]
        wait = self.ln_wait_2(wait)
        cross = self.ln_wait_2(cross).unsqueeze(0).expand(wait.shape[0], -1, -1) # (B, V, hidden_size)
        light = self.ln_wait_2(light)
        wait = wait + self.ca_wait_corss(wait, cross, cross)[0] + self.ca_wait_light(wait, light, light)[0]
        wait = self.ln_wait_3(wait+self.ffwd_wait(wait))

        return wait # (B, V, E)

# 定义 DQN 模型
class DQN(nn.Module):
    def __init__(self, num_layers = 6, hidden_size = 64, num_heads = 4, wait_quantization = 10, dropout = 0.1):
        super(DQN, self).__init__()
        self.wait_embedding = nn.Embedding(wait_quantization+2, hidden_size)
        self.cross_embedding = nn.Embedding(2, hidden_size)
        self.light_embedding = nn.Embedding(7, hidden_size)
        self.in_proj = nn.Sequential(nn.Linear(hidden_size, hidden_size*3), nn.GELU(), nn.Linear(hidden_size*3, hidden_size),nn.AdaptiveAvgPool2d((1, None)))
        self.blocks = nn.ModuleList([blcok(hidden_size, num_heads, dropout) for _ in range(num_layers)])
        self.out_proj = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.GELU(), nn.Linear(hidden_size, 2))

    def forward(self, wait, cross_type, light):
        # wait (B, V, 7): the waiting number at each node, int [-1, 10] ,V is the number of node, 7 means 7 directions
        # wait = -1 means not exist or observable, wait =0 means no car, wait = 1-10 means the number of cars
        # cross_type (V): int {3, 4}, 3 means 3-direction crossroad, 4 means 4-direction crossroad
        # light (B, V): current traffic light status, int 0-3 or 4-6, corresponding to 4-direction or 3-direction

        wait = wait+1 # -1 -> 0
        if wait.max() > 16 or wait.min() < 0:
            print(wait.max(), wait.min())
        wait = self.wait_embedding(wait) # (B, V, 7, hidden_size)
        wait = self.in_proj(wait).squeeze(2) # (B, V, 1, hidden_size)
        wait = wait.squeeze(2) # (B, V, hidden_size)
        cross_type = self.cross_embedding(cross_type-3) # (V, hidden_size)
        light = self.light_embedding(light) # (B, V, hidden_size)

        for block in self.blocks:
            wait = block(wait, cross_type, light) # (B, V, hidden_size)

        act_values = self.out_proj(wait) # (B, V, 2)

        # act_values (B, V, 2): the Q value of each node, 0 means the value of keep, 1 means the value of change
        return act_values

class DQNAgent:
    #agent = DQNAgent(args.device, args.memory_device, args.memory_len, args.n_layer, args.n_embd, args.n_head, args.wait_quantization, args.dropout,lr=args.learning_rate)
    def __init__(self, device, memory_device = 'cpu', memory_len=2000, num_layers = 6, hidden_size = 64, num_heads = 4, wait_quantization = 10, dropout = 0.1,lr=5e-4, batch_size = 32):
        self.memory = deque(maxlen=memory_len)
        self.gamma = 0.95    # 折扣因子
        self.epsilon = 1.0    # 探索率
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = batch_size
        self.device = device
        self.memory_device = memory_device
        self.lr = lr

        self.model = DQN(num_layers, hidden_size, num_heads, wait_quantization, dropout)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        self.criterion = nn.MSELoss()

    def act(self, wait, cross_type, light, epsilon = 0):
        # wait (B, V, 7): the waiting number at each node, int [-1, 10] ,V is the number of node, 7 means 7 directions
        # wait = -1 means not exist or observable, wait =0 means no car, wait = 1-10 means the number of cars
        # cross_type (V): int {3, 4}, 3 means 3-direction crossroad, 4 means 4-direction crossroad
        # light (B, V): current traffic light status, int 0-3 or 4-6, corresponding to 4-direction or 3-direction
        B, V, _ = wait.shape
        with torch.no_grad():
            act_values = self.model(wait, cross_type, light) # (B, V, 2)
        action = torch.argmax(act_values, dim = -1) # (B, V), 0 or 1, 0 means keep, 1 means change
        random_index = torch.rand(B, V, device = action.device) < epsilon
        random_action = torch.randint(0, 2, (B, V), device = action.device)
        action = torch.where(random_index, random_action, action)
        return action # (B, V), 0 or 1, 0 means keep, 1 means change
    
    def turn_light(self, cross_type, light, action):
        B, V = light.shape
        cross_type = cross_type.unsqueeze(0).expand(B, -1) # (B, V)
        cross3_index = torch.where(cross_type == 3)
        light[cross3_index] = light[cross3_index] - 4 # 4-6 -> 0-2
        light = light + action # move to next direction
        light = light % cross_type # 4 -> 0 or 3 -> 0
        light[cross3_index] = light[cross3_index] + 4 # 0-2 -> 4-6
        return light # (B, V)
    
    def best_light(self, full_wait):
        # wait (B, V, 7): the waiting number at each node, int [-1, 10] ,V is the number of node, 7 means 7 directions
        # wait = -1 means not exist or observable, wait =0 means no car, wait = 1-10 means the number of cars
        max_values, max_indices = torch.max(full_wait, dim = -1)
        prob = (full_wait == max_values.unsqueeze(-1)).float()
        prob = prob / prob.sum(dim = -1, keepdim = True) # (B, V, 7)
        return prob # (B, V, 7)
    
    def remember(self, state, action, reward, next_state, done):
        # state = (wait, cross_type, light)
        # wait (B, V, 7)
        # cross_type (V)
        # light (B, V)
        # action (B, V)
        # reward (B, V)
        # next_state = (wait, cross_type, light)
        self.memory.append(((state[0].to(self.memory_device),state[1].to(self.memory_device),state[2].to(self.memory_device)), action.to(self.memory_device), reward.to(self.memory_device), (next_state[0].to(self.memory_device),next_state[1].to(self.memory_device),next_state[2].to(self.memory_device)), done))

    def replay(self, optimize = True):
        if len(self.memory) < self.batch_size:
            return torch.tensor(0)
        loss_list = []
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = (state[0].clone().to(self.device),state[1].clone().to(self.device),state[2].clone().to(self.device))
            action = action.clone().to(self.device)
            reward = reward.clone().to(self.device)
            next_state = (next_state[0].clone().to(self.device),next_state[1].clone().to(self.device),next_state[2].clone().to(self.device))
            
            target = reward # (B, V)
            if not done:
                target += self.gamma * torch.max(self.model(next_state[0],next_state[1],next_state[2]))
            target_f = self.model(state[0],state[1],state[2]) # (B, V, 2)
            action = nn.functional.one_hot(action, 2).float()
            target_f = target_f* (1 - action) + target.unsqueeze(-1) * action # (B, V, 2) change the corresponding action to target

            self.optimizer.zero_grad()
            loss = self.criterion(target_f, self.model(state[0],state[1],state[2]))
            if optimize:
                loss.backward()
                self.optimizer.step()
            loss_list.append(loss.item())

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return np.mean(loss_list)


if __name__ == '__main__':
    # test
    agent = DQNAgent()
    wait = torch.randint(-1, 11, (2, 4, 7))
    cross_type = torch.tensor([3, 4, 3, 4])
    light = torch.randint(0, 7, (2, 4))
    action = agent.act(wait, cross_type, light)
    print(action)
    light = agent.turn_light(cross_type, light, action)
    print(light)
    best_light = agent.best_light(wait)
    print(best_light)
    reward = torch.rand(2, 4)
    agent.remember((wait, cross_type, light), action, reward, (wait, cross_type, light), False)
    loss = agent.replay()
    print(loss)
    print(agent.epsilon )