import torch
import math
import time
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from typing import Optional

def get_prefix_mask(T, window_size=4, device='cuda'):
    ones_matrix = torch.ones(T, T, device=device, dtype=torch.bool)
    mask = torch.zeros(T, T, device=device, dtype=torch.bool) | \
            torch.tril(ones_matrix, -window_size) | torch.triu(ones_matrix, 1)
    return mask

def get_adjmask(X, graph_adjmat:torch.Tensor):
    # X: [B, T, N] or [T,N]
    # adjmat: [V, V]
    s = X.shape
    adj_mask = graph_adjmat[X.reshape(-1)].reshape(*s,-1)
    
    return adj_mask

def modulate(x, shift, scale):
    """Scale & shift
    """
    return x * (1 + scale) + shift

class NormalizedEmbedding(nn.Module):
    def __init__(self, 
                 vocab_size, 
                 n_embd,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, n_embd)
        self.n_embd = n_embd

    def forward(self, 
                x,
    ):
        x = self.embedding(x)
        return x/torch.norm(x,dim=-1,keepdim=True)

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, 
                 n_embd, 
                 dropout=0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 2 * n_embd),
            nn.SiLU(),
            nn.Linear(2 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class SelfAttention(nn.Module):
    
    def __init__(self, 
                 n_heads: int, 
                 n_hidden: int, 
                 dropout=0.1, 
                 in_proj_bias=False, 
                 out_proj_bias=False,
    ):
        super().__init__()
        
        self.q_proj = nn.Linear(n_hidden, n_hidden, bias=in_proj_bias)
        self.k_proj = nn.Linear(n_hidden, n_hidden, bias=in_proj_bias)
        self.v_proj = nn.Linear(n_hidden, n_hidden, bias=in_proj_bias)

        self.out_proj = nn.Linear(n_hidden, n_hidden, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = n_hidden // n_heads
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask=None):
        # x: (Traj) (B * T, N, H) agent attention
        # x: (Traj) (B * N, T, H) temporal attention
        # Take temporal attention as an example

        input_shape = x.shape
        batch_size, sequence_length, n_embd = input_shape
        
        interm_shape = (batch_size, -1, self.n_heads, self.d_head)

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # (B * N, T, H) -> (B * N, T, n_heads, d_head) -> (B * N, n_heads, T, d_head)
        q = q.view(interm_shape).transpose(1, 2)
        k = k.view(interm_shape).transpose(1, 2)
        v = v.view(interm_shape).transpose(1, 2)
        
        weight = q @ k.transpose(-1, -2)
        weight /= math.sqrt(self.d_head)
        if mask is not None:
            weight = torch.masked_fill(weight, mask.unsqueeze(1), value=-1e7)
        weight = F.softmax(weight, dim=-1)
        weight = self.dropout(weight)

        # (B, n_heads, T, T) @ (B, n_heads, T, d_head) -> (B, n_heads, T, d_head)
        output = weight @ v
        
        # (B, n_heads, T, d_head) -> (B, T, n_heads, d_head)
        output = output.transpose(1, 2).contiguous()
        
        # (B, T, n_heads, d_head) -> (B, T, H)
        output = output.view(input_shape)
        
        output = self.out_proj(output)
        
        # (B, T * N, H)
        
        return output

class CrossAttention(nn.Module):
    
    def __init__(self, 
                 n_heads: int, 
                 n_hidden: int, 
                 n_embd: int, 
                 dropout=0.1, 
                 in_proj_bias=False, 
                 out_proj_bias=False,
    ):
        super().__init__()
        
        self.q_proj = nn.Linear(n_hidden, n_hidden, bias=in_proj_bias)
        self.k_proj = nn.Linear(n_embd, n_hidden, bias=in_proj_bias)
        self.v_proj = nn.Linear(n_embd, n_hidden, bias=in_proj_bias)

        self.out_proj = nn.Linear(n_hidden, n_hidden, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = n_hidden // n_heads
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj_embed: torch.Tensor):
        # x: (Traj) (B, T * N, H)
        # adj_embed: (Road) (B, V, D)

        input_shape = x.shape
        batch_size, sequence_length, n_embd = input_shape
        
        interm_shape = (batch_size, -1, self.n_heads, self.d_head)

        q = self.q_proj(x)
        k = self.k_proj(adj_embed)
        v = self.v_proj(adj_embed)
        
        # (B, T * N, H) -> (B, T * N, n_heads, d_head) -> (B, n_heads, T * N, d_head)
        q = q.view(interm_shape).transpose(1, 2)
        # (B, V, H) -> (B, V, n_heads, d_head) -> (B, n_heads, V, d_head)
        k = k.view(interm_shape).transpose(1, 2)
        v = v.view(interm_shape).transpose(1, 2)
        
        weight = q @ k.transpose(-1, -2)
        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight, dim=-1)
        weight = self.dropout(weight)

        # (B, n_heads, T * N, V) @ (B, n_heads, V, d_head) -> (B, n_heads, T * N, d_head)
        output = weight @ v
        
        # (B, n_heads, T * N, d_head) -> (B, T * N, n_heads, d_head)
        output = output.transpose(1, 2).contiguous()
        
        # (B, T * N, n_heads, d_head) -> (B, T * N, H)
        output = output.view(input_shape)
        
        output = self.out_proj(output)
        # (B, T * N, H)
        
        return output

class SpatialTemporalCrossBlock(nn.Module):
    """Spatial Temporal Block
    1. Temporal Attention: (B,*T*,N,C) -> (B,*T*,N,C)
    2. Spatial Attention: (B,T,*N*,C) -> (B,T,*N*,C)
    3. Cross Attention: 
    4. Local Feed Forward: (B,T,N,*C*) -> (B,T,N,*C*)
    """

    def __init__(self, 
                 n_hidden, 
                 n_head, 
                 block_size, 
                 norm_position, 
                 n_embd, 
                 dropout=0.1, 
                 flag_ta=True, 
                 flag_sa=False, 
                 flag_ca=False,
                 use_adaLN=False,
                 device='cuda',
                 window_size=4,
    ):
        super().__init__()
        #head_size = n_hidden // n_head
        self.norm_position = norm_position
        #self.ta = MultiHeadAttention(n_head, head_size, n_hidden, block_size, dropout=dropout) if flag_ta else None
        #self.sa = MultiHeadAttention(n_head, head_size, n_hidden, block_size, dropout=dropout) if flag_sa else None
        self.ta = SelfAttention(n_head, n_hidden, dropout=dropout) if flag_ta else None
        self.sa = SelfAttention(n_head, n_hidden, dropout=dropout) if flag_sa else None
        self.ca = CrossAttention(n_head, n_hidden, n_embd, dropout) if flag_ca else None
        if flag_ta and flag_ca:
            print("Temporal, Adj attention")
        self.ffwd = FeedFoward(n_hidden, dropout=dropout)
        if use_adaLN:
            self.ln1 = nn.LayerNorm(n_hidden, elementwise_affine=False, eps=1e-6)
            self.ln2 = nn.LayerNorm(n_hidden, elementwise_affine=False, eps=1e-6)
            self.ln3 = nn.LayerNorm(n_hidden, elementwise_affine=False, eps=1e-6)
        else:
            self.ln1 = nn.LayerNorm(n_hidden)
            self.ln2 = nn.LayerNorm(n_hidden)
            self.ln3 = nn.LayerNorm(n_hidden)
        self.ln_adj = nn.LayerNorm(n_hidden) if flag_ca else None
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(n_hidden, 6 * n_hidden, bias=True)
        ) if use_adaLN else None
        self.ln4 = nn.LayerNorm(n_hidden)
        self.window_size = window_size
        #self.prefix_mask = get_prefix_mask(block_size, window_size, device=device)
        
    def forward(self, x, x_valid, adj_embed, mask=None):
        # Input & Output: (B, T, N, C)
        # x_valid: (B, N)
        # mask: (T, T)
        B, T, N, C = x.shape

        if self.adaLN_modulation is not None:
            # (B, 1, 1, n_hidden)
            shift_mta, scale_mta, gate_mta, shift_ffwd, \
            scale_ffwd, gate_ffwd = self.adaLN_modulation(adj_embed).chunk(6, dim=-1)

        if self.ta is not None:
            #mask = torch.ones((T, T), dtype=torch.bool, device=x.device).triu(1)
            # mask: (1, T, T)
            #mask = mask.unsqueeze(0)
            if self.norm_position == 'prenorm':
                if self.adaLN_modulation is not None:
                    x = x + gate_mta * self.ta(
                                modulate(self.ln1(x), shift_mta, scale_mta).transpose(1,2).reshape(B*N,T,C),
                                mask=mask
                                ).view(B,N,T,C).transpose(1,2)
                else:
                    x = x + self.ta(
                                self.ln1(x.transpose(1,2).reshape(B*N,T,C)),
                                mask=mask
                                ).view(B,N,T,C).transpose(1,2)
            elif self.norm_position == 'postnorm':
                x = self.ln1(x + self.ta(x.transpose(1,2).reshape(B*N,T,C), mask=mask).view(B,N,T,C).transpose(1,2))
            
        if self.sa is not None:
            mask =None
            if self.norm_position == 'prenorm':
                x = x + self.sa(
                            self.ln2(x.view(B*T,N,C)),
                            mask = mask,
                            ).view(B,T,N,C)
            elif self.norm_position == 'postnorm':
                x = self.ln2(x + self.sa(x.view(B*T,N,C),mask=mask).view(B,T,N,C))
        
        if self.ca is not None:
            # cross_att
            if self.norm_position == 'prenorm':
                x = x + self.ca(self.ln3(x).reshape(B, N*T, C), self.ln_adj(adj_embed)).view(B, T, N, C)
            elif self.norm_position == 'postnorm':
                x = self.ln3(x + self.ca(x.reshape(B, N*T, C), adj_embed).view(B, T, N, C))

        
        if self.norm_position == 'prenorm':
            if self.adaLN_modulation is not None:
                x = x + gate_ffwd * self.ffwd(
                            modulate(self.ln4(x), shift_ffwd, scale_ffwd))
            else:
                x = x + self.ffwd(self.ln4(x))
        elif self.norm_position == 'postnorm':
            x = self.ln4(x + self.ffwd(x))
            
        return x

    def cache(self, x, adj_embed, make_cache=False):
        # Input & Output: (B, T, N, C)
        B, T, N, C = x.shape

        if self.adaLN_modulation is not None and make_cache:
            # (B, 1, 1, n_hidden)
            self.shift_mta, self.scale_mta, self.gate_mta, \
            self.shift_ffwd, self.scale_ffwd, self.gate_ffwd = self.adaLN_modulation(adj_embed).chunk(6, dim=-1)

        if self.ta is not None:
            mask = torch.ones((T, T), dtype=torch.bool, device=x.device).triu(1)
            # mask: (1, T, T)
            mask = mask.unsqueeze(0)
            if self.norm_position == 'prenorm':
                if self.adaLN_modulation is not None:
                    x = x + self.gate_mta * self.ta(
                                modulate(self.ln1(x), self.shift_mta, self.scale_mta).transpose(1,2).reshape(B*N,T,C),
                                mask=mask
                                ).view(B,N,T,C).transpose(1,2)
                else:
                    x = x + self.ta(
                                self.ln1(x.transpose(1,2).reshape(B*N,T,C)),
                                mask=mask
                                ).view(B,N,T,C).transpose(1,2)
            elif self.norm_position == 'postnorm':
                x = self.ln1(x + self.ta(x.transpose(1,2).reshape(B*N,T,C), mask=mask).view(B,N,T,C).transpose(1,2))
            
        if self.sa is not None:
            if self.norm_position == 'prenorm':
                x = x + self.sa(
                            self.ln2(x.view(B*T,N,C)),
                            mask=None,
                            ).view(B,T,N,C)
            elif self.norm_position == 'postnorm':
                x = self.ln2(x + self.sa(x.view(B*T,N,C),mask=None).view(B,T,N,C))
        
        if self.ca is not None:
            # cross_att
            if self.norm_position == 'prenorm':
                x = x + self.ca(self.ln3(x).reshape(B, N*T, C), self.ln_adj(adj_embed)).view(B, T, N, C)
            elif self.norm_position == 'postnorm':
                x = self.ln3(x + self.ca(x.reshape(B, N*T, C), adj_embed).view(B, T, N, C))

        
        if self.norm_position == 'prenorm':
            if self.adaLN_modulation is not None:
                x = x + self.gate_ffwd * self.ffwd(
                            modulate(self.ln4(x), self.shift_ffwd, self.scale_ffwd))
            else:
                x = x + self.ffwd(self.ln4(x))
        elif self.norm_position == 'postnorm':
            x = self.ln4(x + self.ffwd(x))
            
        return x

class SpatialTemporalCrossMultiAgentModel(nn.Module):
    '''
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
                                                    '''
    def __init__(self, 
                 vocab_size: int, 
                 n_embd: int, 
                 n_hidden: int, 
                 n_layer: int, 
                 n_head: int, 
                 block_size: int,
                 n_embed_adj: int,
                 window_size: int,
                 dropout=0.1, 
                 use_ne=True, 
                 use_ge=False,
                 use_agent_mask=False, 
                 norm_position='prenorm',
                 device='cuda',
                 postprocess=False,
                 graph_embedding_mode='adaLN',
                 use_adjembed=True,
    ):
        super().__init__()
        if use_ne:
            self.token_embedding_table = NormalizedEmbedding(vocab_size, n_embd)
        else:
            self.token_embedding_table = nn.Embedding(vocab_size, n_embd)

        self.use_ne = use_ne
        #self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.position_embedding_table = nn.Embedding(window_size, n_embd)

        self.in_proj = nn.Sequential(
            nn.Linear(n_embd, n_hidden),
            nn.LayerNorm(n_hidden),
            nn.SiLU(),
        )
        # Mul adj before softmax, block unreachable nodes
        self.postprocess = postprocess
        self.blocks = nn.ModuleList([SpatialTemporalCrossBlock(n_hidden, n_head, 
                                                            block_size, norm_position, 
                                                            n_embd, dropout, 
                                                            flag_sa=False, 
                                                            flag_ta=True,
                                                            flag_ca=True if graph_embedding_mode=='cross' else False,
                                                            use_adaLN=True if graph_embedding_mode=='adaLN' else False,
                                                            device=device,
                                                            window_size=window_size,
                                                            ) for l in range(n_layer)])
        print("Window size: %d" % (window_size))
        self.graph_embedding_mode = graph_embedding_mode
        
        # Generate affine based on graph structure
        if self.graph_embedding_mode=='adaLN':
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(n_hidden, 2 * n_hidden, bias=True)
            )
            self.ln_f = nn.LayerNorm(n_hidden, elementwise_affine=False, eps=1e-6)  # final layer norm
        else:
            self.adaLN_modulation = None
            self.ln_f = nn.LayerNorm(n_hidden)  # final layer norm

        self.lm_head = nn.Linear(n_hidden, vocab_size)
        
        self.condition_proj = nn.Sequential(
            nn.Linear(n_embd*2, n_hidden),
            nn.LayerNorm(n_hidden),
            nn.SiLU(),
            nn.Linear(n_hidden, 2),
        )
        
        # Adj embed from stratch
        if use_adjembed:
            self.adj_embedding_table = nn.Embedding(vocab_size, n_embed_adj)
            self.adj_proj = nn.Sequential(
                nn.LayerNorm(n_embed_adj),
                nn.SiLU(),
                nn.Linear(n_embed_adj, n_hidden),
                nn.LayerNorm(n_hidden),
            )
        elif self.graph_embedding_mode in ['adaLN', 'add', 'cross']:
            self.adj_embedding_table = self.token_embedding_table
            self.adj_proj = nn.Sequential(
                nn.LayerNorm(n_embd),
                nn.SiLU(),
                nn.Linear(n_embd, n_hidden),
                nn.LayerNorm(n_hidden),
            )
        else:
            self.adj_proj = None

        self.device = device
        self.block_size = block_size
        self.window_size = window_size
        self.use_agent_mask = use_agent_mask
        self.sawtooth_mask = self.get_sawtooth_mask(block_size, window_size, device)
        self.initialize_weights(use_adaLN=True if graph_embedding_mode=='adaLN' else False)

    def get_sawtooth_mask(self, T, window_size=4, device='cuda'):
        n = T + window_size - 1
        mask = torch.zeros((n, n), dtype=torch.bool, device=device)
        for i in range((n + window_size-1) // window_size):
            mask[:(i+1)*window_size, i*window_size:(i+1)*window_size] = 1
        mask = mask.tril()
        return mask

    def initialize_weights(self, use_adaLN=False):
        if use_adaLN:
            def init_weights(m):
                if type(m) == nn.Linear:
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                if type(m) == nn.Embedding:
                    nn.init.normal_(m.weight)
            self.apply(init_weights)
        
            # Zero-out adaLN modulation layers in Attention-FFWD blocks:
            for block in self.blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
            
            # Init LayerNorm in in_proj, condition and adj embedding
            if self.adj_proj:
                for layer in self.adj_proj:
                    if isinstance(layer, nn.LayerNorm):
                        nn.init.ones_(layer.weight)
                        nn.init.zeros_(layer.bias)
            for layer in self.condition_proj:
                if isinstance(layer, nn.LayerNorm):
                    nn.init.ones_(layer.weight)
                    nn.init.zeros_(layer.bias)
            for layer in self.in_proj:
                if isinstance(layer, nn.LayerNorm):
                    nn.init.ones_(layer.weight)
                    nn.init.zeros_(layer.bias)
            # Zero-out output layers:
            nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(self.lm_head.weight, 0)
            nn.init.constant_(self.lm_head.bias, 0)
        else:
            def init_weights(m):
                if type(m) == nn.Linear:
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                if type(m) == nn.Embedding:
                    nn.init.normal_(m.weight)
                if type(m) == nn.LayerNorm:
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
            self.apply(init_weights)

    def forward(self, 
                x:torch.Tensor, 
                x_valid:Optional[torch.Tensor]=None, 
                y:Optional[torch.Tensor]=None, 
                condition:Optional[torch.Tensor]=None, 
                adj:Optional[torch.Tensor]=None, 
    ):
        # Input: 
            # x and y: (B, T, N)  
            # x_valid: (B, 1), valid length for each trajectory
            # condition: (B, T, N, 2) 
            # adj: (B, V, V) 
        # Output: (B, T, N, V)
        B, T, N = x.shape

        # adj_indices: (B, V, max_degree+2), adj_values: (B, V, max_degree+2)
        adj_indices, adj_values  = adj
        adj_values = adj_values.unsqueeze(-1)
        if self.adj_proj is not None:
            # adj_embed: sum((B, V, max_degree+2, 2*n_hidden) * (B, V, max_degree+2, 1)) -> (B, V, 2*n_hidden)
            adj_embed = (self.adj_embedding_table(adj_indices) * adj_values).sum(-2)
            # adj_embed: (B, V, 2*n_hidden) -> (B, V , n_hidden)
            adj_embed = self.adj_proj(adj_embed)
            if self.graph_embedding_mode in ['add', 'adaLN']:
                # (B, V , n_hidden) -> (B, 1, 1, n_hidden)
                adj_embed = torch.mean(adj_embed, dim=-2, keepdim=True).unsqueeze(-2)
        else:
            adj_embed = None

            
        # x and y are both (B, T) tensor of integers
        tok_emb = self.token_embedding_table(x) # (B, T, N ,C)
        
        #pos_emb = self.position_embedding_table(torch.arange(T, device=self.device)).view(1, T, 1, -1) # (1, T, 1, C)
        idx = np.random.choice(np.arange(self.window_size))
        mask = self.sawtooth_mask[idx:idx + T, idx:idx + T]
        pos_index = torch.bincount(mask.nonzero().transpose(0, 1)[0]) - 1
        # postional embedding
        pos_emb = self.position_embedding_table(pos_index).view(1, T, 1, -1) # (1, T, 1, C)
        # sawtooth mask
        mask = ~mask
        mask = mask.unsqueeze(0)
        # if x_valid is not None:
        #     pos = torch.arange((T), dtype=torch.float32, device=x.device)[:, None]
        #     # mask: (T, N) -> (1, T, N) -> (B, T, N)
        #     mask_padding = torch.repeat_interleave(pos, N, axis=-1)[None, :, :] >= x_valid[:, None, :]
        #     # mask_padding: (B, T, N) -> (B*N, 1, T)
        #     mask_padding = mask_padding.transpose(1,2).reshape(B*N, -1, T)
        #     # broadcasting: (B*N, T, T)
        #     mask = (mask | mask_padding)
            
        #     mask_mid = torch.repeat_interleave(pos, N, axis=-1)[None, :, :]
        #     mask_mid = ((x_valid - 2)[:, None, :] <= mask_mid) & (mask_mid < x_valid[:, None, :])
        #     mask_mid = mask_mid.transpose(1,2).reshape(B*N, -1, T).repeat(1, T, 1)
        #     mask_mid = ~(mask_mid.tril())
            
        #     #mask = (mask | mask_padding) & mask_mid
        #     mask = mask[2].tolist()
        #     print(x_valid[2])
        #     for i in mask:
        #         print(i)
        #     exit()

        if condition is not None:
            condition_d = condition[:, :, :, 0]  # (B, T, N)
            condition_d_emb = self.token_embedding_table(condition_d.long())  # (B, T, N, C)
            condition_emb = torch.cat((tok_emb, condition_d_emb), dim=-1)  # (B, T, N, 2C)
            condition_score = torch.softmax(self.condition_proj(condition_emb), dim=-1)  # (B, T, N, 2)
            condition_emb = torch.einsum('btnd,btndc->btnc', condition_score,condition_emb.view(B, T, N, 2, -1)) # (B, T, N, C)
        else:
            condition_emb = 0

        if self.graph_embedding_mode=='add':
            x = tok_emb + pos_emb + condition_emb + adj_embed # (B, T, N, C)
        else:
            x = tok_emb + pos_emb + condition_emb # (B, T, N, C)
        x = self.in_proj(x)
        
        for block in self.blocks:
            x = block(x, x_valid, adj_embed, mask)
        
        if self.adaLN_modulation is not None:
            shift, scale = self.adaLN_modulation(adj_embed).chunk(2, dim=-1)    
            x = modulate(self.ln_f(x), shift, scale)    
        else:
            x = self.ln_f(x) # (B, T, N, C)
        logits = self.lm_head(x) # (B, T, N, V)
        
        if y is None:
            loss = None
        else:
            B, T, N, V = logits.shape
            logits_ = logits.view(B*T*N, V)
            y = y.view(B*T*N)
            loss = F.cross_entropy(logits_, y)
        
        return logits, loss

    def cache(self,
              position_index,
              x:torch.Tensor, 
              condition:Optional[torch.Tensor]=None, 
              adj_embed:Optional[torch.Tensor]=None, 
              shift_proj:Optional[torch.Tensor]=None, 
              scale_proj:Optional[torch.Tensor]=None, 
              make_cache:bool=False,
    ):
        # cache adaLN
        # Input: 
            # x and y: (B, T, N)  
            # x_valid: (B, 1), valid length for each trajectory
            # condition: (B, T, N, 2) 
            # adj_embed: (B, 1, H) 
        # Output: (B, T, N, V)

        B, T, N = x.shape
            
        # x and y are both (B, T) tensor of integers
        tok_emb = self.token_embedding_table(x) # (B, T, N ,C)
        # (1, T, 1, C)
        pos_emb = self.position_embedding_table(torch.arange(start=position_index, \
                            end=position_index+T, device=self.device)).view(1, T, 1, -1)
        
        if condition is not None:
            condition_d = condition[:, :, :, 0]  # (B, T, N)
            condition_d_emb = self.token_embedding_table(condition_d.long())  # (B, T, N, C)
            condition_emb = torch.cat((tok_emb, condition_d_emb), dim=-1)  # (B, T, N, 2C)
            condition_score = torch.softmax(self.condition_proj(condition_emb), dim=-1)  # (B, T, N, 2)
            condition_emb = torch.einsum('btnd,btndc->btnc', condition_score,condition_emb.view(B, T, N, 2, -1))  # (B, T, N, C)
        else:
            condition_emb = 0

        if self.graph_embedding_mode=='add':
            x = tok_emb + pos_emb + condition_emb + adj_embed # (B, T, N, C)
        else:
            x = tok_emb + pos_emb + condition_emb # (B, T, N, C)
        x = self.in_proj(x) # (B, T, N, H)
        
        for block in self.blocks:
            x = block.cache(x, adj_embed, make_cache=make_cache)

        if self.adaLN_modulation is not None:
            x = modulate(self.ln_f(x), shift_proj, scale_proj)    
        else:
            x = self.ln_f(x) # (B, T, N, C)
        logits = self.lm_head(x) # (B, T, N, V)

        return logits
    
    def decode_strategy(self, 
                        logits:torch.Tensor, 
                        agent_mask:Optional[torch.Tensor]=None, 
                        sampling_strategy="top_1",
                        **kwargs,
    ):
        # logits: (M, V)
        # agent_mask: (M)
        M,V = logits.shape
        if not self.use_agent_mask:
            agent_mask = None
        
        temp = kwargs.pop("tempperature",1.0)
        # apply softmax to get probabilities
        probs = F.softmax(logits/temp, dim=-1) # (B*N, V)


        # sample from the distribution
        if sampling_strategy == "random":
            idx_next = torch.multinomial(probs, num_samples=1) # (B*N, 1)
        
        elif sampling_strategy == "top_1":
            idx_next = torch.argmax(probs,dim=-1)
            
        elif sampling_strategy == "top_k":
            top_k = kwargs.pop("top_k",10)
            topk_idx = torch.topk(probs,top_k,dim=-1)
            idx_next__ = torch.multinomial(topk_idx.values, num_samples=1) # (B*N, 1)
            idx_next = topk_idx.indices.gather(-1,idx_next__).view(M,1)
            
        elif sampling_strategy == "top_p":
            top_p = kwargs.pop("top_p",0.9)
            
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = (cumulative_probs > top_p)
            sorted_indices_to_remove = torch.roll(sorted_indices_to_remove, 1, 1)
            sorted_indices_to_remove[:, 0] = False
            
            unsorted_indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
            probs[unsorted_indices_to_remove] = 0
            probs /= probs.sum(dim=-1, keepdim=True)
            
            idx_next = torch.multinomial(probs, num_samples=1) # (B*N, 1)
            
        elif sampling_strategy =="greedy":
            idx_next = torch.argmax(probs,dim=-1).view(M,1)
        else:
            raise NotImplementedError
        
        if agent_mask is not None:
            idx_next = torch.masked_fill(idx_next,agent_mask.view(-1,1)==0,0)
        
        return idx_next, probs

    def generate(self, 
                 idx, 
                 condition=None, 
                 adj=None, 
                 max_new_tokens=100,
                 agent_mask=None,
                 sampling_strategy="random", 
                 **kwargs,
    ): 
        # Input: 
            # idx: (B, 1, N)  
            # condition: (B, 1, N, 1) 
        B, T, N = idx.shape

        # adj_indices: (B, V, max_degree+2), adj_values: (B, V, max_degree+2)
        adj_indices, adj_values  = adj
        adj_values = adj_values.unsqueeze(-1)
        condition = condition.repeat(1, self.window_size, 1, 1)
        #condition = condition + torch.zeros((*idx.shape, 1)).to(self.device).long()

        if not self.use_agent_mask:
            agent_mask = None
        #start_time = time.time()
        if self.adj_proj is not None:
            # adj: sum((B, V, max_degree+2, 2*n_hidden) * (B, V, max_degree+2, 1)) -> (B, V, 2*n_hidden)
            adj_embed = (self.adj_embedding_table(adj_indices) * adj_values).sum(-2)
            # adj: (B, V, 2*n_hidden) -> (B, V , n_hidden)
            adj_embed = self.adj_proj(adj_embed)
            if self.graph_embedding_mode in ['add', 'adaLN']:
                # (B, V , n_hidden) -> (B, 1, 1, n_hidden)
                adj_embed = torch.mean(adj_embed, dim=-2, keepdim=True).unsqueeze(-2)
        else:
            adj_embed = None
        #print("Adj embedding time: %.3f" % (time.time() - start_time))

        if self.adaLN_modulation is not None:
            shift_proj, scale_proj = self.adaLN_modulation(adj_embed).chunk(2, dim=-1)
        else:
            shift_proj, scale_proj = None, None

        for i in range(max_new_tokens):
                   
            logits = self.cache(
                                #position_index=0 if i < self.window_size else i - self.window_size + 1,
                                position_index=0,
                                x=idx[:, -self.window_size:, :],
                                condition=condition[:, :i+1 if i < self.window_size else self.window_size, :, :], 
                                adj_embed=adj_embed, 
                                shift_proj=shift_proj, scale_proj=scale_proj,
                                make_cache=True if i==0 else False
                                )
            
            # focus only on the last time step
            logits = logits[:, -1, :, :].view(B*N, -1)  # becomes (B*N, V)
            
            idx_next, probs = self.decode_strategy(logits, None, sampling_strategy, **kwargs)
            #if torch.all(idx_next == 0):
                # (B, max_new_tokens, N)
            #    idx = torch.cat((idx, torch.zeros((B, max_new_tokens-i, N), device=idx.device, dtype=idx.dtype)), dim=1)
            #    break
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next.view(B, 1, N)), dim=1)  # (B, T+1, N)

        return idx.view(101).cpu().tolist()

    def log_prob_traj(self, 
                      x:torch.Tensor, 
                      y:torch.Tensor, 
                      x_valid:torch.Tensor,
                      condition:torch.Tensor, 
                      agent_mask:Optional[torch.Tensor]=None, 
                      adj=None, 
    ):
        # x: (B, T, N), condition: (B, T, N, 2)
        B, T, N = x.shape
        
        if not self.use_agent_mask:
            agent_mask = None
        
        logits, _ = self(x, x_valid=x_valid, condition=condition,
                        adj=adj,
                        agent_mask=agent_mask)
                
        log_prob = F.log_softmax(logits, dim=-1)
        V = logits.shape[-1]

        y_hat = torch.argmax(log_prob, dim=-1)

        log_prob_targ = torch.gather(log_prob, 3, y.unsqueeze(-1)).squeeze(-1)
        # (B, T, N, V) -> (B, T, N)
        
        log_prob_sum = log_prob_targ.sum(dim=(-1, -2))
            
        return log_prob_sum, log_prob_targ, y_hat
        ...

def get_1d_sincos_geo_embed(d_cross, pos):
    """
    d_cross: output dimension for each position
    pos: a list of positions to be encoded: size (V,)
    out: (M, D)
    """
    assert d_cross % 2 == 0
    omega = np.arange(d_cross // 2, dtype=np.float64)
    omega /= d_cross / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (V,)
    out = np.einsum('v,d->vd', pos, omega)  # (V, D/2), outer product

    emb_sin = np.sin(out)  # (V, D/2)
    emb_cos = np.cos(out)  # (V, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=-1)  # (V, D)
    return emb