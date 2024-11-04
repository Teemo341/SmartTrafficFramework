from math import sin
import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional,Tuple,Union
from data_type_task1 import Batch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import distributions as pyd

# class RoPE(nn.Module):
#     def __init__(self, base=10000, d=128):
#         """
#         Requires the input to be (seqlen, bs, d)
#         """
#         super().__init__()
#         assert d%2 == 0, "RoPE only works with even dimensions"
#         
#         self.base = base
#         self.d = d
#         self.cos_cached = None
#         self.sin_cached = None
# 
#     def build_cache(self, x:torch.Tensor):
#         theta = 1./(self.base**(torch.arange(0, self.d, 2).float()/self.d)).to(x.device)
#         seq_idx = torch.arange(x.shape[0], device=x.device).float()
#         idx_theta1 = torch.einsum('i,j->ij', seq_idx, theta) #(seqlen, d//2)
#         idx_theta2 = torch.cat([idx_theta1, idx_theta1], dim=1) # (seqlen, d)
#         self.cos_cached = idx_theta2.cos()[:,None,:]
#         self.sin_cached = idx_theta2.sin()[:,None,:]
#     def _neg_half(self, x: torch.Tensor):
#         d_2 = self.d // 2
#         return torch.cat([-x[:, :, d_2:], x[:, :, :d_2]], dim=-1)
#     
#     def forward(self, x:torch.Tensor):
#         if self.cos_cached is None or self.sin_cached is None:
#             self.build_cache(x)
#         if self.cos_cached.shape[0]!=x.shape[0]:
#             self.build_cache(x)
#         negHalfX = self._neg_half(x)
#         x_rope = (x*self.cos_cached + negHalfX*self.sin_cached)
#         return x_rope


class MulSin(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)*x

class SigmoidGaussian(pyd.Distribution):
    r"""
        This class is used to sample from a sigmoid-squashed normal distribution.
        Mapping  $R\times R+ -> \Delta[0,1]$
        Or equivalently, $R -> [0,1]$
    """
    
    def __init__(self, loc:torch.Tensor, log_scale:torch.Tensor):
        self.loc = loc
        self.scale = log_scale.clamp(self._log_sigma_min,self._log_sigma_max  ).exp()
        self.normal = pyd.normal.Normal(loc, self.scale)
        
    @classmethod
    def arc_sigmoid(cls,x):
        # There is always contain 0 or 1 in the input
        # We MUST clamp the input to avoid NaN
        x = torch.clamp(x,cls._eps,1-cls._eps)
        return torch.log(x/(1-x))

    def sample(self, sample_shape=torch.Size()):
        # Gradients will and should *not* pass through this operation.
        z = self.normal.sample(sample_shape=sample_shape).detach()
        return torch.sigmoid(z)

    def rsample(self, sample_shape=torch.Size()):
        # Gradients will and should pass through this operation.
        z = self.normal.rsample(sample_shape=sample_shape)
        return torch.sigmoid(z)

    def log_prob(self, action:torch.Tensor):
        # action: [0,1] -> R
        pre_sigmoid_value = self.arc_sigmoid(action)
        # assert torch.sigmoid(pre_sigmoid_value).allclose(action)
        
        
        logp_pi = self.normal.log_prob(pre_sigmoid_value)#.sum(dim=-1)
        logp_pi += (pre_sigmoid_value + 2*F.softplus(-pre_sigmoid_value)) #.sum(dim=-1)

        return logp_pi

    @classmethod
    def log_prob_(cls, x:torch.Tensor, mu:torch.Tensor, log_sigma:torch.Tensor):
        # x: [S, B], mu: [B], sigma: [B]
        # log_prob: [S, B]
        
        # It's differentiable version of log_prob
        
        
        log_sigma = log_sigma.clamp(cls._log_sigma_min,cls._log_sigma_max )
        pre_sigmoid_value = cls.arc_sigmoid(x)
        
        log_pre = -((pre_sigmoid_value - mu) ** 2) / (2 * torch.exp(log_sigma)**2)
        log_p = log_pre - log_sigma + cls._cons_pi2
        
        log_p += pre_sigmoid_value + 2*F.softplus(-pre_sigmoid_value)
        
        # assert not torch.isnan(log_pre).any(), "log_pre contains NaN" 
        # assert not torch.isnan(log_p).any(), "log_p contains NaN"
        
        return log_p
        
    _cons_pi2 = - 0.5 * np.log(2 * np.pi)
    _eps = 1e-7
    _log_sigma_max = 10
    _log_sigma_min = -2

class NormalizedEmbedding(nn.Module):
    def __init__(self, vocab_size, n_embd):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, n_embd)
        self.n_embd = n_embd

    def forward(self, x):
        x = self.embedding(x)
        return x/torch.norm(x,dim=-1,keepdim=True)

def _generate_sliding_window_mask(seq_len:int,win:int=5)->torch.Tensor:
    """
    Create a sliding window attention mask.
    
    Parameters:
    seq_len (int): Length of the sequence.
    window_size (int): Size of the sliding window.
    
    Returns:
    torch.Tensor: Attention mask of shape (seq_len, seq_len).
    """
    mask = torch.ones(seq_len, seq_len, dtype=torch.bool)
    
    for i in range(seq_len):
        start = max(0, i - win + 1)
        end = min(seq_len, i + 1)
        mask[i, start:end] = 0
        
    return mask

def _generate_alibi_pe(num_heads:int, seq_len:int)->torch.Tensor:
        #return: (H,T,T)
        
        # _mask = torch.zeros(num_heads,seq_len,seq_len,dtype=torch.float32)
        
        dgrid = torch.arange(seq_len,dtype=torch.float32).view(1,-1).expand(seq_len,seq_len)-torch.arange(seq_len,dtype=torch.float32).view(-1,1).expand(seq_len,seq_len)
        slopes = torch.pow(2, -8/num_heads*(torch.arange(num_heads,dtype=torch.float32)+1)) 
        
        print(f"{dgrid=}")
        print(f"{slopes=}")
        
        _mask = slopes.view(-1,1,1)*dgrid.view(1,seq_len,seq_len)
        _mask = torch.masked_fill(_mask, dgrid>0, -torch.inf)
        
        return _mask
class MultiHeadAttention(nn.Module):


    def __init__(self, num_heads:int, 
                    n_embd:int,
                    block_size:int, 
                    sliding_win_size:int=-1, 
                    dropout:float=0.0,
                    use_pe:str="abs",
                    ):
        super().__init__()
        # implement with nn.MultiheadAttention, but the same Interface
        self.mha = nn.MultiheadAttention(n_embd,num_heads,dropout=dropout,batch_first=True,bias=False)
        # self._causal_mask = nn.Transformer.generate_square_subsequent_mask(block_size)
        if sliding_win_size > 0:
            raise NotImplementedError
            _causal_mask = _generate_sliding_window_mask(block_size,win=sliding_win_size)
            # It has some problem!!!
            # See `notes.md`, 24/07/18
        elif use_pe=="alibi":
            _causal_mask = _generate_alibi_pe(num_heads,block_size)
        else:
            _causal_mask= nn.Transformer.generate_square_subsequent_mask(block_size)
            
            
        # if use_pe=="alibi":
        
        self.use_pe = use_pe
        self.register_buffer('_causal_mask',_causal_mask)
            
        
        
    def forward(self, x, mask=None):
        # x: (B,T,H)
        if mask is None:
            if self.use_pe=="alibi":
                mask = self._causal_mask[...,:x.size(1),:x.size(1)].tile(x.size(0),1,1)
            else:
                mask = self._causal_mask[:x.size(1),:x.size(1)]
            is_causal = True
        else:
            is_causal = False
        # print(f"DEBUG: {(mask,is_causal)=}")
        # breakpoint()
        return self.mha(x,x,x,
                        attn_mask=mask,
                        is_causal = is_causal,
                        need_weights=False)[0]


class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 2 * n_embd),
            nn.SiLU(),
            nn.Linear(2 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class SpatialTemporalBlock(nn.Module):
    """Spatial Temporal Block
    1. Temporal Attention: (B,*T*,N,C) -> (B,*T*,N,C)
    2. Spatial Attention: (B,T,*N*,C) -> (B,T,*N*,C)
    3. Local Feed Forward: (B,T,N,*C*) -> (B,T,N,*C*)
    """

    def __init__(self, 
                n_embd, n_head,block_size,
                norm_position, 
                sliding_win_size=-1, dropout=0.1,
                flag_ta=True,flag_sa=False,
                use_pe:str="abs",
                ):
        super().__init__()
        self.norm_position = norm_position
        self.ta = MultiHeadAttention(n_head, n_embd,block_size, sliding_win_size, dropout=dropout,use_pe=use_pe) if flag_ta else None
        self.sa = MultiHeadAttention(n_head, n_embd,block_size, -1, dropout=dropout) if flag_sa else None
        self.ffwd = FeedFoward(n_embd, dropout=dropout)
        self.ln1 = nn.LayerNorm(n_embd) if flag_ta else None
        self.ln2 = nn.LayerNorm(n_embd) if flag_sa else None
        self.ln3 = nn.LayerNorm(n_embd)
        
        

    def forward(self, x, cross_agent_mask = None):
        # Input & Output: (B,T,N,C)
        # cross_agent_mask: (B,T,N,N), bool, 1 for available, 0 for unavailable
        B,T,N,C = x.shape
        if self.ta is not None:
            if self.norm_position == 'prenorm':
                x = x + self.ta(
                            self.ln1(x.transpose(1,2).reshape(B*N,T,C)) #type:ignore
                            ).view(B,N,T,C).transpose(1,2)
            elif self.norm_position == 'postnorm':
                x = self.ln1(x + self.ta(x.transpose(1,2).reshape(B*N,T,C)).view(B,N,T,C).transpose(1,2))#type:ignore
            
        if self.sa is not None:
            if cross_agent_mask is not None:
                raise NotImplementedError
                mask = cross_agent_mask
                # need to reconsider the shape of mask
                # since we replace the MultiHeadAttention implementation with nn.MultiheadAttention
            else:
                mask = torch.ones(N,N,dtype=torch.bool,device=x.device) 
            if self.norm_position == 'prenorm':
                x = x + self.sa(
                            self.ln2(x.view(B*T,N,C)),#type:ignore
                            mask = torch.logical_not(mask),
                            ).view(B,T,N,C)
            elif self.norm_position == 'postnorm':
                x = self.ln2(x + self.sa(x.view(B*T,N,C),mask=torch.logical_not(mask)).view(B,T,N,C))#type:ignore
        
        
        if self.norm_position == 'prenorm':
            x = x + self.ffwd(self.ln3(x))
        elif self.norm_position == 'postnorm':
            x = self.ln3(x + self.ffwd(x))
            
        # assert not torch.any(torch.isnan(x)), "x contains NaN"
        return x

class EncoderHead(nn.Module):
    def __init__(self, 
                vocab_size:int,
                n_embd:int,
                n_hidden:int,
                block_size:int,
                use_ne:bool,
                use_len_ratio:bool,
                use_agent_mask:bool,
                ):
        
        super().__init__()
        if use_ne:
            self.token_embedding_table = NormalizedEmbedding(vocab_size, n_embd)
        else:
            self.token_embedding_table = nn.Embedding(vocab_size, n_embd)

        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.in_proj = nn.Sequential(
            nn.Linear(n_embd, n_hidden),
            nn.LayerNorm(n_hidden),
            nn.SiLU(),
        )
        self.condition_proj = nn.Sequential(
            nn.Linear(n_embd*3, n_hidden),
            nn.LayerNorm(n_hidden),
            nn.SiLU(),
            nn.Linear(n_hidden, 3),
        )
        # if use_len_ratio:
        #     # self.ratio_proj = nn.Linear(1,n_embd)
        #     self.ratio_proj = nn.Sequential(
        #         nn.Linear(1,n_embd),
        #         nn.LayerNorm(n_embd),
        #         nn.SiLU(), 
        #     )
        
        self.use_len_ratio = use_len_ratio
        self.use_agent_mask = use_agent_mask
        
        
    def forward(self, batch):
        B, T, N = batch['traj'].shape
        
        
        idx = batch['traj']
        condition = batch['cond']
        
        if 'adjmask' in batch.keys():
            raise NotImplementedError

        if not self.use_agent_mask or 'reagent_mask' not in batch.keys():
            agent_mask = None
            cross_agent_mask = None
        else:
            agent_mask = batch['reagent_mask']
            cross_agent_mask = (~torch.logical_and(agent_mask.unsqueeze(-1), agent_mask.unsqueeze(-2)).view(B*T,N,N))*(-1e9)
            # Shape: (B*T,N,N) <- (B,T,N,N), <- (B,T,N,1) @ (B,T,1,N)
            
        if self.use_len_ratio :
            assert 'ratio' in batch.keys(), "ratio is required"
            ratio:torch.Tensor = batch['ratio'].unsqueeze(-1) # (B,T,N,1)
            ratio_emb = self.ratio_proj(ratio) # (B,T,N,C)
        else:
            ratio_emb = 1
        
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)*ratio_emb # (B,T,N,C)
        
        pos_emb = self.position_embedding_table(torch.arange(T, device=tok_emb.device)).view(1,T,1,-1) # (1,T,1,C)
        
        
        if condition is not None:
            # broadcastTensor = torch.zeros((B,T,N,2)).to(self.device).long()
            # condition = condition + broadcastTensor
            condition_s = condition[:,:,:,0] # (B,T,N)
            condition_e = condition[:,:,:,1] # (B,T,N)
            condition_s_emb = self.token_embedding_table(condition_s) # (B,T,N,C)
            condition_e_emb = self.token_embedding_table(condition_e) # (B,T,N,C)
            condition_emb = torch.cat((tok_emb,condition_s_emb,condition_e_emb),dim=-1) # (B,T,N,3C)
            condition_score = torch.softmax(self.condition_proj(condition_emb),dim=-1) # (B,T,N,3)
            condition_emb = torch.einsum('btnd,btndc->btnc',condition_score,condition_emb.view(B,T,N,3,-1)) # (B,T,N,C)
        else:
            condition_emb = 0
        
        x = tok_emb + pos_emb + condition_emb # (B,T,N,C)
        x = self.in_proj(x)
        
        return x, agent_mask, cross_agent_mask

class EncoderHeadV2(nn.Module):
    def __init__(self, 
                vocab_size:int,
                n_embd:int,
                n_hidden:int,
                block_size:int,
                use_ne:bool,
                use_len_ratio:bool,
                use_agent_mask:bool,
                ):
        
        super().__init__()
        if use_ne:
            self.token_embedding_table = NormalizedEmbedding(vocab_size, n_embd)
        else:
            self.token_embedding_table = nn.Embedding(vocab_size, n_embd)

        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        
        
        if use_len_ratio:
            # self.ratio_proj = nn.Linear(1,n_embd)
            self.ratio_proj = nn.Sequential(
                nn.Linear(1,n_embd),
                nn.LayerNorm(n_embd),
                nn.SiLU(),
            )
        self.tok_proj = nn.Sequential(
            nn.Linear(n_embd*2,n_hidden*4),
            nn.LayerNorm(n_hidden*4),
            nn.SiLU(),
            nn.Linear(n_hidden*4,n_embd),
            nn.LayerNorm(n_embd),
            nn.SiLU(),
        )
        self.condition_proj = nn.Sequential(
            nn.Linear(n_embd*3, n_hidden*4),
            nn.LayerNorm(n_hidden*4),
            nn.SiLU(),
            nn.Linear(n_hidden*4, n_hidden*4),
            nn.LayerNorm(n_hidden*4),
            nn.SiLU(),
            nn.Linear(n_hidden*4, n_embd*3),
            nn.LayerNorm(n_embd*3),
            nn.SiLU(),
            
        )
        
        self.in_proj = nn.Sequential(
            nn.Linear(n_embd, n_hidden*2),
            nn.LayerNorm(n_hidden*2),
            nn.SiLU(),
            nn.Linear(n_hidden*2, n_hidden),
            nn.LayerNorm(n_hidden),
            nn.SiLU(),
        )
        
        self.use_len_ratio = use_len_ratio
        self.use_agent_mask = use_agent_mask
        
        
    def forward(self, batch):
        B, T, N = batch['traj'].shape
        
        
        idx = batch['traj']
        condition = batch['cond']
        
        # if 'adjmask' in batch.keys():
        #     raise NotImplementedError

        if not self.use_agent_mask or 'reagent_mask' not in batch.keys():
            agent_mask = None
            cross_agent_mask = None
        else:
            agent_mask = batch['reagent_mask']
            cross_agent_mask = (~torch.logical_and(agent_mask.unsqueeze(-1), agent_mask.unsqueeze(-2)).view(B*T,N,N))*(-1e9)
            # Shape: (B*T,N,N) <- (B,T,N,N), <- (B,T,N,1) @ (B,T,1,N)
            
        
        # idx and targets are both (B,T) tensor of integers
        # tok_emb = self.token_embedding_table(idx)*ratio_emb # (B,T,N,C)
        tok_emb = self.token_embedding_table(idx) # (B,T,N,C)
        if self.use_len_ratio :
            assert 'ratio' in batch.keys(), "ratio is required"
            ratio:torch.Tensor = batch['ratio'].unsqueeze(-1) # (B,T,N,1)
            ratio_emb = self.ratio_proj(ratio) # (B,T,N,C)
        else:
            ratio_emb = torch.ones_like(tok_emb)
        tok_emb = self.tok_proj(torch.cat((tok_emb,ratio_emb),dim=-1))
        
        pos_emb = self.position_embedding_table(torch.arange(T, device=tok_emb.device)).view(1,T,1,-1) # (1,T,1,C)
        
        
        if condition is not None:
            # broadcastTensor = torch.zeros((B,T,N,2)).to(self.device).long()
            # condition = condition + broadcastTensor
            
            
            # condition_s = condition[:,:,:,0] # (B,T,N)
            # condition_e = condition[:,:,:,1] # (B,T,N)
            # condition_s_emb = self.token_embedding_table(condition_s) # (B,T,N,C)
            # condition_e_emb = self.token_embedding_table(condition_e) # (B,T,N,C)
            condition_emb = torch.cat((tok_emb, self.token_embedding_table(condition).view(B,T,N,-1)),dim=-1) # (B,T,N,3C)
            # condition_emb = torch.cat((tok_emb,condition_s_emb,condition_e_emb),dim=-1) # (B,T,N,3C)
            condition_emb:torch.Tensor = self.condition_proj(condition_emb) # (B,T,N,3C)
            condition_score = torch.softmax( #type:ignore
                    torch.norm(condition_emb.view(B,T,N,3,-1),dim=-1),#type:ignore
                    dim=-1)
            # condition_score = torch.softmax(self.condition_proj(condition_emb),dim=-1) # (B,T,N,3)
            condition_emb = torch.einsum('btnd,btndc->btnc',condition_score,condition_emb.view(B,T,N,3,-1)) # (B,T,N,C)
        else:
            condition_emb = 0 #type:ignore
        
        x = tok_emb + pos_emb + condition_emb # (B,T,N,C)
        x = self.in_proj(x)
        
        return x, agent_mask, cross_agent_mask
        
class EncoderHeadV3(nn.Module):
    def __init__(self, 
                vocab_size:int,
                n_embd:int,
                n_hidden:int,
                block_size:int,
                use_ne:bool,
                use_len_ratio:bool,
                use_agent_mask:bool,
                ):
        
        super().__init__()
        if use_ne:
            self.token_embedding_table = NormalizedEmbedding(vocab_size, n_embd)
        else:
            self.token_embedding_table = nn.Embedding(vocab_size, n_embd)

        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        
        
        if use_len_ratio:
            # self.ratio_proj = nn.Linear(1,n_embd)
            self.ratio_proj = nn.Sequential(
                nn.Linear(1,n_embd),
                MulSin(),
                nn.Linear(n_embd,n_embd),
                nn.LayerNorm(n_embd),
                nn.SiLU(),
            )
        self.tok_proj = nn.Sequential(
            nn.Linear(n_embd*2,n_hidden*2),
            nn.LayerNorm(n_hidden*2),
            nn.SiLU(),
            nn.Linear(n_hidden*2,n_embd),
            nn.LayerNorm(n_embd),
            nn.SiLU(),
        )
        self.condition_proj = nn.Sequential(
            nn.Linear(n_embd*3, n_hidden*4),
            nn.LayerNorm(n_hidden*4),
            nn.SiLU(),
            nn.Linear(n_hidden*4, n_hidden*4),
            nn.LayerNorm(n_hidden*4),
            nn.SiLU(),
            nn.Linear(n_hidden*4, n_embd*3),
            nn.LayerNorm(n_embd*3),
            nn.SiLU(),
            
        )
        
        self.in_proj = nn.Sequential(
            nn.Linear(n_embd, n_hidden*2),
            nn.LayerNorm(n_hidden*2),
            nn.SiLU(),
            nn.Linear(n_hidden*2, n_hidden),
            nn.LayerNorm(n_hidden),
            nn.SiLU(),
        )
        
        self.use_len_ratio = use_len_ratio
        self.use_agent_mask = use_agent_mask
        
        
    def forward(self, batch:Batch):
        B, T, N = batch.traj.shape
        
        
        idx = batch['traj']
        condition = batch['cond']
        
        if 'adjmask' in batch.keys():
            raise NotImplementedError

        if not self.use_agent_mask or 'reagent_mask' not in batch.keys():
            agent_mask = None
            cross_agent_mask = None
        else:
            agent_mask = batch['reagent_mask']
            cross_agent_mask = (~torch.logical_and(agent_mask.unsqueeze(-1), agent_mask.unsqueeze(-2)).view(B*T,N,N))*(-1e9)
            # Shape: (B*T,N,N) <- (B,T,N,N), <- (B,T,N,1) @ (B,T,1,N)
            
        
        # idx and targets are both (B,T) tensor of integers
        # tok_emb = self.token_embedding_table(idx)*ratio_emb # (B,T,N,C)
        tok_emb = self.token_embedding_table(idx) # (B,T,N,C)
        if self.use_len_ratio :
            assert 'ratio' in batch.keys(), "ratio is required"
            ratio:torch.Tensor = batch['ratio'].unsqueeze(-1) # (B,T,N,1)
            ratio_emb = self.ratio_proj(ratio) # (B,T,N,C)
        else:
            ratio_emb = torch.ones_like(tok_emb)
        tok_emb = self.tok_proj(torch.cat((tok_emb,ratio_emb),dim=-1))
        
        pos_emb = self.position_embedding_table(torch.arange(T, device=tok_emb.device)).view(1,T,1,-1) # (1,T,1,C)
        
        
        if condition is not None:
            # broadcastTensor = torch.zeros((B,T,N,2)).to(self.device).long()
            # condition = condition + broadcastTensor
            
            
            # condition_s = condition[:,:,:,0] # (B,T,N)
            # condition_e = condition[:,:,:,1] # (B,T,N)
            # condition_s_emb = self.token_embedding_table(condition_s) # (B,T,N,C)
            # condition_e_emb = self.token_embedding_table(condition_e) # (B,T,N,C)
            condition_emb = torch.cat((tok_emb, self.token_embedding_table(condition).view(B,T,N,-1)),dim=-1) # (B,T,N,3C)
            # condition_emb = torch.cat((tok_emb,condition_s_emb,condition_e_emb),dim=-1) # (B,T,N,3C)
            condition_emb:torch.Tensor = self.condition_proj(condition_emb) # (B,T,N,3C)
            condition_score = torch.softmax( #type:ignore
                    torch.norm(condition_emb.view(B,T,N,3,-1),dim=-1),#type:ignore
                    dim=-1)
            # condition_score = torch.softmax(self.condition_proj(condition_emb),dim=-1) # (B,T,N,3)
            condition_emb = torch.einsum('btnd,btndc->btnc',condition_score,condition_emb.view(B,T,N,3,-1)) # (B,T,N,C)
        else:
            condition_emb = 0 #type:ignore
        
        x = tok_emb + pos_emb + condition_emb # (B,T,N,C)
        x = self.in_proj(x)
        
        return x, agent_mask, cross_agent_mask
        
class EncoderHeadV4(nn.Module):
    # Diff to V3: 
    #   - remove start point condition; 
    #   - replace MulSin with SiLU+LayerNorm
    #   - support use_pe == "alibi"
    def __init__(self, 
                vocab_size:int,
                n_embd:int,
                n_hidden:int,
                block_size:int,
                use_ne:bool,
                use_len_ratio:bool,
                use_agent_mask:bool,
                use_pe:str,
                ):
        
        super().__init__()
        if use_ne:
            self.token_embedding_table = NormalizedEmbedding(vocab_size, n_embd)
        else:
            self.token_embedding_table = nn.Embedding(vocab_size, n_embd)

        if use_pe=="abs":
            self.position_embedding_table = nn.Embedding(block_size, n_embd)
        
        
        if use_len_ratio:
            # self.ratio_proj = nn.Linear(1,n_embd)
            self.ratio_proj = nn.Sequential(
                nn.Linear(1,n_embd),
                nn.SiLU(),
                nn.LayerNorm(n_embd),
                nn.Linear(n_embd,n_embd),
                nn.LayerNorm(n_embd),
                nn.SiLU(),
            )
        self.tok_proj = nn.Sequential(
            nn.Linear(n_embd*2,n_hidden*2),
            nn.LayerNorm(n_hidden*2),
            nn.SiLU(),
            nn.Linear(n_hidden*2,n_embd),
            nn.LayerNorm(n_embd),
            nn.SiLU(),
        )
        self.condition_proj = nn.Sequential(
            nn.Linear(n_embd*2, n_hidden*4),
            nn.LayerNorm(n_hidden*4),
            nn.SiLU(),
            nn.Linear(n_hidden*4, n_hidden*4),
            nn.LayerNorm(n_hidden*4),
            nn.SiLU(),
            nn.Linear(n_hidden*4, n_embd*2),
            nn.LayerNorm(n_embd*2),
            nn.SiLU(),
            
        )
        
        self.in_proj = nn.Sequential(
            nn.Linear(n_embd, n_hidden*2),
            nn.LayerNorm(n_hidden*2),
            nn.SiLU(),
            nn.Linear(n_hidden*2, n_hidden),
            nn.LayerNorm(n_hidden),
            nn.SiLU(),
        )
        
        self.use_pe = use_pe
        self.use_len_ratio = use_len_ratio
        self.use_agent_mask = use_agent_mask
        
        
    def forward(self, batch:Batch):
        B, T, N = batch.traj.shape
        
        
        idx = batch['traj']
        condition = batch['cond']
        
        if condition is None:
            raise NotImplementedError
        if 'adjmask' in batch.keys():
            raise NotImplementedError
        if not self.use_agent_mask or 'reagent_mask' not in batch.keys():
            agent_mask = None
            cross_agent_mask = None
        else:
            agent_mask = batch['reagent_mask']
            cross_agent_mask = (~torch.logical_and(agent_mask.unsqueeze(-1), agent_mask.unsqueeze(-2)).view(B*T,N,N))*(-1e9)
            # Shape: (B*T,N,N) <- (B,T,N,N), <- (B,T,N,1) @ (B,T,1,N)
            
        
        # idx and targets are both (B,T) tensor of integers
        # tok_emb = self.token_embedding_table(idx)*ratio_emb # (B,T,N,C)
        tok_emb = self.token_embedding_table(idx) # (B,T,N,C)
        if self.use_len_ratio :
            assert 'ratio' in batch.keys(), "ratio is required"
            ratio:torch.Tensor = batch['ratio'].unsqueeze(-1) # (B,T,N,1)
            ratio_emb = self.ratio_proj(ratio) # (B,T,N,C)
        else:
            ratio_emb = torch.ones_like(tok_emb)
        tok_emb = self.tok_proj(torch.cat((tok_emb,ratio_emb),dim=-1))
        
        if self.use_pe=="abs":
            pos_emb = self.position_embedding_table(torch.arange(T, device=tok_emb.device)).view(1,T,1,-1) # (1,T,1,C)
        else:
            pos_emb = 0
        
        
        
        condition_emb = torch.cat((tok_emb, self.token_embedding_table(condition).view(B,T,N,-1)),dim=-1) # (B,T,N,2C)
        condition_emb:torch.Tensor = self.condition_proj(condition_emb) # (B,T,N,2C)
        condition_score = torch.softmax( #type:ignore
                torch.norm(condition_emb.view(B,T,N,2,-1),dim=-1),#type:ignore
                dim=-1)
        # condition_score = torch.softmax(self.condition_proj(condition_emb),dim=-1) # (B,T,N,2)
        condition_emb = torch.einsum('btnd,btndc->btnc',condition_score,condition_emb.view(B,T,N,2,-1)) # (B,T,N,C)
        
        x = tok_emb + pos_emb + condition_emb # (B,T,N,C)
        x = self.in_proj(x)
        
        return x, agent_mask, cross_agent_mask
    
class EncoderHeadV5(nn.Module):
    # Diff to V4: 
    #   - remove all arch design, just MLP
    def __init__(self, 
                vocab_size:int,
                n_embd:int,
                n_hidden:int,
                block_size:int,
                use_ne:bool,
                use_len_ratio:bool,
                use_agent_mask:bool,
                use_pe:str,
                ):
        
        super().__init__()
        if use_ne:
            self.token_embedding_table = NormalizedEmbedding(vocab_size, n_embd)
        else:
            self.token_embedding_table = nn.Embedding(vocab_size, n_embd)

        if use_pe=="abs":
            self.position_embedding_table = nn.Embedding(block_size, n_embd)
        
        
        self.in_proj = nn.Sequential(
            nn.Linear(n_embd*2+1,n_hidden*2),
            nn.LayerNorm(n_hidden*2),
            nn.SiLU(),
            nn.Linear(n_hidden*2,n_hidden),
            nn.LayerNorm(n_hidden),
            nn.SiLU(),
            nn.Linear(n_hidden,n_embd),
            nn.LayerNorm(n_embd),
            nn.SiLU(),
        )
        
        self.use_pe = use_pe
        self.use_len_ratio = use_len_ratio
        self.use_agent_mask = use_agent_mask
        
        
    def forward(self, batch:Batch):
        B, T, N = batch.traj.shape
        
        
        idx = batch['traj']
        condition = batch['cond']
        
        if condition is None:
            raise NotImplementedError
        if 'adjmask' in batch.keys():
            raise NotImplementedError
        if not self.use_agent_mask or 'reagent_mask' not in batch.keys():
            agent_mask = None
            cross_agent_mask = None
        else:
            agent_mask = batch['reagent_mask']
            cross_agent_mask = (~torch.logical_and(agent_mask.unsqueeze(-1), agent_mask.unsqueeze(-2)).view(B*T,N,N))*(-1e9)
            # Shape: (B*T,N,N) <- (B,T,N,N), <- (B,T,N,1) @ (B,T,1,N)
            
        
        tok_emb = self.token_embedding_table(idx) # (B,T,N,C)
        
        if self.use_len_ratio :
            assert 'ratio' in batch.keys(), "ratio is required"
            ratio:torch.Tensor = batch['ratio'].unsqueeze(-1) # (B,T,N,1)
            ratio_emb = ratio # (B,T,N,1)
        else:
            ratio_emb = torch.ones(B,T,N,1,device=tok_emb.device)
            
        condition_emb = self.token_embedding_table(condition) # (B,T,N,C)
        
        
        tok_emb = self.in_proj(torch.cat((tok_emb,ratio_emb,condition_emb),dim=-1))
        
        if self.use_pe=="abs":
            pos_emb = self.position_embedding_table(torch.arange(T, device=tok_emb.device)).view(1,T,1,-1) # (1,T,1,C)
        else:
            pos_emb = 0
        
        
        x = tok_emb + pos_emb # (B,T,N,C)

        
        return x, agent_mask, cross_agent_mask
    

class EncoderHeadV6(nn.Module):
    # Diff to V5: 
    #   - add pos_emb to tok_emb, rather than x
    def __init__(self, 
                vocab_size:int,
                n_embd:int,
                n_hidden:int,
                block_size:int,
                use_ne:bool,
                use_len_ratio:bool,
                use_agent_mask:bool,
                use_pe:str,
                ):
        
        super().__init__()
        if use_ne:
            self.token_embedding_table = NormalizedEmbedding(vocab_size, n_embd)
        else:
            self.token_embedding_table = nn.Embedding(vocab_size, n_embd)

        if use_pe=="abs":
            self.position_embedding_table = nn.Embedding(block_size, n_embd)
        
        
        self.in_proj = nn.Sequential(
            nn.Linear(n_embd*2+1,n_hidden*2),
            nn.LayerNorm(n_hidden*2),
            nn.SiLU(),
            nn.Linear(n_hidden*2,n_hidden),
            nn.LayerNorm(n_hidden),
            nn.SiLU(),
            nn.Linear(n_hidden,n_embd),
            nn.LayerNorm(n_embd),
            nn.SiLU(),
        )
        
        self.use_pe = use_pe
        self.use_len_ratio = use_len_ratio
        self.use_agent_mask = use_agent_mask
        
        
    def forward(self, batch:Batch):
        B, T, N = batch.traj.shape
        
        
        idx = batch['traj']
        condition = batch['cond']
        
        if condition is None:
            raise NotImplementedError
        if 'adjmask' in batch.keys():
            raise NotImplementedError
        if not self.use_agent_mask or 'reagent_mask' not in batch.keys():
            agent_mask = None
            cross_agent_mask = None
        else:
            agent_mask = batch['reagent_mask']
            cross_agent_mask = (~torch.logical_and(agent_mask.unsqueeze(-1), agent_mask.unsqueeze(-2)).view(B*T,N,N))*(-1e9)
            # Shape: (B*T,N,N) <- (B,T,N,N), <- (B,T,N,1) @ (B,T,1,N)
            
        print('来过这里')
        tok_emb = self.token_embedding_table(idx) # (B,T,N,C)
        
        if self.use_pe=="abs":
            pos_emb = self.position_embedding_table(torch.arange(T, device=tok_emb.device)).view(1,T,1,-1) # (1,T,1,C)
        else:
            pos_emb = 0
        tok_emb = tok_emb + pos_emb # (B,T,N,C)
        
        if self.use_len_ratio :
            assert 'ratio' in batch.keys(), "ratio is required"
            ratio:torch.Tensor = batch['ratio'].unsqueeze(-1) # (B,T,N,1)
            ratio_emb = ratio # (B,T,N,1)
        else:
            ratio_emb = torch.ones(B,T,N,1,device=tok_emb.device)
            
        condition_emb = self.token_embedding_table(condition) # (B,T,N,C)
        
        
        x = self.in_proj(torch.cat((tok_emb,ratio_emb,condition_emb),dim=-1))
        
        
        

        
        return x, agent_mask, cross_agent_mask


def _comp_time_weight(weight_type:str,block_size:int,**kwargs)->torch.Tensor:
    tarange = torch.arange(block_size,dtype=torch.float)+1
    if weight_type == "none":
        res= torch.ones(block_size,dtype=torch.float)
    elif weight_type == "exp":
        base = kwargs.get("base",np.exp(1)) 
        # return torch.exp(torch.arange(block_size).float())
        # return torch.exp(tarange/base)
        res= torch.exp(tarange*torch.log(torch.tensor(base,dtype=torch.float)))
    elif weight_type == "linear":
        res= tarange
    elif weight_type == "quadratic":
        res= tarange**2
    elif weight_type == "inv_lin":
        rat = kwargs.get("rat",0.9999)
        res= 1/(1-tarange/block_size*rat)
    elif weight_type == "inv_quad":
        rat = kwargs.get("rat",0.9999)
        res= 1/(1-(tarange/block_size)**2*rat)
    else:
        raise NotImplementedError
    return res/ res.mean()

class SpatialTemporalMultiAgentModel(nn.Module):

    def __init__(self, 
                vocab_size:int, 
                n_embd:int, 
                n_hidden:int, 
                n_layer:int, 
                n_head:int, 
                block_size:int, 
                window_size:int=-1,
                dropout=0.1, 
                use_pe:str="abs",
                use_ne=True, 
                use_agent_mask=False, 
                use_len_ratio = False,
                norm_position='prenorm',
                time_weighted_loss="none",
                use_enchead_ver=1,
                use_twl=False,
                ):
        super().__init__()
        
        assert use_pe in ["abs","alibi"], f"Positional Encoding {use_pe} is not supported yet"
        
        match use_enchead_ver:
            case 1:
                self.encoder = EncoderHead(vocab_size,n_embd,n_hidden,block_size,use_ne,use_len_ratio,use_agent_mask)
            case 2:
                self.encoder = EncoderHeadV2(vocab_size,n_embd,n_hidden,block_size,use_ne,use_len_ratio,use_agent_mask)
            case 3:
                self.encoder = EncoderHeadV3(vocab_size,n_embd,n_hidden,block_size,use_ne,use_len_ratio,use_agent_mask)
            case 4:
                self.encoder = EncoderHeadV4(vocab_size,n_embd,n_hidden,block_size,use_ne,use_len_ratio,use_agent_mask,use_pe)
            case 5:
                self.encoder = EncoderHeadV5(vocab_size,n_embd,n_hidden,block_size,use_ne,use_len_ratio,use_agent_mask,use_pe)
            case 6:
                self.encoder = EncoderHeadV6(vocab_size,n_embd,n_hidden,block_size,use_ne,use_len_ratio,use_agent_mask,use_pe)
            
            
        self.blocks = nn.ModuleList([SpatialTemporalBlock(n_hidden, n_head,block_size,
                                                            norm_position,
                                                            window_size,
                                                            dropout,
                                                            # flag_sa=(l%3!=3),
                                                            flag_sa=False,
                                                            flag_ta=(l%3!=3),   
                                                           ) for l in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_hidden) # final layer norm
        
        if use_len_ratio:
            self.lm_head = nn.Linear(n_hidden, vocab_size*2+1)
                # Vocab (V), ratio mu (V), ratio log sigma (1)
        else:
            self.lm_head = nn.Linear(n_hidden, vocab_size)
        
        
        assert time_weighted_loss in ["none","exp","linear","quadratic","inv_lin","inv_quad"], \
            f"time_weighted_loss type {time_weighted_loss} is not supported"
        self.use_twl = use_twl
        self.time_weighted_loss = time_weighted_loss
        self.register_buffer("time_weight",_comp_time_weight(time_weighted_loss,block_size).requires_grad_(False))
        print(f"Model: {self.time_weight/self.time_weight.mean()=}")
        
        
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.use_agent_mask = use_agent_mask
        self.use_len_ratio = use_len_ratio

    
    def forward(self, batch):
        # Input: 
            # idx, ratio and targets: (B,T,N)  
            # condition: (B,T,N,2) 
            # adjmask: (B,T,N,V) 
            # agent_mask: (B,T,N)
                # both mask: 1 for reachable, 0 for unreachable
        # Output: (B,T,N,V)

        assert all(a in batch.keys() for a in ['traj','cond']),\
            f"Batch should have keys ['traj','cond'], got {batch.keys()}"
        
        x, agent_mask, cross_agent_mask = self.encoder(batch)
        
        
        for block in self.blocks:
            x = block(x,cross_agent_mask) # (B,T,N,C)
        
        x = self.ln_f(x) # (B,T,N,C)
        logits:torch.Tensor = self.lm_head(x) # (B,T,N,V) or (...,2V+1)
        
        loss = self.loss_fn(logits,batch,agent_mask)

        return logits, loss
    
    def loss_fn(self,logits:torch.Tensor,batch,agent_mask=None):
        # raise NotImplementedError
            # logit_ratio = None
        
        # if 'adjmask' in batch.keys():
        #     adjmask = batch['adjmask']
        #     logits = torch.masked_fill(logits,adjmask==0,-1e7)

        if not 'traj_targ' in batch.keys():
            return None
            
        if self.use_len_ratio:
            logit_state, logit_ratio, logit_sigma = logits.split([self.vocab_size,self.vocab_size, 1],dim=-1)
        else:
            logit_state = logits
            
        targets = batch['traj_targ']
        B, T, N, V = logit_state.shape
        
        logit_state_ = logit_state.view(B*T*N, V)
        targets = targets.view(B*T*N)
        state_loss = F.cross_entropy(logit_state_, targets, reduction='none')
        state_loss = state_loss.view(B,T,N).sum(dim=-1)
        last_state_loss = state_loss[:,-1].mean()
        
        
        if self.use_len_ratio:
            assert 'ratio_targ' in batch.keys()
            ratio_targ = batch['ratio_targ'].view(B*T*N,1)

            logit_ratio_ = logit_ratio.view(B*T*N, V) # type:ignore
            logit_ratio_ = torch.gather(logit_ratio_,1,targets.view(-1,1)) # (B*T*N,1), in [-\inf,+\inf]
            
            # ratio_loss = F.mse_loss(F.sigmoid(logit_ratio_),ratio_targ,reduction='mean') # (B*T*N)
            ratio_loss = -SigmoidGaussian.log_prob_(ratio_targ,logit_ratio_,(logit_sigma.view(-1,1))) # (B*T*N,1)
            ratio_loss = ratio_loss.view(B,T,N).sum(dim=-1)
            last_ratio_loss = ratio_loss[:,-1].mean()
            assert not torch.isnan(ratio_loss).any(), "raw_loss contains NaN"
        else:
            ratio_loss = torch.tensor([0.0],device=logit_state.device)
            last_ratio_loss = torch.tensor([0.0],device=logit_state.device)
            
        # raw_loss = (state_loss + ratio_loss) #(B*T*N,)  #.view(B,T,N)
        #  Now: state_loss [B,T], ratio_loss [B,T]
        
        
        if not self.use_twl:
            time_weight_mat = torch.tensor(1.0,device=logit_state.device)
        else:
            is_start = batch.is_start.view(-1, 1)#.float() #(B,1)
            time_weight_mat:torch.Tensor = is_start + (1 - is_start) * self.time_weight #(B,T)
            time_weight_mat = time_weight_mat/(time_weight_mat.mean(dim=-1,keepdim=True))*(is_start*(T-1)+1)
            
        # breakpoint()
        
            # idx=5; from pprint import pprint;pprint (list(zip(range(T),targets.view(B,T).cpu().numpy()[idx],ratio_targ.view(B,T).cpu().numpy()[idx],state_loss.view(B,T).detach().cpu().numpy()[idx],ratio_loss.view(B,T).detach().cpu().numpy()[idx])))
            #   torch.where(batch.traj[:,0,0]==0)
            # torch.topk(logit_state[idx,55,0],5)
        state_loss = (state_loss*time_weight_mat).mean()
        ratio_loss = (ratio_loss*time_weight_mat).mean()
        
        # if agent_mask is not None:
        #     raise NotImplementedError
        #     mask_weight = agent_mask.view(B*T*N)
        #     loss = ((raw_loss)*mask_weight).sum()/mask_weight.sum()
        # else:
        #     loss = raw_loss.mean()
        return (state_loss, ratio_loss, last_state_loss, last_ratio_loss)
    
    def decode_strategy(self, logits:torch.Tensor, agent_mask:Optional[torch.Tensor]=None, sampling_strategy="random",**kwargs):
        # logits: (M,V)
        # agent_mask: (M)
        M,V = logits.shape
        if not self.use_agent_mask:
            agent_mask = None
        
        if self.use_len_ratio:
            logit_ratio:torch.Tensor
            logit_sigma:torch.Tensor
            logit_state, logit_ratio, logit_sigma = logits.split([self.vocab_size,self.vocab_size, 1],dim=-1)
        else:
            logit_state:torch.Tensor = logits
            
        temp = kwargs.pop("tempperature",1.0)
        # apply softmax to get probabilities
        probs_state = F.softmax(logit_state/temp, dim=-1) # (B*N, V)

 
        # sample from the distribution
        if sampling_strategy == "random":
            idx_next = torch.multinomial(probs_state, num_samples=1) # (B*N, 1)
            
        elif sampling_strategy == "top_k":
            top_k = kwargs.pop("top_k",10)
            topk_idx = torch.topk(probs_state,top_k,dim=-1)
            idx_next__ = torch.multinomial(topk_idx.values, num_samples=1) # (B*N, 1)
            idx_next = topk_idx.indices.gather(-1,idx_next__).view(M,1)
            
        elif sampling_strategy == "top_p":
            top_p = kwargs.pop("top_p",0.9)
            
            sorted_probs, sorted_indices = torch.sort(probs_state, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = (cumulative_probs > top_p)
            sorted_indices_to_remove = torch.roll(sorted_indices_to_remove, 1, 1)
            sorted_indices_to_remove[:, 0] = False
            
            unsorted_indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
            probs_state[unsorted_indices_to_remove] = 0
            # print("sorted_indices_to_remove:",sorted_indices_to_remove)
            # print("unsorted_indices_to_remove:",unsorted_indices_to_remove)
            # print("probs",probs)
            probs_state /= probs_state.sum(dim=-1, keepdim=True)
            # print("probs.sum",probs.sum(dim=-1, keepdim=True))
            
            idx_next = torch.multinomial(probs_state, num_samples=1) # (B*N, 1)
            
        elif sampling_strategy =="greedy":
            idx_next = torch.argmax(probs_state,dim=-1).view(M,1)
        else:
            raise NotImplementedError
        
        if agent_mask is not None:
            idx_next = torch.masked_fill(idx_next,agent_mask.view(-1,1)==0,0)
            
        if self.use_len_ratio:
            dist_ratio_next = SigmoidGaussian(loc=logit_ratio.gather(-1,idx_next),log_scale=(logit_sigma)) # (B*N,1)
            ratio_next = dist_ratio_next.sample() # (B*N,1)
            # probs_ratio = dist_ratio_next.log_prob(ratio_next) # (B*N,1)
        else:
            ratio_next = None
        
        
        
        probs = probs_state #TODO: add ratio prob
        return idx_next,ratio_next, probs

    def generate(self, idx, max_new_tokens,
                 condition = None, get_adj_mask = None, agent_mask = None,
                 sampling_strategy="random",**kwargs): 
        raise NotImplementedError
        
        if not self.use_agent_mask:
            agent_mask = None
        if agent_mask is not None:
            raise NotImplementedError
        B, T, N = idx.shape
        
        prob_list = torch.ones((B,T,N)).to(self.device)
        
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            
            condition_ = condition+torch.zeros((*idx_cond.shape,2)).to(self.device).long()
            logits, loss = self(idx_cond,condition=condition_,adjmask=get_adj_mask(idx_cond) if get_adj_mask is not None else None)
            
            # focus only on the last time step
            logits = logits[:, -1, :, :].view(B*N,-1) # becomes (B*N,V)
            
            idx_next, probs = self.decode_strategy(logits,None,sampling_strategy,**kwargs)
                
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next.view(B,1,N)), dim=1) # (B, T+1, N)
            prob_list = torch.cat((prob_list,probs.gather(-1,idx_next).view(B,1,N)),dim=1)
        return idx,prob_list

    def log_prob_traj(self, batch:Batch):
        # traj: (B,T,N), condition: (B,T,N,2)
        B, T, N = batch.shape
        assert all(a in batch.keys() for a in ['traj','traj_targ','cond']),\
            f"Batch should have keys ['traj','traj_targ','cond'], got {batch.keys()}"
        traj = batch['traj']
        target = batch['traj_targ']
        condition = batch['cond']
        
        if not self.use_agent_mask or 'reagent_mask' not in batch.keys():
            agent_mask = None
        else:
            agent_mask = batch['reagent_mask']
        if 'adjmask' in batch.keys():
            raise NotImplementedError
        
        if T>self.block_size:
            unfold = True
            # we need to re organize traj, so that we compute the prob in a sliding windows ways with overlapping
            traj = traj.unfold(1,self.block_size,1) # (B,T,N) -> (B,T-block_size+1,N, block_size)
            condition = condition.unfold(1,self.block_size,1) # (B,T,N,2) -> (B,T-block_size+1,N,2,block_size)
            
            traj = traj.transpose(2,3).view(-1,self.block_size ,N).contiguous() # (B*(T-block_size+1),block_size,N)
            condition = condition.transpose(2,4).transpose(3,4).view(-1,self.block_size ,N,2) # (B*(T-block_size+1),block_size,N,2)
            
            if agent_mask is not None:
                agent_mask = agent_mask.unfold(1,self.block_size,1) # (B,T,N) -> (B,T-block_size+1,N,block_size)
                agent_mask = agent_mask.transpose(2,3).view(-1,self.block_size ,N).contiguous() # (B*(T-block_size+1),block_size,N)
            
        else:
            unfold = False
            
        rebatch = {'traj':traj,'cond':condition}
        if agent_mask is not None:
            rebatch['reagent_mask'] = agent_mask
            
        logits, _ = self(Batch(rebatch))
        log_prob = F.log_softmax(logits,dim=-1)
        V = logits.shape[-1]
        
        if unfold:
            ...
            log_prob = log_prob.view(B,-1,self.block_size,N,V)
            log_prob = torch.concatenate([log_prob[:,0,:],log_prob[:,1:,-1]],dim=1)
            # (B,T-block_size+1,block_size,N) -> (B,T-block_size,N)+(B,block_size,N) -> (B,T,N)
            # for each subblock except the first, we pick the last token, for the first block, we pick all tokens
        
        
        log_prob_targ = torch.gather(log_prob,3,target.unsqueeze(-1)).squeeze(-1)
        # (B,T,N,V) -> (B,T,N)
        
        log_prob_sum = log_prob_targ.sum(dim=(-1,-2))
            
            
            
        return log_prob_sum, log_prob_targ
        ...

    
    @property
    def device(self):
        return self.time_weight.device
    
    
# ----------------- DP Wrapper -----------------
# Here we wrap the model with DistributedDataParallel, so that we can use multiple GPUs
# The inner and outer wrapper is used for deconstruct the `Batch` object, because DDP may not support it

# DDP class only do one thing: [[Sync Grad]] among different GPUs
# Parameters are never broadcast between processes. It's user's responsibility to make sure that all processes are using the same model at the beginning.
# for DDP: Constructor, forward method, and differentiation of the output (or a function of the output of this module) are distributed synchronization points. 
#

# after loss.backward(), the grad in each GPU is synchronized to be AVERAGE of all grads

class DPWrapperIn(nn.Module):
    
    def __init__(self,model):
        super().__init__()
        self.model = model
    def forward(self,**kwargs):
        return self.model(Batch(kwargs))
    
class DPWrapperOut(nn.Module):
    def __init__(self,model,rank):
        super().__init__()
        # self.model = model
        self.rank = rank
        self.dp = DDP(DPWrapperIn(model),device_ids=[rank])
        
    def forward(self,batch:Batch):
        return self.dp(**batch) #type:ignore
        
    def __getattr__(self, name: str):
        if name in ["model","device_ids","dp"]:
            return super().__getattr__(name)
        return getattr(self.model, name)
    ...
    
def dp_wrap_model(model, rank):
    return DPWrapperOut(model, rank)


MAMODEL = Union[SpatialTemporalMultiAgentModel,nn.DataParallel[SpatialTemporalMultiAgentModel],DPWrapperOut]