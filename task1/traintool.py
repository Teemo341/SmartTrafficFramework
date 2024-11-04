from util import *
import tqdm
from thop import profile, clever_format
import wandb

from torch.optim.lr_scheduler import LRScheduler

class UCBGradClip():
    # Keep a running average of the gradient norm, and clip the gradient norm to be within a certain range
    def __init__(self, alpha:float=0.9,beta:float=2.0, min_std:float=1e-10,max_grad_norm:float=1.0):
        self.alpha = alpha
        self.beta = beta
        self.min_std = min_std
        self.max_grad_norm = max_grad_norm
        
        self.running_mean:float = 0
        self.running_square:float = 0
        
    def __call__(self, model:nn.Module)->Tuple[float,float,float,float]:
        
        std:float = np.clip(np.sqrt(self.running_square - self.running_mean**2), a_min=self.min_std, a_max=None)
        
        bound:float = np.min([self.max_grad_norm, self.running_mean + self.beta*std])
        
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), bound).cpu().item())
        
        self.running_mean = self.alpha * self.running_mean + (1 - self.alpha) * grad_norm
        
        self.running_square = self.alpha * self.running_square + (1 - self.alpha) * grad_norm**2
        
        return grad_norm, self.running_mean, bound, std
        

@torch.no_grad()
def estimate_loss(model:MAMODEL,
                  eval_iters, get_batch,use_adj_mask=False,device='cuda'):
    out = {}    
    device = model.dp.module.model.device if isinstance(model, DPWrapperOut) else model.device
    model.eval()
    for split in ['train', 'val']:
        # losses = torch.zeros(eval_iters,device = device)
        # state_loss = 0
        # ratio_loss = 0
        res = np.zeros((4,))
        for k in range(eval_iters):
            
            mb = get_batch(split)
            logits, loss = model(mb)
            
            # state_loss += loss[0].mean().item()
            # ratio_loss += loss[1].mean().item()
            res +=np.array([ loss[k].mean().item() for k in range(4)])
            # losses[k] = loss[0].mean().item()+loss[1].mean().item()
            if (res[:2].sum())/(k+1)>1e5:
                print("Loss is too high, check the model")
                breakpoint()
            # assert losses[k]<1e5, "Loss is too high, check the model"
        out[split] = res/eval_iters
    model.train()
    return out


@torch.no_grad()
def check_error_rate(model,N,_val_mb:Batch,device='cuda'):
    # See `wjxie/notes.md` `24/06/12` section for why we compute it
    model.eval()
    _val_mb=_val_mb[:1]
    # _val_mb['traj'] = _val_mb['cond'][:,:,:,0]
    
    logits, _ = model(_val_mb)
    
    idx = 0
    prob = torch.softmax(logits[:,-1,:].view(N,-1), dim=-1)
    
    top_two = (torch.topk(prob[idx], 3).values)
    p_right = top_two.sum()
    p_all = 1-(p_right)**N
    model.train()
    return (top_two,p_right,p_all)




def get_lr_scheder(optimizer:torch.optim.Optimizer,lr_warmup:int,max_iters:int,eta_min:float=0):
    if eta_min<0:
        eta_min=optimizer.param_groups[0]['lr']*1e-2
    if lr_warmup>0:
        lrs_1 = torch.optim.lr_scheduler.LinearLR(optimizer,1e-9,1.0,lr_warmup)
        lrs_2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iters-lr_warmup, eta_min=eta_min)
        lr_sched = torch.optim.lr_scheduler.SequentialLR(optimizer,[lrs_1,lrs_2],milestones=[lr_warmup])
    else:
        lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iters, eta_min=eta_min)
    return lr_sched


def model_init_weights(m):
    # I found that in the module initialization, There IS a good initialization for the model weight
    # which means we don't need to do this manually
    
    # self.reset_parameters()
    
    
    # if type(m) == nn.Linear:
    #     nn.init.xavier_uniform_(m.weight)
    #     if m.bias is not None:
    #         nn.init.zeros_(m.bias)
    # if type(m) == nn.Embedding:
    #     nn.init.normal_(m.weight)
    # if type(m) == nn.LayerNorm:
    #     nn.init.ones_(m.weight)
    #     nn.init.zeros_(m.bias)
    ...