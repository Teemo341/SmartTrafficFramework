from util import *
from data_loader import get_dijkstra_dataloader
from pprint import pprint
from torch import nn
import torch
import dill
import tqdm
import random
import wandb
import time

class UCBGradClip():
    # Keep a running average of the gradient norm, and clip the gradient norm to be within a certain range
    def __init__(self, alpha=0.9, beta=2.0, min_std=1e-10, max_grad_norm=1.0):
        self.alpha = alpha
        self.beta = beta
        self.min_std = min_std
        self.max_grad_norm = max_grad_norm
        
        self.running_mean = 0
        self.running_square = 0
        
    def __call__(self, model):
        
        std = np.clip(np.sqrt(self.running_square - self.running_mean**2), a_min=self.min_std, a_max=None)
        
        bound = np.min([self.max_grad_norm, self.running_mean + self.beta*std])
        
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), bound).cpu().item()
        
        self.running_mean = self.alpha * self.running_mean + (1 - self.alpha) * grad_norm
        
        self.running_square = self.alpha * self.running_square + (1 - self.alpha) * grad_norm**2
        
        return grad_norm, self.running_mean, bound, std
        
@torch.no_grad()
def estimate_loss(model, eval_iters, train_iter, 
                  val_iter, device='cuda'):
    out = {}
    loader = ['train', 'val']
    device = model.module.device if isinstance(model,torch.nn.DataParallel) else model.device
    model.eval()
    for i, dataloader in enumerate([train_iter, val_iter]):
        losses = 0
        iter_count = 0
        for x, x_valid, od_condition, y, adj_indices, adj_values in dataloader:
            iter_count += 1
            x, x_valid, od_condition, y, adj_indices, adj_values = x.to(device), x_valid.to(device), od_condition.to(device),\
                                                                y.to(device), adj_indices[0].unsqueeze(0).to(device), adj_values.to(device)
            y = y.long()
            logits, loss = model(x, x_valid, y, condition=od_condition, adj=(adj_indices, adj_values))

            loss = loss.mean().item()
            if loss > 1e5:
                print("Loss is too high, check the model")
                import pdb; pdb.set_trace()
            losses += loss
            #torch.cuda.empty_cache()
            if iter_count == 100:
                break
        out[loader[i]] = losses / iter_count
        #torch.cuda.empty_cache()
    model.train()
    
    return out

def train(cfg):
    random.seed(1337)
    np.random.seed(1337)
    torch.manual_seed(1337)

    setting = cfg['setting']
    expname = cfg['expname']
    device, use_model, N= cfg['device'], cfg['use_model'], cfg['N']
    use_ucbgc, ucbgc_alpha, ucbgc_beta = cfg['use_ucbgc'], cfg['ucbgc_alpha'], cfg['ucbgc_beta']
    batch_size, block_size, max_grad_norm = cfg['batch_size'], cfg['block_size'], cfg['max_grad_norm']
    max_iters, learning_rate, eval_iters, eval_interval,save_interval = cfg['max_iters'], cfg['learning_rate'], cfg['eval_iters'], cfg['eval_interval'], cfg['save_interval']
    use_wandb = cfg['use_wandb']
    root_path = cfg['root_path']
    graph_path = cfg['graph_path']
    data_path = cfg['data_path']
    length_path = cfg['length_path']
    od_per_graph = cfg['od_per_graph']
    num_file = cfg['num_file']
    iter_per_epoch = cfg['iter_per_epoch']
    model = get_model(cfg, load_from=cfg['eval_load_from'])

    pm = sum(p.numel() for p in model.parameters())/1e6
    pm_non_embedding = count_non_embedding_params(model)/1e6
    print("Network parameters: ", pm, 'M')
    print("Non-embedding network parameters: ", pm_non_embedding, 'M')
    #print("GFLOPs: ",profile(model = get_model(cfg), inputs=(torch.randint(0, 101, (batch_size, block_size, N)).to(device),),verbose=False)[0]/1e9) 
        # This will create new var in the model, which is stored in cpu, causing error in Data Parallel
        # Hence, we decide to create another model to profile the FLOPs, instead of the model used for training

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=1e-15)
    # lr_sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2500, gamma=0.5623)
            # after 1200 steps, lr is 0.1 of the original
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iters, eta_min=0)
    if use_ucbgc:
        ucbgc = UCBGradClip(alpha=ucbgc_alpha, beta=ucbgc_beta, max_grad_norm=max_grad_norm)

    run_name = ("Debug_" if cfg['debug'] else ""   )+f"{expname}_{setting}_N{N}_{use_model}_pm{pm:4f}_it{max_iters}_bs{batch_size}_t{time.strftime('%m%d%H%M%S')}"
    print("Run Name: ",run_name)
    if not os.path.exists(f"./model/{run_name}"):
        os.makedirs(f"./model/{run_name}")
    
    if use_wandb:
        run = wandb.init(
            project="yq_stma",
            name = run_name,
            save_code=True,
            config=cfg,
            )
    
    if setting in ['boston', 'paris', 'porto', 'beijing', 'jinan']:
        train_iter = get_dijkstra_dataloader(batch_size=batch_size, root_path=root_path, graph_path=graph_path,
                                              data_path=data_path, length_path=length_path, flag='train', hop=cfg['hop'],
                                              od_per_graph=od_per_graph, mode='train', num_file=num_file)
        val_iter = get_dijkstra_dataloader(batch_size=batch_size, root_path=root_path, graph_path=graph_path,
                                            data_path=data_path, length_path=length_path, flag='val', hop=cfg['hop'],
                                            od_per_graph=od_per_graph, mode='train', num_file=num_file)
    else:
        raise NotImplementedError
    
    early_stopping = EarlyStopping(patience=cfg['patience'], verbose=True)

    for it in tqdm.tqdm(range(max_iters), ncols=120):
        iter_count = 0
        mean_loss = 0
        if it % eval_interval == 0 or it == max_iters - 1:
            losses = estimate_loss(model, eval_iters, train_iter, 
                                   val_iter, device=device)
            print()
            print(f"step {it}: train loss {losses['train']:.4e}, val loss {losses['val']:.4e}")
            early_stopping(losses['val'], model, f"./model/{run_name}")

            if early_stopping.early_stop:
                print("Early stopping")
                break
    
            if use_wandb:
                wandb.log({"train_loss":losses['train'],
                           "val_loss":losses['val'],}, step=it)
                
        if cfg['save_model'] and it % save_interval == 0:
            module = model.module if isinstance(model,torch.nn.DataParallel) else model
            torch.save(module.state_dict(), f"./model/{run_name}/{it}.pth")
            
        for x, x_valid, od_condition, y, adj_indices, adj_values in train_iter:
            iter_count += 1
            x, x_valid, od_condition, y, adj_indices, adj_values = x.to(device), x_valid.to(device), od_condition.to(device),\
                                                                y.to(device), adj_indices[0].unsqueeze(0).to(device), adj_values.to(device)
            y = y.long()
            logits, loss = model(x, x_valid, y, condition=od_condition, adj=(adj_indices, adj_values))
                    
            loss = loss.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if use_ucbgc:
                grad_norm, grad_running_mean, grad_bound, grad_running_std = ucbgc(model)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            mean_loss += loss.item()
            optimizer.step()
            #torch.cuda.empty_cache()
            if iter_count == iter_per_epoch:
                break
        
        lr_sched.step()
        mean_loss /= iter_count
        
        if use_wandb:
            wandb.log({"loss":mean_loss,
                       "grad_norm":grad_norm,}, step=it)
            if use_ucbgc:
                wandb.log({"grad_running_mean":grad_running_mean,
                           "grad_bound":grad_bound,
                            "grad_running_std":grad_running_std,
                            'grad_clipped':int(grad_norm>grad_bound),}, step=it)
        #torch.cuda.empty_cache()

    if cfg['save_model']:
        save_path = f"./model/{run_name}/final.pth"
        module = model.module if isinstance(model,torch.nn.DataParallel) else model
        torch.save(module.state_dict(), save_path)
        print(f"Model is saved to: {save_path}")
        if use_wandb: 
            wandb.config.update({"model_path":save_path})
    else:
        print("Model is not saved")
            
    if use_wandb:      
       run.finish()

def main(args):
    cfg = get_cfg(args)
    train(cfg)
    
if __name__ == '__main__':
    # main()
    smart_run(main,fire=False)