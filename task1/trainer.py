from time import time_ns
from util import *
from util import _batch_collate_fn 
import tqdm
from thop import profile, clever_format
import wandb
import copy 
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import RandomSampler, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from traintool import estimate_loss,UCBGradClip,check_error_rate, get_lr_scheder, model_init_weights
from fttool import LoRA,get_ft_model

def reduce_val(val):
    """
        It's in-place operation
    """
    world_size = dist.get_world_size()
    with torch.no_grad():
        dist.all_reduce(val, async_op=False)
        val /= world_size
    return val



def train(rank:int=0,raw_data:Optional[dict]=None, world_size=1,cfg:dict={},
          get_batch:Callable[[str],Batch]=None): #type:ignore
# def train(rank:int=0,train_data=None, val_data=None, world_size=4,cfg:dict={},
#           get_batch:Callable[[str],Batch]=None): #type:ignore
    torch.cuda.set_device(rank)
    
    setting = cfg['setting']
    expname = cfg['expname']
    device, use_model, N= cfg['device'], cfg['use_model'], cfg['N']
    use_ucbgc, ucbgc_alpha, ucbgc_beta = cfg['use_ucbgc'], cfg['ucbgc_alpha'], cfg['ucbgc_beta']
    batch_size, block_size, max_grad_norm = cfg['batch_size'], cfg['block_size'], cfg['max_grad_norm']
    max_iters, learning_rate, eval_iters, eval_interval,save_interval = cfg['max_iters'], cfg['learning_rate'], cfg['eval_iters'], cfg['eval_interval'], cfg['save_interval']
    lr_min = cfg['lr_min']
    lr_warmup = cfg['lr_warmup']
    lamb_ratio= cfg['lamb_ratio']
    
    
    grad_accumulation = cfg['grad_accumulation']
    if grad_accumulation>1:
        max_iters*=grad_accumulation
        eval_interval*=grad_accumulation
        save_interval*=grad_accumulation
        lr_warmup*=grad_accumulation
        
    
    save_model = cfg['save_model']
    use_wandb = cfg['use_wandb']
    use_adj_mask = cfg['use_adj_mask']
    use_len_ratio = cfg['use_len_ratio']
    use_dp = cfg['use_dp']
    
    is_ft = cfg['is_ft']
    ft_type = cfg['ft_type']
    use_kl_reg = cfg['use_kl_reg']
    kl_lambda:float = cfg['kl_reg_factor']
    assert not cfg['use_twl'], "Here we hardcore KL_loss with twl=nones"
        
    
    def save_model_fn(model,save_path:str):
        
        module = model if isinstance(model,SpatialTemporalMultiAgentModel) else model.dp.module.model
        if is_ft and ft_type == "lora":
            module = LoRA.lora_comb(copy.deepcopy(module),cfg)
        torch.save(module.state_dict(), save_path)
    
    ...
    if not is_ft:
        model = get_model(cfg,load_from=None) 
        model.train()
    else:
        model, pre_model = get_ft_model(cfg)
        
    if use_dp:
        # before this point, the model has been the right device
        dist.init_process_group(backend='nccl', world_size=world_size, rank=rank)
        
        # Ensure each model replica is the same
        for param in model.parameters():
            dist.broadcast(param.data, src=0)
        dist.barrier()
        
        print("Rank: ",rank, "device: ",model.device)
        model = dp_wrap_model(model,rank=rank)
        
        if is_ft and pre_model is not None:
            pre_model = dp_wrap_model(pre_model,rank=rank)
            
        is_head = rank == 0
        
        # Reprepare the data loader, because 
        #   1. the data loader is not picklable
        #   2. each process should have different data for each iteration, i.e. shuffled with each random seed
        #   3. the data should not be replicated in each process
        train_data,val_data = raw_data['train_data'],raw_data['val_data']#type:ignore
        train_trajlen,val_trajlen = raw_data['train_trajlen'],raw_data['val_trajlen']#type:ignore
        get_batch = make_jux_get_batch_fn(train_data, val_data, device, cfg['seed']+rank,
                                   cfg['batch_size'], cfg['tdl_num_workers'], cfg['tdl_prefetch_factor'], 
                                    cfg['block_size'], 
                                    cfg['__use_blockjux'],
                                    train_trajlen, 
                                    val_trajlen
                                   )
    else:
        is_head = True

    
        
    if is_head:
        if not is_ft:
            pm = sum(p.numel() for p in model.parameters())/1e6    
            color_print("Network parameters:",pm, 'M')
            cfg['num_params'] = pm
        else:
            trainable_params_num = sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6
            total_params_num = sum(p.numel() for p in model.parameters())/1e6
            cfg['ft_num_trainable_params'] = trainable_params_num
            cfg['ft_num_total_params'] = total_params_num
            color_print(f"Trainable parameters: {trainable_params_num} M, Total parameters: {total_params_num} M, Ratio: {trainable_params_num/total_params_num:.4f}")
            
        color_print("Equivalent Batch Size: ",f"{batch_size*world_size*grad_accumulation=}")
        print("GFLOPs: ",profile(model = get_model(cfg), 
                                inputs=(get_batch('val'),),
                                verbose=False)[0]/1e9)
        
        run_name = get_runname(cfg)
        cfg['run_name'] = run_name
        color_print("Run Name: ",run_name)
        run_dir = f"./model/{setting}/{run_name}"
        if not os.path.exists(run_dir):
            os.makedirs(run_dir,exist_ok=True)
        write_cfg(cfg,f"{run_dir}/cfg.yaml")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=1e-15)
    # lr_sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2500, gamma=0.5623)
            # after 1200 steps, lr is 0.1 of the original
    # lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iters, eta_min=0)
    lr_sched = get_lr_scheder(optimizer,lr_warmup,max_iters,lr_min)
    if use_ucbgc:
        ucbgc = UCBGradClip(alpha=ucbgc_alpha,beta=ucbgc_beta,max_grad_norm=max_grad_norm)

    
    
    if use_wandb and is_head:
        run = wandb.init(
                project="yq_stma",
                name = run_name,
                save_code=True,
                config=cfg,
            )
    
    # loss_list = []
    ranger = tqdm.tqdm(range(max_iters),ncols=120) if is_head else range(max_iters)
    for it in ranger:
        gc_it = it//grad_accumulation
            # when logging with ga, we use the gc_it, i.e. the number of grad iterations
            
        if (it % eval_interval == 0 or it == max_iters - 1) and True \
            and is_head:
            
            losses = estimate_loss(model,eval_iters, get_batch, 
                                   use_adj_mask=use_adj_mask,
                                   device=device)

            _val_mb = get_batch('val')
            top_two,p_right,p_all = check_error_rate(model,N,_val_mb,device  )
            
            print()
            print(f"step {gc_it}: train loss {losses['train']},",end=" ")
            color_print(f"val loss {losses['val']}",color='cyan')
            print(f"check error rate: ",top_two.cpu().numpy(),p_all.item(),torch.log(p_all).item())
            
            if use_wandb and is_head:
                wandb.log({
                    "loss/train_state":losses['train'][0],
                    "loss/train_ratio":losses['train'][1],    
                    "loss/train_state_last":losses['train'][2],
                    "loss/train_ratio_last":losses['train'][3],    
                    
                    "loss/val_state":losses['val'][0],
                    "loss/val_ratio":losses['val'][1],
                    "loss/val_state_last":losses['val'][2],
                    "loss/val_ratio_last":losses['val'][3],
                    
                    "log_p_all": torch.log(p_all)
                    },step=gc_it)

        if save_model and it % save_interval == 0 and is_head:
            # # module = model.module if isinstance(model,torch.nn.DataParallel) else model
            # module = model if isinstance(model,SpatialTemporalMultiAgentModel) else model.dp.module.model
            # torch.save(module.state_dict(), f"{run_dir}/{gc_it}.pth")
            save_model_fn(model,f"{run_dir}/{gc_it}.pth")

        mb = get_batch('train')

        logits, loss = model(mb)

        state_loss, ratio_loss = loss[0].mean(), loss[1].mean()
        sl_loss:torch.Tensor = (state_loss + ratio_loss*lamb_ratio  ) 
        
        
        if is_ft and use_kl_reg:
            with torch.no_grad():
                pre_logits, _ = pre_model(mb) #type:ignore
            B,T,N,V = logits.shape
            kl_loss = F.kl_div(F.log_softmax(logits,dim=-1),F.softmax(pre_logits,dim=-1), 
                               reduction='none').sum(-1) #B,T,N
            kl_loss = kl_loss.mean() #B,T
            # DEBUG: Check the shape
            
        else:
            kl_loss = torch.tensor(0.0,device=device)

        
        total_loss:torch.Tensor = (sl_loss + kl_lambda * kl_loss)/ grad_accumulation
        
        
        reduce_loss = total_loss.detach().clone()
        reduce_state_loss = state_loss.detach().clone()
        reduce_ratio_loss = ratio_loss.detach().clone()
        reduce_kl_loss = kl_loss.detach().clone()

        total_loss.backward()  # for ddp, we get the average grad over all the gpus
        
        if (it+1) % grad_accumulation:continue
        
            
        if use_ucbgc:
            grad_norm, grad_running_mean, grad_bound, grad_running_std = ucbgc(model)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        lr_sched.step()
        optimizer.zero_grad(set_to_none=True)  
        
        
        if use_dp:
            reduce_val(reduce_loss)
            reduce_val(reduce_state_loss)
            reduce_val(reduce_ratio_loss)
            reduce_val(reduce_kl_loss)
        
        if use_wandb and is_head and gc_it%50==0:
            if gc_it%200==0:
                with torch.no_grad():
                    all_params = torch.cat([p.view(-1) for p in model.parameters()])
                    param_l2_norm = torch.norm(all_params,p=2).item()
                    param_l1_norm = torch.norm(all_params,p=1).item()
                    param_l4_norm = torch.norm(all_params,p=4).item()
                    param_mean, param_std = all_params.mean().item(), all_params.std().item()
                    param_third_mom, param_fourth_mom = \
                        ((all_params - param_mean)**3).mean().item(), ((all_params - param_mean)**4).mean().item()
                    
                    # param_Q1 = torch.quantile(all_params,0.25).item() # It may be too large
                    # param_Q3 = torch.quantile(all_params,0.75).item()
                    
                    
                wandb.log({"param/l2_norm":param_l2_norm,
                            "param/l1_norm":param_l1_norm,
                            "param/l4_norm":param_l4_norm,
                            "param/mean":param_mean,
                            "param/std":param_std,
                            "param/third_mom":param_third_mom,
                            "param/fourth_mom":param_fourth_mom,
                            # "param/Q1":param_Q1,
                            # "param/Q3":param_Q3,
                            },step=gc_it)
                
            wandb.log({
                    "loss/total":reduce_loss.item() ,
                    "loss/total_state":reduce_state_loss.item(),
                    "loss/total_ratio":reduce_ratio_loss.item(),
                    "loss/total_kl":reduce_kl_loss.item(),
                    "grad/norm":grad_norm,
                    "lr":lr_sched.get_last_lr()[0], 
                    },step=gc_it)
            
            if use_len_ratio:
                with torch.no_grad():
                    log_sigma = logits[...,-1].mean().item()
                wandb.log({"ratio_log_sigma":log_sigma},step=gc_it)
            
            if use_ucbgc:
                wandb.log({"grad/running_mean":grad_running_mean,
                        "grad/bound":grad_bound,
                            "grad/running_std":grad_running_std,
                            'grad/clipped':int(grad_norm>grad_bound),
                        },step=gc_it)
                    
    if is_head:
        if save_model:
            save_path:str = f"{run_dir}/final.pth"
            # module = model if isinstance(model,SpatialTemporalMultiAgentModel) else model.dp.module.model
            # torch.save(module.state_dict(), save_path)
            save_model_fn(model,save_path)
            color_print(f"Model is saved to: {save_path}")
            print(f"For evaluation, run:\n CUDA_VISIBLE_DEVICES=X python eval.py -lmc 1 -mlf {save_path}")
            if use_wandb:
                wandb.config.update({"model_path":save_path})
        else:
            color_print("Model is not saved")
            
    if use_wandb and is_head:
       run.finish()
    

def main(args, _runner:SmartRunner):
    cfg = get_cfg(args)
    run_name = get_runname(cfg)
    _runner.mov_log(f"./model/{cfg['setting']}/{run_name}/train.log")
    
    adjlist, adjmat, V = get_graph(cfg)
    
    get_batch,data_shape,raw_data = build_dataloader(cfg)
    
    if cfg['use_dp']:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(29534+ (time_ns()%1000) )
        world_size = cfg['ddp_world_size']
        mp.spawn(train,#type:ignore
            args=(raw_data, world_size, cfg),
            nprocs=world_size,
            join=True)
    else:
        
        train(cfg=cfg,get_batch=get_batch)
    
    
if __name__ == '__main__':
    # main()
    # SmartRunner(main,fire=False)
    curfile_basename = os.path.basename(__file__).split('.')[0]
    log_dir = f"./log/{curfile_basename}/{time.strftime('%d%H_%Y%m_%M%S')}"
    SmartRunner(main,fire=False,log_dir=log_dir)
