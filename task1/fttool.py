from util import *
from traintool import estimate_loss,UCBGradClip,check_error_rate
import wandb
import tqdm
import copy 


class LoRA(nn.Module):
    def __init__(self, layer:nn.Linear, rank:int, use_bias:bool=True):
        super(LoRA, self).__init__()
        self.layer = layer
        self.rank = rank
        self.U = nn.Parameter(torch.Tensor(layer.in_features, rank))
        self.V = nn.Parameter(torch.Tensor(layer.out_features, rank))
        if use_bias and layer.bias is not None:
            self.bias = nn.Parameter(torch.Tensor(layer.out_features))
        else:
            self.bias = None
            
    def forward(self, x):
        return torch.matmul(x, self.U) @ self.V.T + (self.bias if self.bias is not None else 0) + self.layer(x)
    
    @classmethod
    def lora_wrap(cls,model:nn.Module,cfg:dict):
        rank=8
        to_rep = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                to_rep.append((name,cls(module, rank=rank).to(module.weight.device)))
        for name, module in to_rep:
            setattr(model, name, module)
            
    @classmethod
    def lora_init(cls,model:nn.Module,cfg:dict):
        for name, module in model.named_modules():
            if isinstance(module, cls):
                # nn.init.constant_(module.U, 0)
                # nn.init.constant_(module.V, 1)
                nn.init.xavier_normal_(module.U)
                nn.init.xavier_normal_(module.V)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    @classmethod
    def lora_comb(cls,model:nn.Module,cfg:dict):
        for name, module in model.named_modules():
            if isinstance(module, cls):
                layer = module.layer
                layer.weight = nn.Parameter(layer.weight + torch.matmul(module.V, module.U.T))
                if layer.bias is not None and module.bias is not None:
                    layer.bias = nn.Parameter(layer.bias + module.bias) 
                setattr(model, name, layer)
                # setattr(model, name, module.layer)
        return model

def get_ft_model(cfg)->Tuple[MAMODEL,Optional[MAMODEL]]:
    pre_model = get_model(cfg,load_from=cfg['model_load_from'])
    ft_type = cfg['ft_type']
    use_kl_reg = cfg['use_kl_reg']
    if use_kl_reg:
        ft_model = copy.deepcopy(pre_model)
        
        pre_model.eval() 
        for param in pre_model.parameters(): 
            param.requires_grad = False
    else:
        ft_model  = pre_model 
        pre_model = None
        
    if ft_type == 'lora':
        for param in ft_model.parameters():
            param.requires_grad = False
        LoRA.lora_wrap(ft_model,cfg)
        LoRA.lora_init(ft_model,cfg)
    
    ft_model.train()
    
    return ft_model,pre_model


def _finetune(model:MAMODEL,
             run_name:str,get_batch:GET_BATCH,cfg:dict, 
             pre_model:Optional[MAMODEL]=None
             ):
    
    
    run_dir = f"./model/{run_name}"
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    write_cfg(cfg,f"{run_dir}/cfg.yaml")
    
    device, N = cfg['device'],  cfg['N']
    use_ucbgc, ucbgc_alpha, ucbgc_beta = cfg['use_ucbgc'], cfg['ucbgc_alpha'], cfg['ucbgc_beta']
    max_grad_norm = cfg['max_grad_norm']
    max_iters, learning_rate, eval_iters, eval_interval,save_interval = cfg['max_iters'], cfg['learning_rate'], cfg['eval_iters'], cfg['eval_interval'], cfg['save_interval']
    use_wandb = cfg['use_wandb']
    use_new_dataloader = cfg['new_dataloader']
    use_adj_mask = cfg['use_adj_mask']
    use_dp = cfg['use_dp']
    
    ft_type = cfg['ft_type']
    use_kl_reg = cfg['use_kl_reg']
    kl_lambda:float = cfg['kl_reg_factor']
    assert not (use_kl_reg and pre_model is None), "Please specify the pre_model for kl regularization"
    
    if ft_type == 'lora':
        for param in model.parameters():
            param.requires_grad = False
        LoRA.lora_wrap(model,cfg)
        LoRA.lora_init(model,cfg)
    
    model.train()
    if use_kl_reg:
        pre_model.eval() #type:ignore
        for param in pre_model.parameters(): #type:ignore
            param.requires_grad = False
    
    trainable_params_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params_num = sum(p.numel() for p in model.parameters())
    cfg['ft_num_trainable_params'] = trainable_params_num
    cfg['ft_num_total_params'] = total_params_num
    print(f"Trainable parameters: {trainable_params_num}, Total parameters: {total_params_num}, Ratio: {trainable_params_num/total_params_num:.4f}")
    
    if use_dp:
        raise NotImplementedError
        # model = torch.nn.DataParallel(model, device_ids=cfg['dp_device_ids'] )
        # model = dp_wrap_model(model,dp_device_ids=cfg['dp_device_ids'])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=1e-15)
    # lr_sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2500, gamma=0.5623)
            # after 1200 steps, lr is 0.1 of the original
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iters, eta_min=0)
    if use_ucbgc:
        ucbgc = UCBGradClip(alpha=ucbgc_alpha,beta=ucbgc_beta,max_grad_norm=max_grad_norm)
        
        
        
    if use_wandb:
        run = wandb.init(
            project="yq_stma",
            name = run_name,
            save_code=True,
            config=cfg,
            
            )
        
    loss_list = []
    for it in tqdm.tqdm(range(max_iters),ncols=120):
        if (it % eval_interval == 0 or it == max_iters - 1) and True:
            
            losses = estimate_loss(model,eval_iters, get_batch, 
                                   use_adj_mask=use_adj_mask,
                                   device=device)

            _val_mb = get_batch('val')
            top_two,p_right,p_all = check_error_rate(model,N,_val_mb,device  )
            
            print()
            print(f"step {it}: train loss {losses['train']}, val loss {losses['val']}")
            print(f"check error rate: ",top_two,p_right,p_all,torch.log(p_all))
            
            if use_wandb:
                wandb.log({"train_loss":losses['train'],
                           "val_loss":losses['val'],
                        "log_p_all": torch.log(p_all)
                        },step=it)
                
        if cfg['save_model'] and it % save_interval == 0:
            # module = model.module if isinstance(model,torch.nn.DataParallel) else model
            module = model if isinstance(model,SpatialTemporalMultiAgentModel) else model.dp.module.model
            # torch.save(module.state_dict(), f"./model/{run_name}/{it}.pth")
            if ft_type == "lora":
                module = LoRA.lora_comb(copy.deepcopy(module),cfg)
            torch.save(module.state_dict(), f"{run_dir}/{it}.pth")

        mb = get_batch('train') 
        logits, loss = model(mb)
        state_loss, ratio_loss = loss[0].mean(), loss[1].mean() 
        sl_loss:torch.Tensor = state_loss #DEBUG:
        total_loss = sl_loss
        
        if use_kl_reg:
            with torch.no_grad():
                pre_logits, _ = pre_model(mb) #type:ignore
            B,T,N,V = logits.shape
            kl_loss = F.kl_div(F.log_softmax(logits,dim=-1),F.softmax(pre_logits,dim=-1), 
                               reduction='none').mean(-1) #B,T,N
            kl_loss = (model.time_weight[None,:,None].tile(B,1,N)*kl_loss/model.time_weight.mean()).mean()
            
            total_loss = total_loss+ kl_lambda * kl_loss
            ## it's different from total_loss += xxx
            ## 
        else:
            kl_loss = torch.tensor(0.0)
            
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        if use_ucbgc:
            grad_norm, grad_running_mean, grad_bound, grad_running_std = ucbgc(model)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        loss_list.append(total_loss.item())
        optimizer.step()
        lr_sched.step()
            
        
        if use_wandb:
            wandb.log({"loss":total_loss.item(),
                       'sl_loss':sl_loss.item(),
                       "kl_loss":kl_loss.item(),
                       "grad_norm":grad_norm,
                    #    'agent_sparsity':1-torch.mean(mb.to(torch.float)).cpu().item() 
                       },step=it)
            if use_ucbgc:
                wandb.log({"grad_running_mean":grad_running_mean,
                           "grad_bound":grad_bound,
                            "grad_running_std":grad_running_std,
                            'grad_clipped':int(grad_norm>grad_bound),
                           },step=it)
            
    if cfg['save_model']:
        # save_path = f"./model/{run_name}/final.pth"
        save_path = f"{run_dir}/final.pth"
        # module = model.module if isinstance(model,torch.nn.DataParallel) else model
        module = model if isinstance(model,SpatialTemporalMultiAgentModel) else model.dp.module.model
        if ft_type == "lora":
            module = LoRA.lora_comb(copy.deepcopy(module),cfg)
        torch.save(module.state_dict(), save_path)
        print(f"Model is saved to: {save_path}")
        if use_wandb:
            wandb.config.update({"model_path":save_path})
    else:
        print("Model is not saved")
            
    if use_wandb:      
       run.finish()




def main(args, _runner=None):
    cfg = get_cfg(args)
    
        
    get_graph(cfg)[1]
    get_batch,data_shape,raw_data = build_dataloader(cfg) 
    

    assert cfg['model_load_from'] is not None, "Please specify the model to finetune, got None. e.g. --model_load_from=\"your_model.pth\""
    pre_model = get_model(cfg,load_from=cfg['model_load_from'])
    ft_model = copy.deepcopy(pre_model)
    # ft_model = get_model(cfg)
    
    
    ori_run_name =  cfg['model_load_from'].split('/')[-2] 
    new_run_name = (
        ("Debug_" if cfg['debug'] else ""   ) + 
        f"{cfg['expname']}_it{cfg['max_iters']}_bs{cfg['batch_size']}_t{time.strftime('%m%d%H%M%S')}_ft[{ori_run_name}]"
                    )
    cfg['run_name'] = new_run_name
    print("Run Name: ",new_run_name)
    if not os.path.exists(f"./model/{new_run_name}"):
        os.makedirs(f"./model/{new_run_name}")
        
    
    _finetune(ft_model,new_run_name,get_batch,cfg,pre_model)


    
    ...
    
if __name__ == '__main__':
    raise NotImplementedError("Please Don't run this file directly")
    SmartRunner(main,fire=False)










