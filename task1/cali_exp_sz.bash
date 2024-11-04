#!/bin/bash
set -e
# 验证实验
# Pre Train= ../Jinan/CK_bothmask_train.pkl = real5s_mask_train_jux.pkl + real5s_not_mask_train_jux.pkl

# Finetune = real5s_not_mask_val_jux.pkl

# Eval = real5s_mask_val_jux.pkl

YQ_PYTHON=~/anaconda3/envs/yq/bin/python 
cd ~/2024-intern/people/wjxie/v3 

EXPNAME=CaliB16V5 
CUDA_VISIBLE_DEVICES=4,5  $YQ_PYTHON trainer.py --use_enchead_ver 5 --n_layer 12 --n_head 8 --block_size 16 --n_embd 384 --n_hidden 384 --use_len_ratio 0 --ucbgc_beta 1.0 --batch_size 256 --max_grad_norm 100000.0 --learning_rate 0.0001 --max_iters 200000 --_split_tvs 1 --num_few_shots -1 --datapath ../Shenzhen/cali_jux.pkl --setting shenzhen --expname $EXPNAME --short_runname 1 --use_wandb 1 --debug 0 --save_model 1 --use_dp 1 --ddp_world_size 2 --grad_accumulation 1 # &> /dev/null 

CUDA_VISIBLE_DEVICES=5 $YQ_PYTHON eval.py -lmc 1 -mlf ./model/shenzhen/$EXPNAME/final.pth -d ../Shenzhen/real_test_jux.pkl --_split_tvs 0 --batch_size 512 --_eval_all 1 & #&> /dev/null &




# EXPNAME=Cali_FTKL0.1
CUDA_VISIBLE_DEVICES=4 $YQ_PYTHON trainer.py --is_ft 1 --use_kl_reg 1 --kl_reg_factor 0.1 --batch_size 512 --max_grad_norm 1000000 --learning_rate 1e-5 --max_iters 50000 --_split_tvs 0 -d ../Shenzhen/real_train_jux.pkl,../Shenzhen/real_val_jux.pkl -lmc 1 -mlf ./model/shenzhen/CaliB16V5/final.pth  --expname Cali_FTKL0.1 -s 1  &> /dev/null &

sleep 5

# EXPNAME=Cali_FTKL0.01
CUDA_VISIBLE_DEVICES=5 $YQ_PYTHON trainer.py --is_ft 1 --use_kl_reg 1 --kl_reg_factor 0.01 --batch_size 512 --max_grad_norm 1000000 --learning_rate 1e-5 --max_iters 50000 --_split_tvs 0 -d ../Shenzhen/real_train_jux.pkl,../Shenzhen/real_val_jux.pkl -lmc 1 -mlf ./model/shenzhen/CaliB16V5/final.pth  --expname Cali_FTKL0.01 -s 1  &> /dev/null &

wait



CUDA_VISIBLE_DEVICES=4 $YQ_PYTHON eval.py -lmc 1 -mlf ./model/shenzhen/Cali_FTKL0.1/final.pth -d ../Shenzhen/real_test_jux.pkl --_split_tvs 0 --batch_size 512 --_eval_all 1 &> /dev/null &
sleep 2
CUDA_VISIBLE_DEVICES=5 $YQ_PYTHON eval.py -lmc 1 -mlf ./model/shenzhen/Cali_FTKL0.01/final.pth -d ../Shenzhen/real_test_jux.pkl --_split_tvs 0 --batch_size 512 --_eval_all 1 &> /dev/null &

wait
echo "Done"


