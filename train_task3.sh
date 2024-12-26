#!/bin/bash
python train.py  --device cuda:0 --traj_num 1000 --T 60 --max_len 63 --task_type 2 --vocab_size 8909 --batch_size 64 --epochs 100 --learning_rate 0.1 --n_embd 64 --n_hidden 64 --n_layer 8 --n_head 16 --dropout 0.1 --weight_quantization_scale 30  --model_save_path weights/jinan/task3 --trajs_path data/jinan/node_traj_repeat_one_by_one/
