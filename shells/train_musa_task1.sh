#task1
python train.py --device musa:0 --T 150 --max_len 153 --task_type 0 --vocab_size 23313 --batch_size 256 --epochs 2000 --learning_rate 0.01 --n_embd 32 --n_hidden 16 --n_layer 8 --n_head 4  --dropout 0.01  --model_save_path weights/jinan/task1 --trajs_path data/jinan/edge_traj_new/
