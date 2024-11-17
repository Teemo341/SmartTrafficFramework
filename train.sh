# python train.py --device cuda:1 --T 25 --max_len 50 --task_type 1 --vocab_size 8909 --batch_size 256 --epochs 100 --learning_rate 0.01 --n_embd 32 --n_hidden 32 --n_layer 10 --dropout 0.1 --model_read_path weights/jinan/task1/best_model_0.1352.pth --model_save_path weights/jinan/task1
# python train.py --task_type 0 
# python train.py --task_type 1 --vocab_size 101
# python train.py --task_type 2 --vocab_size 101
#python train.py --device cuda:3 --T 192 --max_len 193 --task_type 1 --vocab_size 8909 --batch_size 50 --epochs 1000 --learning_rate 0.01 --n_embd 32 --n_hidden 8 --n_layer 4 --dropout 0.0    --model_save_path weights/jinan/task2New/ --trajs_path data/jinan/traj_min_test/
#python train.py --device cuda:0 --T 192 --max_len 193 --task_type 0 --vocab_size 23313 --batch_size 50 --epochs 1000 --learning_rate 0.01 --n_embd 32 --n_hidden 16 --n_layer 8 --n_head 4  --dropout 0.0  --model_save_path weights/jinan/task1New --trajs_path data/jinan/edge_traj_test/
#python train.py  --device cuda:3 --T 60 --max_len 120 --task_type 2 --vocab_size 8909 --batch_size 32 --epochs 100 --learning_rate 0.001 --n_embd 64 --n_hidden 16 --n_layer 8 --n_head 16 --dropout 0.1 --weight_quantization_scale 30 --model_read_path weights/jinan/task3/best_model_1.5159.pth --model_save_path weights/jinan/task3 --trajs_path data/jinan/node_traj_ft/
#export CUDA_LAUNCH_BLOCKING=1
#python test3.py 

