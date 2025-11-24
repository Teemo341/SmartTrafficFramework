#task1
# python train.py --device cuda:0 --T 150 --max_len 153 --task_type 0 --vocab_size 23313 --batch_size 256 --epochs 2000 --learning_rate 0.01 --n_embd 32 --n_hidden 16 --n_layer 8 --n_head 4  --dropout 0.01  --model_save_path weights/jinan/task1 --trajs_path data/jinan/edge_traj_new/
# task2
python train.py --device musa:1  --T 200 --max_len 203 --task_type 1 --vocab_size 8909 --batch_size 256 --epochs 1000 --learning_rate 0.01 --n_embd 32 --n_hidden 8 --n_layer 4 --dropout 0.1 --adjcent data/jinan/adjcent_class.npy --model_save_path weights/jinan/task2/ --trajs_path data/jinan/traj_jinan_min_one_by_one/
#task3
# python train.py  --device cuda:2 --T 300 --max_len 303 --task_type 2 --vocab_size 8909 --batch_size 64 --epochs 100 --learning_rate 0.01 --n_embd 32 --n_hidden 16 --n_layer 8 --n_head 4 --dropout 0.1 --weight_quantization_scale 10  --model_save_path weights/jinan/task3 --trajs_path data/jinan/node_traj_repeat_one_by_one/
#task4
# python train.py --device cuda:3 --T 48 --max_len 49 --task_type 3 --vocab_size 8909 --batch_size 64 --epochs 100 --learning_rate 0.005 --n_embd 16 --n_hidden 8 --n_layer 4 --dropout 0.0 --model_save_path weights/jinan/task4/ --trajs_path data/jinan/task4_data/
