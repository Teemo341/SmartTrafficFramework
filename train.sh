# python train.py --device cuda:1 --T 25 --max_len 50 --task_type 1 --vocab_size 8909 --batch_size 256 --epochs 100 --learning_rate 0.01 --n_embd 32 --n_hidden 32 --n_layer 10 --dropout 0.1 --model_read_path weights/jinan/task1/best_model_0.1352.pth --model_save_path weights/jinan/task1
# python train.py --task_type 0 
# python train.py --task_type 1 --vocab_size 101
# python train.py --task_type 2 --vocab_size 101


#jinan 

# task1
#python train.py --device cuda:2 --T 300 --max_len 303 --task_type 0 --vocab_size 23313 --batch_size 128 --epochs 2000 --learning_rate 0.01 --n_embd 32 --n_hidden 16 --n_layer 8 --n_head 4  --dropout 0.01  --model_save_path weights/jinan/task1 --trajs_path data/jinan/edge_traj_new/

#task2训练好了，权重为0.0294
#python train.py --device cuda:1 -- trajnum 200000 --T 200 --max_len 203 --task_type 1 --vocab_size 8909 --batch_size 256 --epochs 1000 --learning_rate 0.01 --n_embd 32 --n_hidden 8 --n_layer 4 --dropout 0.1 --adjcent data/jinan/adjcent_class.npy --model_save_path weights/jinan/task2/ --trajs_path data/jinan/traj_jinan_min_one_by_one/ 0.0476

#task3
#python train.py  --device cuda:3 --T 200 --max_len 203 --task_type 2 --vocab_size 8909 --batch_size 32 --epochs 100 --learning_rate 0.01 --n_embd 32 --n_hidden 16 --n_layer 8 --n_head 4 --dropout 0.1 --weight_quantization_scale 13  --model_save_path weights/jinan/task3 --trajs_path data/jinan/node_traj_repeat_one_by_one/

#12.8
#python train.py --device cuda:0 --T 150 --max_len 153 --task_type 0 --vocab_size 23313 --batch_size 256 --epochs 2000 --learning_rate 0.01 --n_embd 32 --n_hidden 16 --n_layer 8 --n_head 4  --dropout 0.01  --model_save_path weights/jinan/task1 --trajs_path data/jinan/edge_traj_new/
#12.8
python train.py  --device cuda:3 --T 300 --max_len 303 --task_type 2 --vocab_size 8909 --batch_size 128 --epochs 100 --learning_rate 0.01 --n_embd 32 --n_hidden 16 --n_layer 8 --n_head 4 --dropout 0.1 --weight_quantization_scale 10  --model_save_path weights/jinan/task3 --trajs_path data/jinan/node_traj_repeat_one_by_one/


#boston

#task2训练好了没必要再训了，两个权重0.04（num_embed为16）和0.01（num_embed为32）的都可以用
#python train.py --device cuda:3 --traj_num 100000 --T 48 --max_len 49 --task_type 1 --vocab_size 242 --batch_size 512 --epochs 5000 --learning_rate 0.001 --n_embd 16 --n_hidden 8 --n_layer 4 --dropout 0.0 --adjcent data/boston/adj_table_list.npy --model_save_path weights/boston/task2/ --trajs_path data/boston/traj_boston_min_one_by_one/ --model_read_path weights/boston/task2/best_model_0.0412.pth 