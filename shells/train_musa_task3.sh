#task3
python train.py  --device musa:2 --T 300 --max_len 303 --task_type 2 --vocab_size 8909 --batch_size 64 --epochs 100 --learning_rate 0.01 --n_embd 32 --n_hidden 16 --n_layer 8 --n_head 4 --dropout 0.1 --weight_quantization_scale 10  --model_save_path weights/jinan/task3 --trajs_path data/jinan/node_traj_repeat_one_by_one/
