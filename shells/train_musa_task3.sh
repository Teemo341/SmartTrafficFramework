#task3
python train.py --device musa:2  --T 200 --max_len 203 --task_type 2 --vocab_size 8909 --batch_size 256 --epochs 1000 --learning_rate 0.01 --n_embd 32 --n_hidden 8 --n_layer 4 --dropout 0.1 --adjcent data/jinan/adjcent_class.npy --model_save_path weights/jinan/task2/ --trajs_path data/jinan/traj_jinan_min_one_by_one/
