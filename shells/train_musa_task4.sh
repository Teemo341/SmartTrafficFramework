#task4
python train.py --device musa:3 --T 48 --max_len 49 --task_type 3 --vocab_size 8909 --batch_size 64 --epochs 100 --learning_rate 0.005 --n_embd 16 --n_hidden 8 --n_layer 32 --dropout 0.0 --model_save_path weights/jinan/task4/ --trajs_path data/jinan/task4_data/
