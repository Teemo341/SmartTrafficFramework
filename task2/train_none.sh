export CUDA_VISIBLE_DEVICES=3

if [ ! -d "./log" ]; then
    mkdir ./log
fi

if [ ! -d "./log/jinan" ]; then
    mkdir ./log/jinan
fi

python -u trainer_new.py --setting jinan --N 1 --batch_size 256 --use_agent_mask 0 --max_iters 5000 \
--eval_interval 50 --block_size 149 --vocab_size 8909 --root_path . --graph_path ./data/jinan/adj \
--data_path ./data/jinan/data --length_path ./data/jinan/valid_length --hop 0 --use_ge False \
--use_adjembed False --postprocess False --use_wandb False --dropout 0.1 --graph_embedding_mode none \
--od_per_graph 1000 --window_size 4 --num_file 100 --n_embed_adj 32 --n_embd 128 --n_hidden 128 --n_layer 6 \
--learning_rate 1e-3 --iter_per_epoch 100 >> ./log/jinan/none.txt