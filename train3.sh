
dataset_name=MovieLens
sampling_N=4
python train.py --dataset_name $dataset_name --sampling_N $sampling_N --log_pos_metrics --device cuda:3 --model GAG --lr 0.003 --weight_decay 0.01 --embed_dim 64 --hidden_dim 64 --n_layer 4 --dropout 0.2 
python train.py --dataset_name $dataset_name --sampling_N $sampling_N --log_pos_metrics --device cuda:3 --model H-RNN --lr 0.005 --weight_decay 0.01 --embed_dim 128 --hidden_dim 128 --n_layer 2 --dropout 0.3 


: << "END"
dataset_name=LFM-BeyMS
sampling_N=16
python train.py --dataset_name $dataset_name --sampling_N $sampling_N --log_pos_metrics --device cuda:5 --model GAG --lr 0.003 --weight_decay 0.01 --embed_dim 64 --hidden_dim 64 --n_layer 4 --dropout 0.2 
python train.py --dataset_name $dataset_name --sampling_N $sampling_N --log_pos_metrics --device cuda:5 --model H-RNN --lr 0.005 --weight_decay 0.01 --embed_dim 128 --hidden_dim 128 --n_layer 2 --dropout 0.3 
python train.py --dataset_name $dataset_name --sampling_N $sampling_N --log_pos_metrics --device cuda:5 --model HierTCN --lr 0.003 --weight_decay 0.01 --embed_dim 128 --hidden_dim 128 --n_layer 1 --dropout 0 
python train.py --dataset_name $dataset_name --sampling_N $sampling_N --log_pos_metrics --device cuda:5 --model NARM --lr 0.01 --weight_decay 0.01 --embed_dim 64 --hidden_dim 64 --n_layer 2 --dropout 0.3 
python train.py --dataset_name $dataset_name --sampling_N $sampling_N --log_pos_metrics --device cuda:5 --model SR-GNN --lr 0.01 --weight_decay 0.0001 --embed_dim 64 --hidden_dim 64 --n_layer 2 --dropout 0 
python train.py --dataset_name $dataset_name --sampling_N $sampling_N --log_pos_metrics --device cuda:5 --model TiSASRec --lr 0.01 --weight_decay 0.01 --embed_dim 64 --hidden_dim 64 --n_layer 1 --dropout 0.1 --train_strategy allseq --test_strategy lastonly
END