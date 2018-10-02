
dataset=$1
model_type=$2
batch_size
embedding_size
l2_lambda
learning_rate  batch_size topn num_preferences
dataset=$1
model_type=$2
gpu_id=$3

for lr in 0.0001 0.0005 0.001 0.005
do
    for reg0 in 1e-5 1e-4 1e-3 1e-2
    do
        dropout=0.1
        printf "\ntuning the regs=[reg0] for model_type=[bpr_mf]"
        printf "SGCMC.py: dataset=$dataset, gpu_id=$gpu_id, lr=$lr, regs=[$reg0], node_dropout=[$dropout], mess_dropout=[$dropout]\n"
        python SGCMC.py --dataset $dataset --regs [$reg0] --embed_size 64 --model_type $model_type --lr $lr --epoch 200 --verbose 10 --gpu_id $gpu_id --save_flag 0 --pretrain 1 --batch_size 1024 --node_dropout [$dropout] --mess_dropout [$dropout]
    done
done