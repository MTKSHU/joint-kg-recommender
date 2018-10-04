embedding_size=$1
l2_lambda=$2
learning_rate=$3
topn=$4
num_preferences=$5

for embedding_size in 64 100 200
do
    for l2_lambda in 1e-5 1e-4 1e-3 1e-2
    do
        for learning_rate in 0.0001 0.0005 0.001 0.005
        do
            for topn in 5 10 20
            do
                for num_preferences in 4 10 20
                do
                    CUDA_VISIBLE_DEVICES=1 python run_item_recommendation.py -data_path ~/joint-kg-recommender/datasets/ -log_path ~/joint-kg-recommender/log/ -num_preferences $num_preferences -optimizer_type Adagrad -l2_lambda $l2_lambda -model_type transup -loss_type bpr -has_visualization -dataset ml1m -embedding_size $embedding_size -learning_rate $learning_rate -topn $topn
                done
            done
        done
    done
done