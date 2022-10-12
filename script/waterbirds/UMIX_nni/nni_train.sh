while [[ ! -z $CHECK ]]; do
    PORT=$(( ( RANDOM % 60000 )  + 1025 ))
    CHECK=$(sudo netstat -ap | grep $PORT)
done

echo $PORT


python3 ./examples/run_expt.py \
        --dataset waterbirds \
        --algorithm UMIX \
        --root_dir data \
        --seed 1 \
        --batch_size 64 \
        --n_epochs 150 \
        --umix_mixup_type vanillamix \
        --NNI True \
        --umix_trajectory_path ./script/waterbirds/UMIX_trajectory/model/trajectory 