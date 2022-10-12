for((seed=0;seed<11;seed=seed+1))
do
python3 examples/run_expt.py \
        --dataset waterbirds \
        --algorithm UMIX \
        --root_dir data \
        --log_dir ./script/waterbirds/UMIX/model \
        --seed $seed \
        --lr 0.00001 \
        --weight_decay 1 \
        --batch_size 64 \
        --n_epochs 300 \
        --umix_eta 80 \
        --umix_sigma 0.5 \
        --umix_alpha 0.5 \
        --umix_T_s 50 \
        --umxi_T 50 \
        --umix_mixup_type vanillamix \
        --umix_trajectory_path ./script/waterbirds/UMIX_trajectory/model/trajectory 
done

python3 examples/evaluate.py \
    --predictions_dir ./script/waterbirds/UMIX/model \
    --output_dir ./script/waterbirds/UMIX/model \
    --root_dir data \
    --dataset waterbirds