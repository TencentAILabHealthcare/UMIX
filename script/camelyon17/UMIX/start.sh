
for((seed=0;seed<11;seed=seed+1))
do
python3 examples/run_expt.py \
        --dataset camelyon17 \
        --algorithm UMIX \
        --root_dir data \
        --log_dir ./script/camelyon17/UMIX/model \
        --seed $seed \
        --n_epochs 5 \
        --lr 1e-5 \
        --umix_T_s 0 \
        --umxi_T 5 \
        --umix_alpha 0.5 \
        --umix_eta 5 \
        --umix_sigma 1 \
        --umix_mixup_type vanillamix \
        --umix_trajectory_path ./script/camelyon17/UMIX_trajectory/model/trajectory
done

python3 examples/evaluate.py \
    --predictions_dir ./script/camelyon17/UMIX/model \
    --output_dir ./script/camelyon17/UMIX/model \
    --root_dir data \
    --dataset camelyon17
