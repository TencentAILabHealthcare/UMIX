for((seed=0;seed<3;seed=seed+1))
do
python3 examples/run_expt.py \
        --dataset celebA \
        --algorithm UMIX \
        --root_dir data \
        --log_dir ./script/CelebA/UMIX/model \
        --seed $seed \
        --lr 1e-4 \
        --weight_decay 1e-4 \
        --batch_size 128 \
        --n_epochs 20 \
        --umix_eta 50 \
        --umix_sigma 0.5 \
        --umix_alpha 1.5 \
        --umix_T_s 0 \
        --umxi_T 5 \
        --umix_mixup_type cutmix \
        --umix_trajectory_path ./script/CelebA/UMIX_trajectory/model/trajectory
done

python3 examples/evaluate.py \
    --predictions_dir ./script/CelebA/UMIX/model \
    --output_dir ./script/CelebA/UMIX/model \
    --root_dir data \
    --dataset celebA