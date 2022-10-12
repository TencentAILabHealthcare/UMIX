for((seed=0;seed<3;seed=seed+1))
do
python3 examples/run_expt.py \
        --dataset celebA \
        --algorithm UMIX_trajectory \
        --root_dir data \
        --log_dir ./script/CelebA/UMIX_trajectory/model \
        --seed $seed \
        --lr 1e-5 \
        --weight_decay 1e-1 \
        --batch_size 128 \
        --n_epochs 50
done
