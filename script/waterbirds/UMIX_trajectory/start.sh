for((seed=0;seed<11;seed=seed+1))
do
python3 examples/run_expt.py \
        --dataset waterbirds \
        --algorithm UMIX_trajectory \
        --root_dir data \
        --log_dir ./script/waterbirds/UMIX_trajectory/model \
        --seed $seed \
        --lr 0.00001 \
        --weight_decay 1 \
        --batch_size 64 \
        --n_epochs 300 \
        --log_every 50
done