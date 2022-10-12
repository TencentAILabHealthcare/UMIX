for((seed=0;seed<11;seed=seed+1))
do
python3 examples/run_expt.py \
        --dataset camelyon17 \
        --algorithm UMIX_trajectory \
        --root_dir data \
        --log_dir ./script/camelyon17/UMIX_trajectory/model \
        --seed $seed \
        --lr 0.001 \
        --weight_decay 0.01
done
