for((seed=0;seed<3;seed=seed+1))
do
python3 examples/run_expt.py \
        --dataset civilcomments \
        --algorithm UMIX_trajectory \
        --root_dir data \
        --log_dir ./script/CivilComments/UMIX_trajectory/model \
        --seed $seed \
        --n_epochs 10 \
        --lr 1e-05 \
        --weight_decay 0.01 \
        --max_grad_norm 1.0
done