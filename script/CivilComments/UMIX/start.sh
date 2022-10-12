for((seed=0;seed<3;seed=seed+1))
do
python3 examples/run_expt.py \
        --dataset civilcomments \
        --algorithm UMIX \
        --root_dir data \
        --log_dir ./script/CivilComments/UMIX/model \
        --seed $seed \
        --n_epochs 5 \
        --data_parallel \
        --batch_size 128 \
        --max_grad_norm 1.0 \
        --lr 5e-5 \
        --weight_decay 0.0001 \
        --umix_eta 3 \
        --umix_sigma 1 \
        --umix_alpha 0.5 \
        --umix_T_s 0 \
        --umxi_T 4 \
        --umix_mixup_type manifoldmix \
        --evaluate_steps 100 \
        --umix_trajectory_path ./script/CivilComments/UMIX_trajectory/model/trajectory 
done

python3 examples/evaluate.py \
    --predictions_dir ./script/CivilComments/UMIX/model \
    --output_dir ./script/CivilComments/UMIX/model \
    --root_dir data \
    --dataset civilcomments


    