#!/bin/bash

#SBATCH --job-name=sbatch_jb
#SBATCH --partition=aida2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00


python train_xtal2dos.py \
    --concat_comp '' \
    --xtal2dos-loss-type 'KL' \
    --label_scaling 'normalized_sum' \
    --data_src 'binned_dos_128' \
    --trainset_subset_ratio 1.0 \
    --train \
    --xtal2dos-label-dim 128 \
    --sche lambda \
    --opt adam \
    --lr 1. \
    --warmup 300 \
    --T0 120 \
    --T_mult 1 \
    --eta_min 1e-4 \
    --num-epochs 200 \
    --c_epochs 121 \
    --batch-size 1024 \
    --d_model 512 \
    --graph_dropout 0.2 \
    --dec_dropout 0.0 \
    --dec_in_dim 256 \
    --dec_layers 6 \
    --temp 1.0 \
    --sum_scale 1. \
    --h 8 \
    --step_interval 30 \
    --lambda_scale 1.0 \
    --lambda_factor 1.0 \
    --accum_step 1 \
    --clip 1e-2 \
    --weight_decay 0. \
    --rate_decay 4. \
    --note ddp_transformer_A100 \
