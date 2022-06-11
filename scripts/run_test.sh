python test_xtal2dos.py \
    --concat_comp '' \
    --xtal2dos-loss-type 'KL' \
    --label_scaling 'normalized_sum' \
    --data_src 'binned_dos_128' \
    --trainset_subset_ratio 1.0 \
    --xtal2dos-label-dim 128 \
    --sche cosine \
    --opt adam \
    --lr 2e-3 \
    --warmup 300 \
    --T0 100 \
    --T_mult 1 \
    --eta_min 1e-4 \
    --num-epochs 200 \
    --c_epochs 50 \
    --batch-size 128 \
    --graph_dropout 0.2 \
    --dec_dropout 0.0 \
    --dec_in_dim 256 \
    --temp 1.0 \
    --sum_scale 1. \
    --step_interval 30 \
    --lambda_scale 1.0 \
    --lambda_factor 1.0 \
    --accum_step 1 \
    --clip 1e-2 \
    --weight_decay 0. \
    --rate_decay 2. \
    --d_model 512 \
    --dec_layers 6 \
    --h 8 \
    --check-point-path \
    model_binned_dos_128_normalized_sum_KL_dropout-0.2-0.0_bs-1024_lr-0.001_adam_gpu-1_cosine-100-0.0001-1_ep-200_dec_6l_temp-1.0_wd-0.0_weighted-False-1.0_accum-1_h-8_d-512_clip-0.01_c-epochs-40_ddp_transformer_A100
