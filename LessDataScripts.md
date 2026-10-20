## AWA2

### DRG 10%

```
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=3 python zerodiff_DRG_train.py \
  --dataset AWA2 --dataroot ./Dataset --class_embedding att --class_embedding_norm \
  --gzsl --encoded_noise --preprocessing --cuda --manualSeed 9182 \
  --nepoch 300 --eval_interval 1 --batch_size 64 \
  --ngh 4096 --ndh 4096 --lr 0.0005 --classifier_lr 0.001 --lambda1 10 --critic_iter 5 \
  --nclass_all 50 --noiseSize 85 --attSize 85 --resSize 2048 --dim_t 85 --n_T 4 \
  --gamma_ADV 10 --gamma_VAE 1.0 --gamma_recons 1.0 --freeze_dec \
  --gamma_x0 1.0 --gamma_xt 1.0 --gamma_dist 0.0 \
  --embed_type VA --split_percent 10 --syn_num 200
```

### DFG 10%

```
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=3 python zerodiff_DFG_train.py \
  --dataset AWA2 --dataroot ./Dataset --class_embedding att --class_embedding_norm \
  --gzsl --encoded_noise --preprocessing --cuda --manualSeed 9182 \
  --nepoch 300 --eval_interval 5 --batch_size 64 \
  --ngh 4096 --ndh 4096 --lr 0.0005 --classifier_lr 0.001 --lambda1 10 --critic_iter 5 \
  --nclass_all 50 --noiseSize 85 --attSize 85 --resSize 2048 --dim_t 85 --n_T 4 \
  --gamma_ADV 10 --gamma_VAE 1.0 --gamma_recons 1.0 --freeze_dec --dec_lr 0.0001 \
  --gamma_x0 1.0 --gamma_xt 1.0 --gamma_dist 5.0 --factor_dist 1.5 \
  --embed_type VA --split_percent 10 --syn_num 600 \
  --netR_model_path './out/AWA2/zerodiff_DRG_10percent_att:att_b:64_lr:0.0005_n_T:4_betas:0.1,20_gamma:ADV:10.0_VAE:1.0_x0:1.0_xt:1.0_dist:0.0_num:200_gzsl.tar'
```

### DRG 30%

```
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=3 python zerodiff_DRG_train.py \
  --dataset AWA2 --dataroot ./Dataset --class_embedding att --class_embedding_norm \
  --gzsl --encoded_noise --preprocessing --cuda --manualSeed 9182 \
  --nepoch 300 --eval_interval 1 --batch_size 64 \
  --ngh 4096 --ndh 4096 --lr 0.0005 --classifier_lr 0.001 --lambda1 10 --critic_iter 5 \
  --nclass_all 50 --noiseSize 85 --attSize 85 --resSize 2048 --dim_t 85 --n_T 4 \
  --gamma_ADV 10 --gamma_VAE 1.0 --gamma_recons 1.0 --freeze_dec \
  --gamma_x0 1.0 --gamma_xt 1.0 --gamma_dist 0.0 \
  --embed_type VA --split_percent 30 --syn_num 600
```

### DFG 30%

```
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=3 python zerodiff_DFG_train.py \
  --dataset AWA2 --dataroot ./Dataset --class_embedding att --class_embedding_norm \
  --gzsl --encoded_noise --preprocessing --cuda --manualSeed 9182 \
  --nepoch 300 --eval_interval 5 --batch_size 64 \
  --ngh 4096 --ndh 4096 --lr 0.0005 --classifier_lr 0.001 --lambda1 10 --critic_iter 5 \
  --nclass_all 50 --noiseSize 85 --attSize 85 --resSize 2048 --dim_t 85 --n_T 4 \
  --gamma_ADV 10 --gamma_VAE 1.0 --gamma_recons 1.0 --freeze_dec --dec_lr 0.0001 \
  --gamma_x0 1.0 --gamma_xt 1.0 --gamma_dist 5.0 --factor_dist 1.5 \
  --embed_type VA --split_percent 30 --syn_num 1800 \
  --netR_model_path './out/AWA2/zerodiff_DRG_30percent_att:att_b:64_lr:0.0005_n_T:4_betas:0.1,20_gamma:ADV:10.0_VAE:1.0_x0:1.0_xt:1.0_dist:0.0_num:600_gzsl.tar'
```

## CUB

### DRG 10%

```
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=3 python zerodiff_DRG_train.py \
  --dataset CUB --dataroot ./Dataset --class_embedding sent \
  --gzsl --encoded_noise --preprocessing --cuda --manualSeed 3483 \
  --nepoch 300 --eval_interval 1 --batch_size 64 \
  --ngh 4096 --ndh 4096 --lr 0.0001 --classifier_lr 0.001 --lambda1 10 --critic_iter 5 \
  --nclass_all 200 --noiseSize 1024 --attSize 1024 --resSize 2048 --dim_t 1024 --n_T 4 \
  --gamma_ADV 1 --gamma_VAE 0.0 --gamma_recons 1.0 --gamma_x0 1.0 --gamma_xt 1.0 --gamma_dist 1.0 \
  --embed_type VA --split_percent 10 --syn_num 40
```

### DFG 10%

```
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=3 python zerodiff_DFG_train.py \
  --dataset CUB --dataroot ./Dataset --class_embedding sent \
  --gzsl --encoded_noise --preprocessing --cuda --manualSeed 3483 \
  --nepoch 300 --eval_interval 5 --batch_size 64 \
  --ngh 4096 --ndh 4096 --lr 0.0001 --classifier_lr 0.001 --lambda1 10 --critic_iter 5 \
  --nclass_all 200 --noiseSize 1024 --attSize 1024 --resSize 2048 --dim_t 1024 --n_T 4 \
  --gamma_ADV 10 --gamma_VAE 1.0 --gamma_recons 0.01 --dec_lr 0.0001 \
  --gamma_x0 1.0 --gamma_xt 1.0 --gamma_dist 2.0 --factor_dist 1.5 \
  --embed_type VA --split_percent 10 --syn_num 140 \
  --netR_model_path './out/CUB/zerodiff_DRG_10percent_att:sent_b:64_lr:0.0001_n_T:4_betas:0.1,20_gamma:ADV:1.0_VAE:0.0_x0:1.0_xt:1.0_dist:1.0_num:40_gzsl.tar'
```

### DRG 30%

```
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=3 python zerodiff_DRG_train.py \
  --dataset CUB --dataroot ./Dataset --class_embedding sent \
  --gzsl --encoded_noise --preprocessing --cuda --manualSeed 3483 \
  --nepoch 300 --eval_interval 1 --batch_size 64 \
  --ngh 4096 --ndh 4096 --lr 0.0001 --classifier_lr 0.001 --lambda1 10 --critic_iter 5 \
  --nclass_all 200 --noiseSize 1024 --attSize 1024 --resSize 2048 --dim_t 1024 --n_T 4 \
  --gamma_ADV 1 --gamma_VAE 0.0 --gamma_recons 1.0 --gamma_x0 1.0 --gamma_xt 1.0 --gamma_dist 1.0 \
  --embed_type VA --split_percent 30 --syn_num 100
```

### DFG 30%

```
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=3 python zerodiff_DFG_train.py \
  --dataset CUB --dataroot ./Dataset --class_embedding sent \
  --gzsl --encoded_noise --preprocessing --cuda --manualSeed 3483 \
  --nepoch 300 --eval_interval 5 --batch_size 64 \
  --ngh 4096 --ndh 4096 --lr 0.0001 --classifier_lr 0.001 --lambda1 10 --critic_iter 5 \
  --nclass_all 200 --noiseSize 1024 --attSize 1024 --resSize 2048 --dim_t 1024 --n_T 4 \
  --gamma_ADV 10 --gamma_VAE 1.0 --gamma_recons 0.01 --dec_lr 0.0001 \
  --gamma_x0 1.0 --gamma_xt 1.0 --gamma_dist 2.0 --factor_dist 1.5 \
  --embed_type VA --split_percent 30 --syn_num 440 \
  --netR_model_path './out/CUB/zerodiff_DRG_30percent_att:sent_b:64_lr:0.0001_n_T:4_betas:0.1,20_gamma:ADV:1.0_VAE:0.0_x0:1.0_xt:1.0_dist:1.0_num:100_gzsl.tar'
```

## SUN

### DRG 10%

```
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=3 python zerodiff_DRG_train.py \
  --dataset SUN --dataroot ./Dataset --class_embedding att --class_embedding_norm \
  --gzsl --encoded_noise --preprocessing --cuda --manualSeed 4115 \
  --nepoch 400 --eval_interval 5 --batch_size 64 \
  --ngh 4096 --ndh 4096 --lr 0.0001 --classifier_lr 0.0005 --lambda1 10 --critic_iter 5 \
  --nclass_all 717 --noiseSize 102 --attSize 102 --resSize 2048 --dim_t 102 --n_T 4 \
  --gamma_ADV 1 --gamma_VAE 1.0 --gamma_recons 0.01 \
  --gamma_x0 1.0 --gamma_xt 1.0 --gamma_dist 0.0 --factor_dist 0.0 \
  --embed_type VA --split_percent 10 --syn_num 40
```

### DFG 10%

```
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=3 python zerodiff_DFG_train.py \
  --dataset SUN --dataroot ./Dataset --class_embedding att --class_embedding_norm \
  --gzsl --encoded_noise --preprocessing --cuda --manualSeed 4115 \
  --nepoch 400 --eval_interval 5 --batch_size 64 \
  --ngh 4096 --ndh 4096 --lr 0.0001 --classifier_lr 0.0005 --lambda1 10 --critic_iter 5 \
  --nclass_all 717 --noiseSize 102 --attSize 102 --resSize 2048 --dim_t 102 --n_T 4 \
  --gamma_ADV 1 --gamma_VAE 1.0 --gamma_recons 0.01 --dec_lr 0.0001 \
  --gamma_x0 1.0 --gamma_xt 1.0 --gamma_dist 1.0 --factor_dist 1.5 \
  --embed_type VA --split_percent 10 --syn_num 40 \
  --netR_model_path './out/SUN/zerodiff_DRG_10percent_att:att_b:64_lr:0.0001_n_T:4_betas:0.1,20_gamma:ADV:1.0_VAE:1.0_x0:1.0_xt:1.0_dist:0.0_num:40_gzsl.tar'
```

### DRG 30%

```
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=3 python zerodiff_DRG_train.py \
  --dataset SUN --dataroot ./Dataset --class_embedding att --class_embedding_norm \
  --gzsl --encoded_noise --preprocessing --cuda --manualSeed 4115 \
  --nepoch 400 --eval_interval 5 --batch_size 64 \
  --ngh 4096 --ndh 4096 --lr 0.0001 --classifier_lr 0.0005 --lambda1 10 --critic_iter 5 \
  --nclass_all 717 --noiseSize 102 --attSize 102 --resSize 2048 --dim_t 102 --n_T 4 \
  --gamma_ADV 1 --gamma_VAE 1.0 --gamma_recons 0.01 \
  --gamma_x0 1.0 --gamma_xt 1.0 --gamma_dist 0.0 --factor_dist 0.0 \
  --embed_type VA --split_percent 30 --syn_num 120
```

### DFG 30%

```
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=3 python zerodiff_DFG_train.py \
  --dataset SUN --dataroot ./Dataset --class_embedding att --class_embedding_norm \
  --gzsl --encoded_noise --preprocessing --cuda --manualSeed 4115 \
  --nepoch 400 --eval_interval 5 --batch_size 64 \
  --ngh 4096 --ndh 4096 --lr 0.0001 --classifier_lr 0.0005 --lambda1 10 --critic_iter 5 \
  --nclass_all 717 --noiseSize 102 --attSize 102 --resSize 2048 --dim_t 102 --n_T 4 \
  --gamma_ADV 1 --gamma_VAE 1.0 --gamma_recons 0.01 --dec_lr 0.0001 \
  --gamma_x0 1.0 --gamma_xt 1.0 --gamma_dist 1.0 --factor_dist 1.5 \
  --embed_type VA --split_percent 30 --syn_num 120 \
  --netR_model_path './out/SUN/zerodiff_DRG_30percent_att:att_b:64_lr:0.0001_n_T:4_betas:0.1,20_gamma:ADV:1.0_VAE:1.0_x0:1.0_xt:1.0_dist:0.0_num:120_gzsl.tar'
```