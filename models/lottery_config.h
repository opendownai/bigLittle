// lottery_config.h — maderix/ANE dynamic-pipeline model header
#pragma once

#define MODEL_NAME "Lottery"

#define DIM 64
#define HIDDEN 128
#define HEADS 4
#define KV_HEADS 4
#define HD (DIM / HEADS)
#define GQA_RATIO 1
#define Q_DIM (HEADS * HD)
#define KV_DIM (KV_HEADS * HD)
#define SEQ 64
#define NLAYERS 2
#define VOCAB 48

#define CKPT_PATH "ane_lottery_dyn_ckpt.bin"
#define DEFAULT_DATA_PATH "lottery_train.bin"

#define TRAIN_DEFAULT_STEPS 4000
#define TRAIN_DEFAULT_LR 3e-4f
#define TRAIN_DEFAULT_WEIGHT_DECAY 0.01f
#define TRAIN_DEFAULT_ACCUM_STEPS 10
#define TRAIN_DEFAULT_WARMUP_STEPS 200
#define TRAIN_DEFAULT_GRAD_CLIP 1.0f
#define TRAIN_DEFAULT_MIN_LR_FRAC 0.0f
#define TRAIN_DEFAULT_SAVE_EVERY 500
