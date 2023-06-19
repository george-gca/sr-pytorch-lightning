#!/bin/bash

telegram-send "Execution of start_here.sh started"

source utils.sh

# ==================================================================
# region variables
# ------------------------------------------------------------------

models=(
  # "edsr"
  "srcnn"
)

# training params
enable_training=1
datasets_dir="/datasets"
epochs=10
gpu_to_use=0
losses="adaptive + lpips"
metrics="BRISQUE FLIP LPIPS MS-SSIM PSNR SSIM"
optimizer="ADAM"

# if train dataset is DIV2K, it will automatically download from HuggingFace datasets
# else, it will look for it in $datasets_dir/DATASET_NAME/
train_dataset="DIV2K"

# if train dataset is one of: DIV2K, Set5, Set14, B100, or Urban100, it will automatically
# download from HuggingFace datasets
# else, it will look for it in $datasets_dir/DATASET_NAME/
# eval_datasets="DIV2K Set5 Set14 B100 Urban100"
eval_datasets="Set5 Set14"

# model params
scale=4
patch_size=128

# log params
log_loss_every_n_epochs=2
check_val_every_n_epoch=5
send_telegram_msg=1

# enable prediction
enable_predict=1
enable_predict_whole_lr=1 # optional, enable if the prediction images are the size of patch
# paths must be like
# $datasets_dir/DATASET_1_NAME/*.png
# $datasets_dir/DATASET_2_NAME/*.png
predict_datasets="G10_downscaled_x2"

# endregion

# enable metrics (optinal, but only enable if prediction=1)
enable_metrics=1
original_datasets="G10_original"

# ==================================================================
# region configuring and running
# ------------------------------------------------------------------

losses_to_str="${losses//[ ]/}"

save_dir="X$scale"
save_dir+="_e_"$(printf "%04d" $epochs)
save_dir+="_p_"$(printf "%03d" $patch_size)
save_dir+="_${losses_to_str//[*+]/_}"
save_dir+="_$optimizer"
save_dir+="_${train_dataset//[ ]/_}"

echo -e "\nRunning using $gpu_to_use"
export CUDA_VISIBLE_DEVICES=$gpu_to_use
array_gpus=(${gpu_to_use//,/ })
n_gpus=${#array_gpus[@]}
echo "Number of gpus: $n_gpus"

SECONDS=0

for model in "${models[@]}"; do
  previous_time=$SECONDS
  upper_case_model_name=${model^^}

  if [ -n "$enable_training" ] ; then
    telegram-send "Training enabled and starting"
    python train.py \
        --accelerator gpu \
        --check_val_every_n_epoch $check_val_every_n_epoch \
        --datasets_dir $datasets_dir \
        --default_root_dir "experiments/$upper_case_model_name"_$save_dir \
        --devices -1 \
        --eval_datasets $eval_datasets \
        --log_level info \
        --log_loss_every_n_epochs $log_loss_every_n_epochs \
        --loggers tensorboard \
        --losses "$losses" \
        --max_epochs $epochs \
        --metrics $metrics \
        --metrics_for_pbar PSNR \
        --model $model \
        --optimizer $optimizer \
        --patch_size $patch_size \
        --save_results -1 \
        --save_results_from_epoch last \
        --scale_factor $scale \
        --train_datasets $train_dataset

    LogElapsedTime $(( $SECONDS - $previous_time )) "$model"_$save_dir $send_telegram_msg
  fi

  if [ -n "$enable_predict" ] ; then
    telegram-send "Predict enabled and starting"
    python predict.py \
        --accelerator gpu \
        --checkpoint "experiments/$upper_case_model_name"_$save_dir/checkpoints/last.ckpt \
        --datasets_dir $datasets_dir \
        --default_root_dir "experiments/$upper_case_model_name"_$save_dir \
        --devices -1 \
        --log_level info \
        --loggers tensorboard \
        --model $model \
        --predict_datasets $predict_datasets \
        --scale_factor $scale \
        ${enable_predict_whole_lr:+--predict_whole_img}

    LogElapsedTime $(( $SECONDS - $previous_time )) "$model"_$save_dir $send_telegram_msg
  fi

  if [ -n "$enable_metrics" ] ; then
    telegram-send "Tracking metrics"
    python metrics.py \
        --datasets_dir $datasets_dir \
        --default_root_dir "experiments/$upper_case_model_name"_$save_dir \
        --predict_datasets $predict_datasets \
        --original_datasets $original_datasets

    LogElapsedTime $(( $SECONDS - $previous_time )) "$model"_$save_dir $send_telegram_msg
  fi
  telegram-send "Tracking metrics finished"

done

telegram-send "Execution of start_here.sh finished"
