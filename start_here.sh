#!/bin/bash

. utils.sh

# ==================================================================
# region variables
# ------------------------------------------------------------------

models=(
  # "edsr"
  "srcnn"
)

# training params
datasets_dir="/datasets"
train_dataset="DIV2K"
# eval_datasets="DIV2K Set5 Set14 B100 Urban100"
eval_datasets="Set5 Set14"
epochs=2000
losses="adaptive + lpips"
metrics="BRISQUE FLIP LPIPS MS-SSIM PSNR SSIM"
optimizer="ADAM"
gpu_to_use=0

# model params
scale=4
patch_size=128

# log params
log_loss_every_n_epochs=50
check_val_every_n_epoch=200
send_telegram_msg=1

# endregion

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

  python train.py \
      --check_val_every_n_epoch $check_val_every_n_epoch \
      --datasets_dir $datasets_dir \
      --default_root_dir "experiments/$model"_$save_dir \
      --eval_datasets $eval_datasets \
      --gpus -1 \
      --log_level info \
      --log_loss_every_n_epochs $log_loss_every_n_epochs \
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
done
