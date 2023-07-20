#!/bin/bash

. utils.sh

# ==================================================================
# region variables
# ------------------------------------------------------------------

models=(
  # "EDSR"
  "SRCNN"
)

# training params
check_val_every_n_epoch=5
enable_training=1
epochs=20
gpu_to_use=0
log_loss_every_n_epochs=2
losses="l1 + l2"
# metrics="BRISQUE FLIP LPIPS MS-SSIM PSNR SSIM"
metrics_for_pbar="PSNR"
metrics_for_save="Set14/PSNR"
optimizer="ADAM"

# if train dataset is DIV2K, it will automatically download from HuggingFace datasets
# else, it will look for it in $datasets_dir/DATASET_NAME/
eval_datasets="Set5 Set14"
train_dataset="DIV2K"

# model params
patch_size=128
scale=4

# log params
send_telegram_msg=1

# enable prediction
# enable_predict=1
# paths must be like
# $datasets_dir/DATASET_1_NAME/*.png
# $datasets_dir/DATASET_2_NAME/*.png
# predict_datasets="DATASET_1_NAME DATASET_2_NAME"

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

  if [ -n "$enable_training" ] ; then
    python main.py fit \
      --model $model \
      --config configs/train_default_sr.yml \
      --log_level info \
      --data.eval_datasets "[${eval_datasets//[ ]/, }]" \
      --data.patch_size $patch_size \
      --data.scale_factor $scale \
      --data.train_datasets=[$train_dataset] \
      --model.init_args.log_loss_every_n_epochs $log_loss_every_n_epochs \
      --model.init_args.losses "$losses" \
      --model.init_args.metrics_for_pbar "[${metrics_for_pbar//[ ]/, }]" \
      --model.init_args.optimizer $optimizer \
      --trainer.check_val_every_n_epoch $check_val_every_n_epoch \
      --trainer.default_root_dir "experiments/$model"_$save_dir \
      --trainer.callbacks.init_args.dirpath "experiments/$model"_$save_dir/checkpoints \
      --trainer.callbacks.init_args.filename "$model"_$save_dir \
      --trainer.callbacks.init_args.monitor $metrics_for_save \
      --trainer.logger.init_args.experiment_name "$model"_$save_dir \
      --trainer.logger.init_args.save_dir "experiments/$model"_$save_dir \
      --trainer.max_epochs $epochs

    LogElapsedTime $(( $SECONDS - $previous_time )) "$model"_$save_dir $send_telegram_msg
  fi

  # if [ -n "$enable_predict" ] ; then
  #   python predict.py \
  #       --accelerator gpu \
  #       --channels $channels \
  #       --checkpoint "experiments/$model"_$save_dir/checkpoints/last.ckpt \
  #       --datasets_dir $datasets_dir \
  #       --default_root_dir "experiments/$model"_$save_dir \
  #       --devices -1 \
  #       --log_level info \
  #       --loggers tensorboard \
  #       --model $model \
  #       --predict_datasets $predict_datasets \
  #       --scale_factor $scale

  #   LogElapsedTime $(( $SECONDS - $previous_time )) "$model"_$save_dir $send_telegram_msg
  # fi
done
