#!/bin/bash

. utils.sh

# ==================================================================
# region variables
# ------------------------------------------------------------------

train_datasets=(
  "DIV2K"
  )

# size of the patch from the HR to be used during training
patch_sizes=(
  128
  )

scales=(
  # 2
  # 3
  4
  )

losses=(
  "l1"
  "adaptive"
  "lpips"
  "l1 + lpips"
  "adaptive + lpips"
  "adaptive + pencil_sketch"
  "adaptive + edge_loss"
  )

models_params=(
  "ddbpn DDBPN"
  "edsr EDSR_baseline --n_resblocks 16 --n_feats 64 --res_scale 0.1"
  "edsr EDSR --n_resblocks 32 --n_feats 256 --res_scale 0.1"
  "rdn RDN_ablation --rdn_config A"
  "rdn RDN --rdn_config B"
  "rcan RCAN --n_feats 64 --reduction 16 --n_resgroups 10 --n_resblocks 20"
  "srcnn SRCNN"
  "srresnet SRResNet"
  "wdsr WDSR_A --type A"
  "wdsr WDSR_B --type B"
)

optimizers=(
  # "SGD"
  "ADAM"
  # "RMSprop"
  # "Ranger"
  # "RangerVA"
  # "RangerQH"
)

## training params
batch_size=16
check_val_every_n_epoch=25
datasets_dir="/datasets"
epochs=2000
eval_datasets="DIV2K Set5 Set14 B100 Urban100"
log_level="info"
log_loss_every_n_epochs=10
metrics="BRISQUE FLIP LPIPS MS-SSIM PSNR SSIM"

## machine params
gpu_to_use=0
send_telegram_msg=1

# endregion

# ==================================================================
# region configure run string and model path based on variables above
# ------------------------------------------------------------------

export CUDA_VISIBLE_DEVICES=$gpu_to_use

base_run_string=" \
  python train.py \
    --check_val_every_n_epoch $check_val_every_n_epoch \
    --datasets_dir $datasets_dir \
    --deterministic True \
    --gpus -1 \
    --log_level $log_level \
    --log_loss_every_n_epochs $log_loss_every_n_epochs \
    --max_epochs $epochs \
    --metrics $metrics \
    --metrics_for_pbar PSNR \
    --save_results -1 \
    --save_results_from_epoch last \
    --weights_summary full"

# endregion

# ==================================================================
# region run grid search
# ------------------------------------------------------------------

possibilities="$((${#train_datasets[@]} *
      ${#patch_sizes[@]} *
      ${#models_params[@]} *
      ${#scales[@]} *
      ${#losses[@]} *
      ${#optimizers[@]}))"

printf "Testing %'d possibilities\n" $possibilities

if [ -n "$send_telegram_msg" ]; then
  telegram-send "$(printf "Testing %'d possibilities" $possibilities) at $HOSTNAME"
fi

test_number=0
SECONDS=0
previous_time=$SECONDS

for train_dataset in "${train_datasets[@]}"; do
  for patch_size in "${patch_sizes[@]}"; do
    for scale in "${scales[@]}"; do
      for loss in "${losses[@]}"; do
        loss="${loss//[ ]/}"

        for optimizer in "${optimizers[@]}"; do
          run_string="$base_run_string \
            --losses $loss \
            --optimizer $optimizer \
            --patch_size $patch_size \
            --scale_factor $scale \
            --train_datasets $train_dataset"

          save_dir="X$scale"
          save_dir+="_e_"$(printf "%04d" $epochs)
          save_dir+="_p_"$(printf "%03d" $patch_size)
          save_dir+="_${loss//[*+]/_}"
          save_dir+="_$optimizer"
          save_dir+="_${train_dataset//[ ]/_}"

          for model_params in "${models_params[@]}"; do
            params_array=($model_params)
            model=${params_array[0]}
            model_name=${params_array[1]}
            params=("${params_array[@]:2}")
            printf -v params ' %s' "${params[@]}"
            params=${params:1}

            test_number=$((test_number+1))
            echo ""
            LogTime "$(printf "Starting test %'d of %'d" $test_number $possibilities)"
            $run_string --model $model $params --default_root_dir "experiments/$model_name"_$save_dir
            LogTime "$(printf "Finished test %'d of %'d" $test_number $possibilities)"
          done

          LogElapsedTime $(( $SECONDS - $previous_time )) "x$scale for $epochs epochs" $send_telegram_msg
          previous_time=$SECONDS
        done
      done
    done
  done
done

# endregion
