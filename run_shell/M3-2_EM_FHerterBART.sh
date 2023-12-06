
PREV_CHECKPOINT="runs/5/FHeterBART_bos_DMPC_norm_13"

OUTPUT_PRE="5/EM_FHeterBART_bos_DMPC_norm_all_initialize_Latest"
PARAMS="config/hgt/EM_FHeterBART_bos_DMPC.json"
DATAPATH="data/MPC_data_5_10_15/5/5_train.json"

if [[ $1 == "43" ]]; then
  GPUID=0
elif [[ $1 == "13" ]]; then
  GPUID=1
elif [[ $1 == "77" ]]; then
  GPUID=2
elif [[ $1 == "91" ]]; then
  GPUID=3
fi

OUTPUT=${OUTPUT_PRE}_$1
RESULT=`date +%Y%m%d`_${OUTPUT}

#--EM_batch_size =[23056,46112,92224,]
#5 [461120]
#10 [495226]
#15 [489812]

if [[ $2 != "test" ]]; then
  CUDA_VISIBLE_DEVICES=${GPUID} python -u Train_EM_Latest.py --params_file ${PARAMS} \
      --output_path ${OUTPUT} --n_epochs 4 --em_batch_size 461120 --train_batch_size 64 --em_iterations 10 --valid_batch_size 512 --gradient_accumulation_steps 1 --seed $1 --resume_from ${OUTPUT} --model_checkpoint ${PREV_CHECKPOINT}  --dataset_path ${DATAPATH}
fi


if [[ $2 != "train" ]]; then
  # generate
  CUDA_VISIBLE_DEVICES=${GPUID} python generator.py --params_file ${PARAMS} \
    --generate_config config/basic/generate.json \
    --model_checkpoint runs/${OUTPUT} \
    --result_file ${RESULT}.txt
fi
