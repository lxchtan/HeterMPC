OUTPUT_PRE=FHeterBART_bos
PARAMS=config/hgt/FHeterBART_bos.json

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

if [[ $2 != "test" ]]; then
  CUDA_VISIBLE_DEVICES=${GPUID} python -u trainer.py --params_file ${PARAMS} \
      --output_path ${OUTPUT} --n_epochs 15 --train_batch_size 64 --valid_batch_size 64 --gradient_accumulation_steps 2 --seed $1
fi

if [[ $2 != "train" ]]; then
  # generate
  CUDA_VISIBLE_DEVICES=${GPUID} python generator.py --params_file ${PARAMS} \
    --generate_config config/basic/generate.json \
    --model_checkpoint runs/${OUTPUT} \
    --result_file ${RESULT}.txt
fi



