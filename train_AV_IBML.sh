#!/bin/bash
 
 
data_path=./data

gpu=0
modulation=Normal
add_noise=1
warmup=1
for dataset in CREMAD
do
for fusion_method in concat
  do
  runx=./runx_test_qy/${dataset}_${modulation}_${fusion_method}_add_noise${add_noise}_warmup${warmup} 
  # Train
  CUDA_VISIBLE_DEVICES=$gpu python main_AV.py --train --dataset ${dataset}  --add_noise $add_noise --epsilon 0  --learning_rate 0.001 --lr_decay_step 70 --batch_size 64 --gpu_ids $gpu  --data_path $data_path --ckpt_path $runx\
              --modulation $modulation --modulation_starts 0 --modulation_ends 90 --fusion_method $fusion_method --alpha 0.1  --warmup $warmup

  # Test
  CUDA_VISIBLE_DEVICES=$gpu python main_AV.py --dataset ${dataset} --batch_size 64 --gpu_ids $gpu --data_path $data_path  --ckpt_path $runx\
              --modulation $modulation --modulation_starts 1 --modulation_ends 70 --fusion_method $fusion_method --alpha 0.1
done
done 

 
