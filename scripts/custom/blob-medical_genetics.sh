#!/bin/bash

# BLoB training script for medical_genetics dataset
modelwrapper=blob
model="ContactDoctor/Bio-Medical-Llama-3-8B"
eps=0.05
beta=0.2
kllr=0.01
gamma=8

for dataset in medical_genetics; do
  for sample in 10; do
    for seed in 1; do
      name=$modelwrapper-$dataset-sample$sample-eps$eps-kllr$kllr-beta$beta-gamma$gamma-seed$seed

      CUDA_VISIBLE_DEVICES=0 python run/main.py \
        --dataset-type mcdataset --dataset $dataset \
        --model-type causallm --model $model --modelwrapper $modelwrapper \
        --load-in-8bit False \
        --lr 1e-4 --batch-size 4 \
        --opt adamw --warmup-ratio 0.06 \
        --max-seq-len 512 \
        --seed $seed \
        --evaluate \
        --wandb-entity jonathandomingue15-university-of-ottawa \
        --wandb-project BLoB-medical-model \
        --wandb-name $name \
        --log-path $name \
        --max-train-steps 100 \
        --eval-per-steps 50 \
        --bayes-klreweighting \
        --bayes-eps $eps --bayes-beta $beta --bayes-gamma $gamma --bayes-kllr $kllr --bayes-datasetrescaling \
        --bayes-train-n-samples 1 --bayes-eval-n-samples 1 --bayes-eval-n-samples-final $sample \
        --apply-classhead-lora --lora-r 8 --lora-alpha 16 --lora-dropout 0 \
        --checkpoint --checkpoint-name medical_genetics-blob
    done
  done
done