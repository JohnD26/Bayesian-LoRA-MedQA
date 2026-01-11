#!/bin/bash
# Test deferral with "always_predict" strategy (for research comparison)
# Logs uncertain cases but still makes predictions on all samples

modelwrapper=blob
model="ContactDoctor/Bio-Medical-Llama-3-8B"
eps=0.05
beta=0.2
kllr=0.01
gamma=8

echo "=========================================="
echo "Test 3: Deferral with 'always_predict' Strategy"
echo "=========================================="
echo "Strategy: Predict on all samples, but log uncertain ones"
echo "Use case: Research baseline comparison"
echo ""

for dataset in usmle medmcqa; do
  for sample in 10; do
    for threshold in 0.12; do
      for seed in 1; do
        name=$modelwrapper-$dataset-always-predict-th$threshold-seed$seed

        echo "Running: Dataset=$dataset, Threshold=$threshold (always_predict)"

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
          --wandb-project BLoB-deferral-tests \
          --wandb-name $name \
          --log-path $name \
          --max-train-steps 100 \
          --eval-per-steps 50 \
          --bayes-klreweighting \
          --bayes-eps $eps --bayes-beta $beta --bayes-gamma $gamma --bayes-kllr $kllr --bayes-datasetrescaling \
          --bayes-train-n-samples 1 --bayes-eval-n-samples 1 --bayes-eval-n-samples-final $sample \
          --apply-classhead-lora --lora-r 8 --lora-alpha 16 --lora-dropout 0 \
          --enable-deferral \
          --deferral-threshold $threshold \
          --deferral-metric max_std \
          --deferral-strategy always_predict \
          --deferral-log-to-wandb

        echo "✓ Completed"
        echo ""
      done
    done
  done
done

echo "✓ All 'always_predict' strategy tests complete"
echo "Expected: val_acc should match baseline (no exclusion)"
echo "But uncertain samples are logged to JSONL for analysis"
