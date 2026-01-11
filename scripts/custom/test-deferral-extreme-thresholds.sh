#!/bin/bash
# Test extreme thresholds to verify deferral logic
# Threshold 0.0 should defer ALL samples
# Threshold 1.0 should defer NO samples

modelwrapper=blob
model="ContactDoctor/Bio-Medical-Llama-3-8B"
eps=0.05
beta=0.2
kllr=0.01
gamma=8

echo "=========================================="
echo "Test 4: Extreme Threshold Verification"
echo "=========================================="
echo "Testing edge cases to verify deferral logic"
echo ""

for dataset in usmle; do  # Just test on USMLE to save time
  for sample in 10; do
    for seed in 1; do

      # Test 1: Threshold = 0.0 (defer everything)
      echo "Test 4a: Threshold=0.0 (should defer ~100%)"
      name=$modelwrapper-$dataset-extreme-th0.0-seed$seed

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
        --deferral-threshold 0.0 \
        --deferral-metric max_std \
        --deferral-strategy exclude \
        --deferral-log-to-wandb

      echo "Expected: deferral_rate ≈ 1.0, coverage ≈ 0.0"
      echo ""

      # Test 2: Threshold = 1.0 (defer nothing)
      echo "Test 4b: Threshold=1.0 (should defer ~0%)"
      name=$modelwrapper-$dataset-extreme-th1.0-seed$seed

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
        --deferral-threshold 1.0 \
        --deferral-metric max_std \
        --deferral-strategy exclude \
        --deferral-log-to-wandb

      echo "Expected: deferral_rate ≈ 0.0, coverage ≈ 1.0"
      echo ""
    done
  done
done

echo "✓ Extreme threshold tests complete"
echo "Verify:"
echo "  - Threshold 0.0: deferral_rate ≈ 1.0 (all deferred)"
echo "  - Threshold 1.0: deferral_rate ≈ 0.0 (none deferred)"
echo "  - coverage + deferral_rate = 1.0 in both cases"
