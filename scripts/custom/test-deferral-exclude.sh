#!/bin/bash
# Test deferral with "exclude" strategy (RECOMMENDED for medical deployment)
# Defers high-uncertainty predictions to human experts, excluding them from metrics

modelwrapper=blob
model="ContactDoctor/Bio-Medical-Llama-3-8B"
eps=0.05
beta=0.2
kllr=0.01
gamma=8

echo "=========================================="
echo "Test 2: Deferral with 'exclude' Strategy"
echo "=========================================="
echo "Strategy: Exclude deferred samples from accuracy metrics"
echo "Use case: Real-world medical deployment"
echo ""

for dataset in usmle medmcqa; do
  for sample in 10; do
    for threshold in 0.08 0.12 0.15; do
      for seed in 1; do
        name=$modelwrapper-$dataset-exclude-th$threshold-seed$seed

        echo "Running: Dataset=$dataset, Threshold=$threshold, Samples=$sample"

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
          --deferral-strategy exclude \
          --deferral-log-to-wandb

        echo "✓ Completed threshold=$threshold"
        echo ""
      done
    done
  done
done

echo "✓ All 'exclude' strategy tests complete"
echo "Check JSONL logs in: checkpoints/blob/ContactDoctor/Bio-Medical-Llama-3-8B/{dataset}/deferred_samples.jsonl"
echo "Expected metrics in WandB:"
echo "  - val_coverage (% handled by model)"
echo "  - val_deferral_rate (% deferred to expert)"
echo "  - val_acc_on_handled (accuracy on non-deferred cases)"
echo "  - val_acc_if_all_predicted (baseline for comparison)"
