#!/bin/bash
# Test different uncertainty metrics
# Compare: max_std (recommended), mean_std, entropy, bald

modelwrapper=blob
model="ContactDoctor/Bio-Medical-Llama-3-8B"
eps=0.05
beta=0.2
kllr=0.01
gamma=8
threshold=0.12  # Fixed threshold for fair comparison

echo "=========================================="
echo "Test 5: Uncertainty Metric Comparison"
echo "=========================================="
echo "Comparing different uncertainty quantification methods"
echo "Fixed threshold=$threshold across all metrics"
echo ""

for dataset in usmle; do  # Just USMLE to save time
  for sample in 10; do
    for metric in max_std mean_std entropy bald; do
      for seed in 1; do
        name=$modelwrapper-$dataset-metric-$metric-th$threshold-seed$seed

        echo "Running: Metric=$metric"

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
          --deferral-metric $metric \
          --deferral-strategy exclude \
          --deferral-log-to-wandb

        echo "✓ Completed metric=$metric"
        echo ""
      done
    done
  done
done

echo "✓ All uncertainty metric tests complete"
echo ""
echo "Metrics tested:"
echo "  - max_std: Maximum class standard deviation (recommended)"
echo "  - mean_std: Average std across classes"
echo "  - entropy: Predictive entropy"
echo "  - bald: Bayesian Active Learning by Disagreement"
echo ""
echo "Compare in WandB to see which metric gives best coverage/accuracy tradeoff"
