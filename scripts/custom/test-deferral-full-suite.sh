#!/bin/bash
# Comprehensive deferral test suite - runs all tests sequentially
# This master script runs all individual test scripts

echo "=========================================="
echo "DEFERRAL MECHANISM - FULL TEST SUITE"
echo "=========================================="
echo "This will run all deferral tests sequentially"
echo "Estimated time: ~2-4 hours depending on hardware"
echo ""
echo "Tests included:"
echo "  1. Backward compatibility (no deferral)"
echo "  2. Exclude strategy (deployment mode)"
echo "  3. Always predict strategy (research mode)"
echo "  4. Extreme thresholds (0.0 and 1.0)"
echo "  5. Uncertainty metric comparison"
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."

echo ""
echo "=========================================="
echo "Starting test suite..."
echo "=========================================="
echo ""

# Make all scripts executable
chmod +x scripts/custom/test-*.sh

# Test 1: Backward compatibility
echo ">>> Test 1/5: Backward Compatibility"
bash scripts/custom/test-backward-compatibility.sh
if [ $? -eq 0 ]; then
  echo "✓ Test 1 passed"
else
  echo "✗ Test 1 failed - stopping"
  exit 1
fi
echo ""

# Test 2: Exclude strategy
echo ">>> Test 2/5: Deferral with 'exclude' Strategy"
bash scripts/custom/test-deferral-exclude.sh
if [ $? -eq 0 ]; then
  echo "✓ Test 2 passed"
else
  echo "✗ Test 2 failed - stopping"
  exit 1
fi
echo ""

# Test 3: Always predict strategy
echo ">>> Test 3/5: Deferral with 'always_predict' Strategy"
bash scripts/custom/test-deferral-always-predict.sh
if [ $? -eq 0 ]; then
  echo "✓ Test 3 passed"
else
  echo "✗ Test 3 failed - stopping"
  exit 1
fi
echo ""

# Test 4: Extreme thresholds
echo ">>> Test 4/5: Extreme Threshold Verification"
bash scripts/custom/test-deferral-extreme-thresholds.sh
if [ $? -eq 0 ]; then
  echo "✓ Test 4 passed"
else
  echo "✗ Test 4 failed - stopping"
  exit 1
fi
echo ""

# Test 5: Uncertainty metrics
echo ">>> Test 5/5: Uncertainty Metric Comparison"
bash scripts/custom/test-deferral-uncertainty-metrics.sh
if [ $? -eq 0 ]; then
  echo "✓ Test 5 passed"
else
  echo "✗ Test 5 failed - stopping"
  exit 1
fi
echo ""

echo "=========================================="
echo "FULL TEST SUITE COMPLETE!"
echo "=========================================="
echo ""
echo "All tests passed successfully!"
echo ""
echo "Next steps:"
echo "  1. Check WandB dashboard: https://wandb.ai/jonathandomingue15-university-of-ottawa/BLoB-deferral-tests"
echo "  2. Review JSONL logs in: checkpoints/blob/.../deferred_samples.jsonl"
echo "  3. Analyze deferred questions to understand model limitations"
echo "  4. Tune threshold based on coverage/accuracy tradeoff"
echo ""
echo "Key metrics to compare:"
echo "  - coverage (% of cases handled by model)"
echo "  - deferral_rate (% deferred to expert)"
echo "  - val_acc_on_handled (accuracy on non-deferred cases)"
echo "  - val_acc_if_all_predicted (baseline comparison)"
echo ""
