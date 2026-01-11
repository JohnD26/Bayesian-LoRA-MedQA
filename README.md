# Bayesian PEFT
This repo contains the code for Bayesian LoRA Inspired on the following paper. Many adaptations were done to make the repo work on medical tasks.

We use the following custom shells:
blob-medmcqa.sh for the largest dataset
blob-usmle.sh  for the second medical dataset based on the USMLE medical exams in the USA

- **BLoB: Bayesian Low-Rank Adaptation by Backpropagation for Large Language Models**  
  Yibin Wang\*, Haizhou Shi\*, Ligong Han, Dimitris Metaxas, Hao Wang  
  *Thirty-eighth Conference on Neural Information Processing Systems, 2024*  
  [[📄 Paper](https://arxiv.org/abs/2406.11675)] [[🌐 OpenReview](https://openreview.net/forum?id=MaDykgj4Ru)] [[📑 Slides](https://nips.cc/media/neurips-2024/Slides/95507.pdf)] [[🖼️ Poster](https://nips.cc/media/PosterPDFs/NeurIPS%202024/95507.png)]

- **Training-Free Bayesianization for Low-Rank Adapters of Large Language Models**  
  Haizhou Shi\*, Yibin Wang\*, Ligong Han, Huan Zhang, Hao Wang  
  *ICLR Workshop: Quantify Uncertainty and Hallucination in Foundation Models, 2025*  
  [[📄 Paper](https://arxiv.org/abs/2412.05723)] [[🌐 OpenReview](https://openreview.net/forum?id=KlTOctRctg)]




## Installation
To install the required conda environment, run:
```sh
conda create --name <env>
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers datasets evaluate accelerate bitsandbytes jaxtyping torchmetrics setproctitle peft wandb nltk scikit-learn
```  

## To Run the Code
Before you run the code, there are a couple of settings you have to modify: 
- `wandb_entity`: at `utils/args.py` line 139, change to your own wandb account;

### To run the  experiment, use the following script:
```sh
bash scripts/custom/blob-usmle.sh
```

```sh
bash scripts/custom/blob-medmcqa.sh
```

## To Use the Code

### Overview of the WrapperBase Class
The `WrapperBase` class in `bayesian-peft/modelwrappers/wrapperbase.py` is designed as a flexible base class that integrates with various PEFT frameworks and datasets. Key features include:

* **Evaluation Metrics:** Includes accuracy, calibration error, negative log-likelihood, and Brier score.
* **Adapter Support:** Seamlessly integrates with the PEFT framework for parameter-efficient fine-tuning.
* **Optimizer and Scheduler:** Configurable optimizer and learning rate scheduler.
* **Training Loop:** Handles training and evaluation with built-in logging and metrics.

### Creating a Custom Wrapper
To implement a custom wrapper:

1. **Inherit from `WrapperBase`:** Your custom wrapper should subclass `WrapperBase`.
2. **Override `forward_logits`:** Implement how your model generates logits from input batches.
3. **Add Custom Behavior:** Extend or modify the existing methods to suit your needs.

Below is an example of creating a custom wrapper, CustomWrapper.

#### Step 1: Subclass `WrapperBase`
To create your custom wrapper, first subclass the `WrapperBase` class. This class manages training and evaluation routines, so when you create a custom wrapper, you can extend or modify any of the existing methods.

```python
from wrapperbase import WrapperBase

class CustomWrapper(WrapperBase):
    def __init__(self, model, peft_config, args, accelerator, adapter_name="default"):
        super().__init__(model, peft_config, args, accelerator, adapter_name)
        # Your custom initialization code
```

#### Step 2: Implement the `forward_logits` Method
The `forward_logits` method is used to define the forward pass for your model. It must return the logits (output) of the model, which are used to calculate the loss during training and for evaluation. Note that the `forward_logits` method is not implemented in the `WrapperBase` class; you need to implement it based on your specific requirements.
```python
def forward_logits(self, batch, sample=True, n_samples=1, **kwargs):
    # Custom logic to process the batch and return logits
    output = self.base_model(**batch)
    logits = output.logits
    return logits
```

#### Step 3: Add Custom Training and Evaluation Logic (Optional)
You can customize the training logic by overriding the `fit` method, which manages the training loop. You can modify how gradients are computed, how the model is updated, and how metrics are logged. The `evaluate` method handles the evaluation of your model. You can customize it to calculate additional metrics, apply different evaluation procedures, or modify how results are logged. You can also customize the `fit_evaluate` and `prepare_for_fit_evaluate` method to further control the procedure of training and evaluating.

For more information about the `WrapperBase` class, refer to the code provided in the project.

