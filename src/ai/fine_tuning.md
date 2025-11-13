# Fine-Tuning

Fine-tuning is the process of taking a pre-trained model and further training it on a specific task or dataset.

## Overview

Fine-tuning adapts a general-purpose model to a specific domain or task with much less data and compute than training from scratch.

**Approaches:**
- Full fine-tuning: Update all parameters
- Parameter-efficient: Update subset (LoRA, adapters)
- Few-shot prompting: No parameter updates

## When to Fine-Tune

✅ **Good use cases:**
- Domain-specific language (medical, legal)
- Specific output format requirements
- Improved performance on narrow tasks
- Style adaptation

❌ **Bad use cases:**
- General knowledge (use prompting)
- Limited data (< 100 examples)
- When prompting works well enough

## Fine-Tuning Process

```python
# Example with Hugging Face
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer

# 1. Load pre-trained model
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 2. Prepare dataset
train_dataset = load_dataset("your_dataset")

# 3. Configure training
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=2e-5,
)

# 4. Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()
```

## LoRA (Low-Rank Adaptation)

```python
from peft import LoraConfig, get_peft_model

# Configure LoRA
config = LoraConfig(
    r=8,                    # Rank
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
)

# Apply LoRA to model
model = get_peft_model(model, config)
model.print_trainable_parameters()
# trainable params: 0.1% (vs 100% for full fine-tuning)
```

## Data Preparation

```jsonl
{"prompt": "Translate to French: Hello", "completion": "Bonjour"}
{"prompt": "Translate to French: Goodbye", "completion": "Au revoir"}
{"prompt": "Translate to French: Thank you", "completion": "Merci"}
```

## Best Practices

1. **Start with quality data**: 100-1000 high-quality examples
2. **Use parameter-efficient methods**: LoRA for large models
3. **Monitor overfitting**: Validate on held-out data
4. **Experiment with hyperparameters**: Learning rate, batch size
5. **Evaluate systematically**: Don't just rely on loss
6. **Consider data augmentation**: Increase training data
7. **Version control**: Track model versions and data

## Evaluation

```python
from sklearn.metrics import accuracy_score

predictions = model.generate(test_inputs)
accuracy = accuracy_score(test_labels, predictions)
print(f"Accuracy: {accuracy}")
```

Fine-tuning enables customization of powerful pre-trained models for specific applications with minimal resources.
