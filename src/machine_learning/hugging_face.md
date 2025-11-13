# Hugging Face Transformers

Comprehensive guide to using the Hugging Face ecosystem for NLP and beyond.

## Table of Contents

1. [Introduction](#introduction)
2. [Transformers Library](#transformers-library)
3. [Model Hub](#model-hub)
4. [Datasets Library](#datasets-library)
5. [Tokenizers](#tokenizers)
6. [Training and Fine-tuning](#training-and-fine-tuning)
7. [Inference and Deployment](#inference-and-deployment)

## Introduction

**Hugging Face Ecosystem:**
- **Transformers**: State-of-the-art NLP models
- **Datasets**: Easy access to datasets
- **Tokenizers**: Fast tokenization
- **Accelerate**: Distributed training
- **Optimum**: Hardware optimization

```bash
# Installation
pip install transformers datasets tokenizers accelerate
pip install torch torchvision torchaudio  # PyTorch
# OR
pip install tensorflow  # TensorFlow
```

## Transformers Library

### Basic Usage

```python
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    AutoModelForSequenceClassification,
    pipeline
)

# Quick start with pipelines
classifier = pipeline("sentiment-analysis")
result = classifier("I love using Hugging Face!")
print(result)
# [{'label': 'POSITIVE', 'score': 0.9998}]

# Multiple examples
results = classifier([
    "I love this!",
    "I hate this!",
    "This is okay."
])
print(results)
```

### Loading Models and Tokenizers

```python
# Load pre-trained model and tokenizer
model_name = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Tokenize text
text = "Hello, how are you?"
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

# Get embeddings
with torch.no_grad():
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    pooler_output = outputs.pooler_output

print(f"Last hidden state shape: {last_hidden_states.shape}")
print(f"Pooler output shape: {pooler_output.shape}")
```

### Common Model Types

```python
# Sequence Classification (e.g., sentiment analysis)
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
)

# Token Classification (e.g., NER)
from transformers import AutoModelForTokenClassification
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

# Question Answering
from transformers import AutoModelForQuestionAnswering
model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-cased-distilled-squad")

# Text Generation
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Masked Language Modeling
from transformers import AutoModelForMaskedLM
model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")

# Sequence-to-Sequence (e.g., translation, summarization)
from transformers import AutoModelForSeq2SeqLM
model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
```

### Pipelines

```python
# Sentiment Analysis
sentiment = pipeline("sentiment-analysis")
print(sentiment("This movie is great!"))

# Named Entity Recognition
ner = pipeline("ner", grouped_entities=True)
print(ner("My name is John and I live in New York"))

# Question Answering
qa = pipeline("question-answering")
context = "The Eiffel Tower is located in Paris, France."
question = "Where is the Eiffel Tower?"
print(qa(question=question, context=context))

# Text Generation
generator = pipeline("text-generation", model="gpt2")
print(generator("Once upon a time", max_length=50, num_return_sequences=2))

# Translation
translator = pipeline("translation_en_to_fr", model="t5-base")
print(translator("Hello, how are you?"))

# Summarization
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
article = """Long article text here..."""
print(summarizer(article, max_length=130, min_length=30))

# Zero-shot Classification
classifier = pipeline("zero-shot-classification")
text = "This is a course about Python programming"
labels = ["education", "politics", "business"]
print(classifier(text, candidate_labels=labels))

# Fill Mask
unmasker = pipeline("fill-mask", model="bert-base-uncased")
print(unmasker("The capital of France is [MASK]."))

# Feature Extraction
feature_extractor = pipeline("feature-extraction")
features = feature_extractor("Hello world!")
print(f"Features shape: {len(features[0])}")

# Image Classification
from transformers import pipeline
image_classifier = pipeline("image-classification", model="google/vit-base-patch16-224")
result = image_classifier("path/to/image.jpg")

# Object Detection
object_detector = pipeline("object-detection", model="facebook/detr-resnet-50")
results = object_detector("path/to/image.jpg")
```

### Custom Pipeline

```python
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch

class CustomSentimentPipeline:
    def __init__(self, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
    
    def __call__(self, texts, batch_size=8):
        if isinstance(texts, str):
            texts = [texts]
        
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=512
            )
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
            
            # Convert to results
            for j, prob in enumerate(probs):
                label_id = prob.argmax().item()
                score = prob[label_id].item()
                label = self.model.config.id2label[label_id]
                
                results.append({
                    'text': batch[j],
                    'label': label,
                    'score': score
                })
        
        return results

# Usage
custom_pipeline = CustomSentimentPipeline()
results = custom_pipeline(["I love this!", "I hate this!"])
print(results)
```

## Model Hub

### Searching and Filtering Models

```python
from huggingface_hub import HfApi, list_models

api = HfApi()

# List models
models = list_models(
    filter="text-classification",
    sort="downloads",
    direction=-1,
    limit=10
)

for model in models:
    print(f"{model.modelId}: {model.downloads} downloads")

# Search for specific models
models = list_models(search="bert", filter="fill-mask")
for model in models:
    print(model.modelId)
```

### Uploading Models

```python
from huggingface_hub import HfApi, create_repo

# Create repository
api = HfApi()
repo_url = create_repo(
    repo_id="username/model-name",
    token="your_token_here",
    private=False
)

# Upload model
api.upload_file(
    path_or_fileobj="path/to/model.bin",
    path_in_repo="model.bin",
    repo_id="username/model-name",
    token="your_token_here"
)

# Or use model.push_to_hub()
model.push_to_hub("username/model-name", token="your_token_here")
tokenizer.push_to_hub("username/model-name", token="your_token_here")
```

## Datasets Library

### Loading Datasets

```python
from datasets import load_dataset, load_metric

# Load popular datasets
dataset = load_dataset("glue", "mrpc")
print(dataset)

# Load specific split
train_dataset = load_dataset("imdb", split="train")
test_dataset = load_dataset("imdb", split="test")

# Load subset
small_train = load_dataset("imdb", split="train[:1000]")

# Stream large datasets
dataset = load_dataset("c4", "en", streaming=True)
for example in dataset:
    print(example)
    break

# Load from CSV
dataset = load_dataset("csv", data_files="path/to/file.csv")

# Load from JSON
dataset = load_dataset("json", data_files="path/to/file.json")

# Load from multiple files
dataset = load_dataset(
    "json", 
    data_files={
        "train": "train.json",
        "test": "test.json"
    }
)
```

### Dataset Operations

```python
from datasets import Dataset, DatasetDict

# Create custom dataset
data = {
    "text": ["Hello", "World", "!"],
    "label": [0, 1, 0]
}
dataset = Dataset.from_dict(data)

# Map function
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Filter
filtered_dataset = dataset.filter(lambda x: x["label"] == 1)

# Select
small_dataset = dataset.select(range(100))

# Shuffle
shuffled = dataset.shuffle(seed=42)

# Split
split_dataset = dataset.train_test_split(test_size=0.2)

# Sort
sorted_dataset = dataset.sort("label")

# Add column
dataset = dataset.map(lambda x: {"length": len(x["text"])})

# Remove columns
dataset = dataset.remove_columns(["length"])

# Save and load
dataset.save_to_disk("path/to/save")
loaded_dataset = Dataset.load_from_disk("path/to/save")
```

### Data Collators

```python
from transformers import DataCollatorWithPadding, DataCollatorForLanguageModeling

# Dynamic padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# For MLM (masked language modeling)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=True, 
    mlm_probability=0.15
)

# For sequence-to-sequence
from transformers import DataCollatorForSeq2Seq
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# Custom data collator
from dataclasses import dataclass
from typing import Dict, List
import torch

@dataclass
class CustomDataCollator:
    tokenizer: AutoTokenizer
    
    def __call__(self, features: List[Dict[str, any]]) -> Dict[str, torch.Tensor]:
        batch = {}
        
        # Extract and pad text
        texts = [f["text"] for f in features]
        tokenized = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        )
        
        batch.update(tokenized)
        
        # Add labels
        if "label" in features[0]:
            batch["labels"] = torch.tensor([f["label"] for f in features])
        
        return batch
```

## Tokenizers

### Using Tokenizers

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Basic tokenization
text = "Hello, how are you?"
tokens = tokenizer.tokenize(text)
print(f"Tokens: {tokens}")

# Encode (text to IDs)
input_ids = tokenizer.encode(text)
print(f"Input IDs: {input_ids}")

# Decode (IDs to text)
decoded = tokenizer.decode(input_ids)
print(f"Decoded: {decoded}")

# Full tokenization with special tokens
encoded = tokenizer(
    text,
    padding="max_length",
    truncation=True,
    max_length=128,
    return_tensors="pt"
)
print(f"Input IDs shape: {encoded['input_ids'].shape}")
print(f"Attention mask shape: {encoded['attention_mask'].shape}")

# Batch tokenization
texts = ["Hello!", "How are you?", "Nice to meet you."]
batch_encoded = tokenizer(
    texts,
    padding=True,
    truncation=True,
    return_tensors="pt"
)

# Token type IDs (for sentence pairs)
text_a = "This is sentence A"
text_b = "This is sentence B"
encoded = tokenizer(text_a, text_b, return_tensors="pt")
print(f"Token type IDs: {encoded['token_type_ids']}")
```

### Fast Tokenizers

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE, WordPiece
from tokenizers.trainers import BpeTrainer, WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace

# Create BPE tokenizer
tokenizer = Tokenizer(BPE())
tokenizer.pre_tokenizer = Whitespace()

# Train tokenizer
trainer = BpeTrainer(vocab_size=30000, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])
files = ["path/to/text1.txt", "path/to/text2.txt"]
tokenizer.train(files, trainer)

# Save tokenizer
tokenizer.save("path/to/tokenizer.json")

# Load tokenizer
from transformers import PreTrainedTokenizerFast
fast_tokenizer = PreTrainedTokenizerFast(tokenizer_file="path/to/tokenizer.json")
```

## Training and Fine-tuning

### Using Trainer API

```python
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification
from datasets import load_dataset, load_metric
import numpy as np

# Load dataset and model
dataset = load_dataset("glue", "mrpc")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples["sentence1"], examples["sentence2"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Define metrics
metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_dir="./logs",
    logging_steps=100,
    save_total_limit=2,
    fp16=True,  # Mixed precision training
    dataloader_num_workers=4,
    gradient_accumulation_steps=2,
    warmup_steps=500,
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    compute_metrics=compute_metrics,
)

# Train
trainer.train()

# Evaluate
results = trainer.evaluate()
print(results)

# Predict
predictions = trainer.predict(tokenized_datasets["test"])
print(predictions.metrics)

# Save model
trainer.save_model("./final_model")
```

### Custom Training Loop

```python
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm

def train_custom(model, train_dataset, eval_dataset, num_epochs=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=16)
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0.1 * total_steps,
        num_training_steps=total_steps
    )
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch in progress_bar:
            # Move to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
        
        # Evaluation
        model.eval()
        eval_loss = 0
        predictions, true_labels = [], []
        
        with torch.no_grad():
            for batch in eval_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                
                eval_loss += outputs.loss.item()
                
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1)
                
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(batch["labels"].cpu().numpy())
        
        # Compute metrics
        accuracy = np.mean(np.array(predictions) == np.array(true_labels))
        
        print(f"Epoch {epoch+1}:")
        print(f"  Train Loss: {total_loss/len(train_loader):.4f}")
        print(f"  Eval Loss: {eval_loss/len(eval_loader):.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
    
    return model
```

## Inference and Deployment

### Optimized Inference

```python
# Model optimization
from transformers import pipeline
from optimum.onnxruntime import ORTModelForSequenceClassification, ORTOptimizer
from optimum.onnxruntime.configuration import OptimizationConfig

# Convert to ONNX
model = ORTModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english",
    from_transformers=True
)

# Optimize
optimizer = ORTOptimizer.from_pretrained(model)
optimization_config = OptimizationConfig(optimization_level=2)
optimizer.optimize(save_dir="optimized_model", optimization_config=optimization_config)

# Use optimized model
optimized_pipeline = pipeline(
    "sentiment-analysis",
    model=model,
    tokenizer=tokenizer
)

# Quantization
from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig

quantizer = ORTQuantizer.from_pretrained(model)
qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False)
quantizer.quantize(save_dir="quantized_model", quantization_config=qconfig)
```

### Batch Inference

```python
def batch_predict(texts, model, tokenizer, batch_size=32):
    """Efficient batch prediction"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    all_predictions = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
        
        all_predictions.extend(predictions.cpu().numpy())
    
    return all_predictions

# Usage
texts = ["text1", "text2", "text3"] * 1000
predictions = batch_predict(texts, model, tokenizer)
```

### API Deployment

```python
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

# Load model once
classifier = pipeline("sentiment-analysis")

class TextRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    label: str
    score: float

@app.post("/predict", response_model=PredictionResponse)
def predict(request: TextRequest):
    result = classifier(request.text)[0]
    return PredictionResponse(label=result['label'], score=result['score'])

# Run with: uvicorn app:app --host 0.0.0.0 --port 8000
```

## Practical Tips

1. **Model Selection**: Choose based on task, speed, and accuracy requirements
2. **Tokenization**: Handle special characters and multiple languages carefully
3. **Batch Size**: Adjust based on GPU memory
4. **Mixed Precision**: Use fp16 for faster training
5. **Gradient Accumulation**: Simulate larger batch sizes
6. **Model Evaluation**: Use appropriate metrics for your task

## Resources

- Hugging Face Documentation: https://huggingface.co/docs
- Course: https://huggingface.co/course
- Model Hub: https://huggingface.co/models
- Datasets Hub: https://huggingface.co/datasets
- Forums: https://discuss.huggingface.co/

