# Transfer Learning

Transfer learning leverages knowledge from pre-trained models to solve new tasks with limited data.

## Table of Contents

1. [Introduction](#introduction)
2. [Pre-training Strategies](#pre-training-strategies)
3. [Fine-tuning Techniques](#fine-tuning-techniques)
4. [Domain Adaptation](#domain-adaptation)
5. [Few-Shot Learning](#few-shot-learning)
6. [Model Distillation](#model-distillation)

## Introduction

**Key Concepts:**
- **Pre-training**: Training on large dataset for general features
- **Fine-tuning**: Adapting pre-trained model to specific task
- **Feature Extraction**: Using pre-trained model as fixed feature extractor
- **Domain Shift**: Difference between source and target distributions

**When to Use Transfer Learning:**
- Limited target data
- Similar source and target tasks
- Computational constraints
- Need for faster convergence

## Pre-training Strategies

### Self-Supervised Pre-training

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms

# Contrastive Learning (SimCLR)
class SimCLR(nn.Module):
    def __init__(self, base_encoder, projection_dim=128):
        super(SimCLR, self).__init__()
        
        self.encoder = base_encoder
        
        # Remove classification head
        self.encoder.fc = nn.Identity()
        
        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )
    
    def forward(self, x):
        h = self.encoder(x)
        z = self.projection(h)
        return F.normalize(z, dim=1)

def contrastive_loss(z_i, z_j, temperature=0.5):
    """NT-Xent loss for contrastive learning"""
    batch_size = z_i.shape[0]
    
    # Concatenate representations
    z = torch.cat([z_i, z_j], dim=0)
    
    # Compute similarity matrix
    similarity_matrix = torch.matmul(z, z.T) / temperature
    
    # Create labels
    labels = torch.arange(batch_size, device=z.device)
    labels = torch.cat([labels + batch_size, labels])
    
    # Mask out self-similarity
    mask = torch.eye(2 * batch_size, device=z.device).bool()
    similarity_matrix = similarity_matrix.masked_fill(mask, -float('inf'))
    
    # Compute loss
    loss = F.cross_entropy(similarity_matrix, labels)
    
    return loss

# Training SimCLR
def train_simclr(model, train_loader, num_epochs=100):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
        total_loss = 0
        
        for (x1, x2), _ in train_loader:  # x1, x2 are augmented views
            optimizer.zero_grad()
            
            # Get representations
            z1 = model(x1)
            z2 = model(x2)
            
            # Compute loss
            loss = contrastive_loss(z1, z2)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")
    
    return model

# Create augmented pairs
class ContrastiveTransform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def __call__(self, x):
        return self.transform(x), self.transform(x)
```

### Masked Language Modeling (BERT-style)

```python
class MaskedLanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model=768, num_heads=12, num_layers=12):
        super(MaskedLanguageModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(512, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model, num_heads, dim_feedforward=3072)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.lm_head = nn.Linear(d_model, vocab_size)
    
    def forward(self, input_ids, attention_mask=None):
        # Embeddings
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0)
        
        embeddings = self.embedding(input_ids) + self.position_embedding(position_ids)
        
        # Transformer
        hidden_states = self.transformer(embeddings.transpose(0, 1)).transpose(0, 1)
        
        # Prediction
        logits = self.lm_head(hidden_states)
        
        return logits

def mask_tokens(inputs, tokenizer, mlm_probability=0.15):
    """Prepare masked tokens for MLM"""
    labels = inputs.clone()
    
    # Create random mask
    probability_matrix = torch.full(labels.shape, mlm_probability)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    
    # Only mask non-special tokens
    special_tokens_mask = tokenizer.get_special_tokens_mask(
        labels.tolist(), already_has_special_tokens=True
    )
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    
    # Set labels for non-masked tokens to -100
    labels[~masked_indices] = -100
    
    # Replace masked tokens
    # 80% [MASK], 10% random, 10% original
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.mask_token_id
    
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]
    
    return inputs, labels

# Training loop
def train_mlm(model, train_loader, tokenizer, num_epochs=10):
    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    for epoch in range(num_epochs):
        total_loss = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids']
            
            # Mask tokens
            masked_inputs, labels = mask_tokens(input_ids, tokenizer)
            
            # Forward pass
            logits = model(masked_inputs)
            
            # Compute loss
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")
```

## Fine-tuning Techniques

### Standard Fine-tuning

```python
def fine_tune_model(pretrained_model, train_loader, val_loader, num_classes, num_epochs=10):
    """Fine-tune pre-trained model on new task"""
    
    # Load pre-trained model
    model = models.resnet50(pretrained=True)
    
    # Replace classification head
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    # Optimizer with different learning rates
    params = [
        {'params': model.fc.parameters(), 'lr': 1e-3},  # New layer: higher LR
        {'params': [p for n, p in model.named_parameters() if 'fc' not in n], 
         'lr': 1e-4}  # Pre-trained layers: lower LR
    ]
    optimizer = optim.Adam(params)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_val_acc = 0
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = 100 * correct / total
        print(f"Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
    
    return model
```

### Progressive Unfreezing

```python
def progressive_unfreezing(model, train_loader, num_epochs=20, unfreeze_every=5):
    """Gradually unfreeze layers during fine-tuning"""
    
    # Initially freeze all layers
    for param in model.parameters():
        param.requires_grad = False
    
    # Only train classification head
    for param in model.fc.parameters():
        param.requires_grad = True
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # Get layer groups (from top to bottom)
    layer_groups = [
        model.fc,
        model.layer4,
        model.layer3,
        model.layer2,
        model.layer1
    ]
    
    for epoch in range(num_epochs):
        # Unfreeze next layer group
        if epoch % unfreeze_every == 0 and epoch > 0:
            group_idx = min(epoch // unfreeze_every, len(layer_groups) - 1)
            print(f"Unfreezing layer group {group_idx}")
            
            for param in layer_groups[group_idx].parameters():
                param.requires_grad = True
            
            # Update optimizer
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                                 lr=1e-3 / (2 ** group_idx))
        
        # Training
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
```

### Discriminative Learning Rates

```python
def get_discriminative_lr_params(model, base_lr=1e-3, lr_mult=2.6):
    """Different learning rates for different layers"""
    
    params = []
    
    # Get all layer names
    layer_names = [name for name, _ in model.named_parameters()]
    
    # Group layers
    num_layers = len(layer_names)
    
    for idx, (name, param) in enumerate(model.named_parameters()):
        # Exponentially decreasing learning rate from top to bottom
        layer_lr = base_lr * (lr_mult ** (num_layers - idx - 1))
        params.append({'params': param, 'lr': layer_lr})
    
    return params

# Usage
model = models.resnet50(pretrained=True)
params = get_discriminative_lr_params(model)
optimizer = optim.Adam(params)
```

### Adapter Layers

```python
class AdapterLayer(nn.Module):
    """Lightweight adapter for efficient fine-tuning"""
    def __init__(self, input_dim, bottleneck_dim=64):
        super(AdapterLayer, self).__init__()
        
        self.down_project = nn.Linear(input_dim, bottleneck_dim)
        self.up_project = nn.Linear(bottleneck_dim, input_dim)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        residual = x
        x = self.down_project(x)
        x = self.activation(x)
        x = self.up_project(x)
        return x + residual

class ModelWithAdapters(nn.Module):
    """Add adapters to pre-trained model"""
    def __init__(self, base_model, adapter_dim=64):
        super(ModelWithAdapters, self).__init__()
        
        self.base_model = base_model
        
        # Freeze base model
        for param in base_model.parameters():
            param.requires_grad = False
        
        # Add adapters after each transformer block
        self.adapters = nn.ModuleList([
            AdapterLayer(768, adapter_dim)  # Assuming 768 hidden dim
            for _ in range(12)  # For each layer
        ])
    
    def forward(self, x):
        # Forward through base model with adapters
        for i, (layer, adapter) in enumerate(zip(self.base_model.layers, self.adapters)):
            x = layer(x)
            x = adapter(x)
        
        return x
```

### LoRA (Low-Rank Adaptation)

```python
class LoRALayer(nn.Module):
    """Low-Rank Adaptation layer"""
    def __init__(self, input_dim, output_dim, rank=4, alpha=1):
        super(LoRALayer, self).__init__()
        
        self.rank = rank
        self.alpha = alpha
        
        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.randn(input_dim, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, output_dim))
        
        self.scaling = alpha / rank
    
    def forward(self, x):
        # Low-rank update: x @ A @ B
        return (x @ self.lora_A @ self.lora_B) * self.scaling

class LinearWithLoRA(nn.Module):
    """Linear layer with LoRA adaptation"""
    def __init__(self, linear_layer, rank=4):
        super(LinearWithLoRA, self).__init__()
        
        self.linear = linear_layer
        
        # Freeze original weights
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False
        
        # Add LoRA
        self.lora = LoRALayer(
            self.linear.in_features,
            self.linear.out_features,
            rank=rank
        )
    
    def forward(self, x):
        return self.linear(x) + self.lora(x)

def add_lora_to_model(model, rank=4):
    """Add LoRA to all linear layers"""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            
            parent = dict(model.named_modules())[parent_name] if parent_name else model
            setattr(parent, child_name, LinearWithLoRA(module, rank))
    
    return model
```

## Domain Adaptation

### Domain Adversarial Neural Network (DANN)

```python
class GradientReversalLayer(torch.autograd.Function):
    """Gradient reversal layer for domain adaptation"""
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None

class DomainAdversarialNetwork(nn.Module):
    def __init__(self, feature_extractor, num_classes, num_domains=2):
        super(DomainAdversarialNetwork, self).__init__()
        
        self.feature_extractor = feature_extractor
        
        # Label classifier
        self.label_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        # Domain classifier
        self.domain_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_domains)
        )
    
    def forward(self, x, alpha=1.0):
        # Extract features
        features = self.feature_extractor(x)
        
        # Label prediction
        label_pred = self.label_classifier(features)
        
        # Domain prediction with gradient reversal
        reversed_features = GradientReversalLayer.apply(features, alpha)
        domain_pred = self.domain_classifier(reversed_features)
        
        return label_pred, domain_pred

def train_dann(model, source_loader, target_loader, num_epochs=50):
    """Train DANN for domain adaptation"""
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    label_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        model.train()
        
        for (source_data, source_labels), (target_data, _) in zip(source_loader, target_loader):
            batch_size = source_data.size(0)
            
            # Compute alpha for gradient reversal
            p = float(epoch) / num_epochs
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            
            # Source domain
            source_label_pred, source_domain_pred = model(source_data, alpha)
            source_label_loss = label_criterion(source_label_pred, source_labels)
            source_domain_loss = domain_criterion(
                source_domain_pred,
                torch.zeros(batch_size, dtype=torch.long)
            )
            
            # Target domain
            _, target_domain_pred = model(target_data, alpha)
            target_domain_loss = domain_criterion(
                target_domain_pred,
                torch.ones(target_data.size(0), dtype=torch.long)
            )
            
            # Total loss
            loss = source_label_loss + source_domain_loss + target_domain_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
```

### Maximum Mean Discrepancy (MMD)

```python
def mmd_loss(source_features, target_features, kernel='rbf', gamma=1.0):
    """Compute MMD between source and target distributions"""
    
    def gaussian_kernel(x, y, gamma):
        """RBF kernel"""
        x_size = x.size(0)
        y_size = y.size(0)
        dim = x.size(1)
        
        x = x.unsqueeze(1)  # (x_size, 1, dim)
        y = y.unsqueeze(0)  # (1, y_size, dim)
        
        tiled_x = x.expand(x_size, y_size, dim)
        tiled_y = y.expand(x_size, y_size, dim)
        
        kernel_input = (tiled_x - tiled_y).pow(2).sum(2)
        return torch.exp(-gamma * kernel_input)
    
    # Compute kernels
    xx = gaussian_kernel(source_features, source_features, gamma).mean()
    yy = gaussian_kernel(target_features, target_features, gamma).mean()
    xy = gaussian_kernel(source_features, target_features, gamma).mean()
    
    # MMD
    return xx + yy - 2 * xy

class MMDDomainAdaptation(nn.Module):
    def __init__(self, feature_extractor, num_classes):
        super(MMDDomainAdaptation, self).__init__()
        self.feature_extractor = feature_extractor
        self.classifier = nn.Linear(512, num_classes)
    
    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.classifier(features)
        return output, features

def train_mmd(model, source_loader, target_loader, num_epochs=50, lambda_mmd=0.1):
    """Train with MMD for domain adaptation"""
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        for (source_data, source_labels), (target_data, _) in zip(source_loader, target_loader):
            optimizer.zero_grad()
            
            # Forward pass
            source_pred, source_features = model(source_data)
            _, target_features = model(target_data)
            
            # Classification loss
            class_loss = criterion(source_pred, source_labels)
            
            # MMD loss
            mmd = mmd_loss(source_features, target_features)
            
            # Total loss
            loss = class_loss + lambda_mmd * mmd
            
            loss.backward()
            optimizer.step()
```

## Few-Shot Learning

### Prototypical Networks

```python
class PrototypicalNetwork(nn.Module):
    def __init__(self, encoder):
        super(PrototypicalNetwork, self).__init__()
        self.encoder = encoder
    
    def forward(self, support_set, support_labels, query_set, n_way, k_shot):
        """
        support_set: (n_way * k_shot, C, H, W)
        query_set: (n_query, C, H, W)
        """
        # Encode support and query sets
        support_embeddings = self.encoder(support_set)
        query_embeddings = self.encoder(query_set)
        
        # Compute prototypes (class centroids)
        prototypes = []
        for c in range(n_way):
            class_embeddings = support_embeddings[c * k_shot:(c + 1) * k_shot]
            prototype = class_embeddings.mean(dim=0)
            prototypes.append(prototype)
        
        prototypes = torch.stack(prototypes)
        
        # Compute distances
        distances = torch.cdist(query_embeddings, prototypes)
        
        # Convert to probabilities
        log_p_y = F.log_softmax(-distances, dim=1)
        
        return log_p_y

def train_prototypical(model, train_loader, num_episodes=1000, n_way=5, k_shot=5):
    """Train prototypical network"""
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for episode in range(num_episodes):
        # Sample episode
        support_set, support_labels, query_set, query_labels = train_loader.sample_episode(n_way, k_shot)
        
        # Forward pass
        log_p_y = model(support_set, support_labels, query_set, n_way, k_shot)
        
        # Loss
        loss = F.nll_loss(log_p_y, query_labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if episode % 100 == 0:
            print(f"Episode {episode}, Loss: {loss.item():.4f}")
```

### MAML (Model-Agnostic Meta-Learning)

```python
class MAML:
    def __init__(self, model, inner_lr=0.01, outer_lr=0.001):
        self.model = model
        self.inner_lr = inner_lr
        self.outer_optimizer = optim.Adam(model.parameters(), lr=outer_lr)
    
    def inner_update(self, support_x, support_y, num_steps=5):
        """Adapt model on support set"""
        # Clone model for inner loop
        adapted_params = {name: param.clone() for name, param in self.model.named_parameters()}
        
        for step in range(num_steps):
            # Forward pass with adapted parameters
            logits = self.model(support_x)
            loss = F.cross_entropy(logits, support_y)
            
            # Compute gradients
            grads = torch.autograd.grad(loss, adapted_params.values(), create_graph=True)
            
            # Update adapted parameters
            adapted_params = {
                name: param - self.inner_lr * grad
                for (name, param), grad in zip(adapted_params.items(), grads)
            }
        
        return adapted_params
    
    def meta_update(self, tasks):
        """Meta-update on batch of tasks"""
        self.outer_optimizer.zero_grad()
        
        meta_loss = 0
        
        for support_x, support_y, query_x, query_y in tasks:
            # Inner loop: adapt to task
            adapted_params = self.inner_update(support_x, support_y)
            
            # Outer loop: evaluate on query set
            with torch.set_grad_enabled(True):
                query_logits = self.model.forward_with_params(query_x, adapted_params)
                task_loss = F.cross_entropy(query_logits, query_y)
                meta_loss += task_loss
        
        # Meta-gradient step
        meta_loss /= len(tasks)
        meta_loss.backward()
        self.outer_optimizer.step()
        
        return meta_loss.item()
```

## Model Distillation

Knowledge distillation transfers knowledge from large teacher to small student.

```python
class DistillationTrainer:
    def __init__(self, teacher_model, student_model, temperature=3.0, alpha=0.5):
        self.teacher = teacher_model
        self.student = student_model
        self.temperature = temperature
        self.alpha = alpha
        
        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        self.teacher.eval()
    
    def distillation_loss(self, student_logits, teacher_logits, labels):
        """Compute distillation loss"""
        # Hard loss (student vs true labels)
        hard_loss = F.cross_entropy(student_logits, labels)
        
        # Soft loss (student vs teacher)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean')
        soft_loss *= self.temperature ** 2
        
        # Combined loss
        loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss
        
        return loss
    
    def train(self, train_loader, num_epochs=10):
        """Train student with distillation"""
        optimizer = optim.Adam(self.student.parameters(), lr=0.001)
        
        for epoch in range(num_epochs):
            self.student.train()
            total_loss = 0
            
            for images, labels in train_loader:
                # Teacher predictions
                with torch.no_grad():
                    teacher_logits = self.teacher(images)
                
                # Student predictions
                student_logits = self.student(images)
                
                # Distillation loss
                loss = self.distillation_loss(student_logits, teacher_logits, labels)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

# Example usage
teacher = models.resnet50(pretrained=True)
student = models.resnet18(pretrained=False)

trainer = DistillationTrainer(teacher, student, temperature=3.0, alpha=0.5)
# trainer.train(train_loader, num_epochs=10)
```

## Practical Tips

1. **Start with Pre-trained Models**: Use ImageNet, BERT, GPT weights
2. **Learning Rate**: Use smaller LR for pre-trained layers
3. **Gradual Unfreezing**: Unfreeze layers progressively
4. **Data Augmentation**: Critical when fine-tuning with limited data
5. **Early Stopping**: Monitor validation to prevent overfitting
6. **Adapter Methods**: More efficient than full fine-tuning

## Resources

- Hugging Face Transformers: https://huggingface.co/transformers/
- timm (PyTorch Image Models): https://github.com/rwightman/pytorch-image-models
- "Transfer Learning" book by Tan et al.
- Papers with Code Transfer Learning: https://paperswithcode.com/task/transfer-learning

