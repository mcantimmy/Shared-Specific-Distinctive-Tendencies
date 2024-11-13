import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from sklearn.model_selection import KFold
import numpy as np
from tqdm import tqdm
import psutil
import os

class RedditAuthorshipDataset(Dataset):
    def __init__(self, texts, authors, tokenizer, max_length=256):  # Increased max length
        self.texts = texts
        self.authors = authors
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text1, author1 = self.texts[idx], self.authors[idx]
        
        # Get multiple samples for each author for better contrastive learning
        positive_samples = []
        negative_samples = []
        
        # Get 2 positive and 2 negative samples
        author_indices = np.where(np.array(self.authors) == author1)[0]
        other_indices = np.where(np.array(self.authors) != author1)[0]
        
        pos_indices = np.random.choice(author_indices, size=2, replace=len(author_indices) < 2)
        neg_indices = np.random.choice(other_indices, size=2, replace=len(other_indices) < 2)
        
        for idx in pos_indices:
            positive_samples.append(self.texts[idx])
        for idx in neg_indices:
            negative_samples.append(self.texts[idx])

        encodings = []
        for text in [text1] + positive_samples + negative_samples:
            encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            encodings.append({
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten()
            })

        return {
            'anchor': encodings[0],
            'positives': encodings[1:3],
            'negatives': encodings[3:],
            'author': author1
        }

class ContrastiveAuthorshipModel(nn.Module):
    def __init__(self, model_name='microsoft/deberta-v3-large', freeze_base=False):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        
        if freeze_base:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        hidden_size = self.encoder.config.hidden_size
        
        # Improved projection head with larger dimensions
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256)
        )
        
        # Additional stylometric features extractor
        self.style_extractor = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 128)
        )
        
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids, attention_mask=attention_mask)
        pooled = self.mean_pooling(outputs, attention_mask)
        
        # Get both semantic and stylometric embeddings
        semantic_embedding = self.projection(pooled)
        style_embedding = self.style_extractor(pooled)
        
        # Combine embeddings
        final_embedding = torch.cat([
            nn.functional.normalize(semantic_embedding, p=2, dim=1),
            nn.functional.normalize(style_embedding, p=2, dim=1)
        ], dim=1)
        
        return final_embedding

def info_nce_loss(anchors, positives, negatives, temperature=0.07):
    # Compute similarities
    pos_similarities = []
    for pos in positives:
        sim = nn.functional.cosine_similarity(anchors.unsqueeze(1), pos.unsqueeze(0), dim=2) / temperature
        pos_similarities.append(sim)
    
    neg_similarities = []
    for neg in negatives:
        sim = nn.functional.cosine_similarity(anchors.unsqueeze(1), neg.unsqueeze(0), dim=2) / temperature
        neg_similarities.append(sim)
    
    # Concatenate all similarities
    logits = torch.cat([torch.cat(pos_similarities, dim=1), torch.cat(neg_similarities, dim=1)], dim=1)
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    
    # Cross entropy loss
    return nn.CrossEntropyLoss()(logits, labels)

def train(model, train_loader, val_loader, optimizer, device, epochs=5, patience=3):
    model.train()
    best_val_loss = float('inf')
    early_stopping_counter = 0
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    
    for epoch in range(epochs):
        start_memory = get_memory_usage()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            anchor_encoding = batch['anchor']
            positive_encodings = batch['positives']
            negative_encodings = batch['negatives']
            
            # Get embeddings for anchor
            anchor_embedding = model(
                anchor_encoding['input_ids'].to(device),
                anchor_encoding['attention_mask'].to(device)
            )
            
            # Get embeddings for positives
            positive_embeddings = []
            for pos in positive_encodings:
                pos_embed = model(pos['input_ids'].to(device), pos['attention_mask'].to(device))
                positive_embeddings.append(pos_embed)
                
            # Get embeddings for negatives
            negative_embeddings = []
            for neg in negative_encodings:
                neg_embed = model(neg['input_ids'].to(device), neg['attention_mask'].to(device))
                negative_embeddings.append(neg_embed)
            
            optimizer.zero_grad()
            loss = info_nce_loss(anchor_embedding, positive_embeddings, negative_embeddings)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
        
        end_memory = get_memory_usage()
        print(f"Memory Usage: {end_memory - start_memory:.2f} MB")
        
        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}")
        
        val_loss, val_accuracy = evaluate(model, val_loader, device)
        print(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            torch.save(model.state_dict(), 'best_authorship_model.pth')
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print(f"Early stopping triggered after epoch {epoch+1}")
                break
    
    return model

def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            anchor_encoding = batch['anchor']
            positive_encodings = batch['positives']
            negative_encodings = batch['negatives']
            
            anchor_embedding = model(
                anchor_encoding['input_ids'].to(device),
                anchor_encoding['attention_mask'].to(device)
            )
            
            positive_embeddings = []
            for pos in positive_encodings:
                pos_embed = model(pos['input_ids'].to(device), pos['attention_mask'].to(device))
                positive_embeddings.append(pos_embed)
                
            negative_embeddings = []
            for neg in negative_encodings:
                neg_embed = model(neg['input_ids'].to(device), neg['attention_mask'].to(device))
                negative_embeddings.append(neg_embed)
            
            loss = info_nce_loss(anchor_embedding, positive_embeddings, negative_embeddings)
            total_loss += loss.item()
            
            # Calculate accuracy
            pos_sim = nn.functional.cosine_similarity(anchor_embedding, positive_embeddings[0])
            neg_sim = nn.functional.cosine_similarity(anchor_embedding, negative_embeddings[0])
            correct += (pos_sim > neg_sim).sum().item()
            total += anchor_embedding.size(0)

    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total
    return avg_loss, accuracy

def predict_authorship(model, tokenizer, text1, text2, device, threshold=0.5):
    model.eval()
    encoding1 = tokenizer.encode_plus(
        text1,
        add_special_tokens=True,
        max_length=256,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    encoding2 = tokenizer.encode_plus(
        text2,
        add_special_tokens=True,
        max_length=256,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    with torch.no_grad():
        embedding1 = model(encoding1['input_ids'].to(device), encoding1['attention_mask'].to(device))
        embedding2 = model(encoding2['input_ids'].to(device), encoding2['attention_mask'].to(device))
        similarity = nn.functional.cosine_similarity(embedding1, embedding2).item()
    
    same_author = similarity > threshold
    return same_author, similarity

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # in MB

def estimate_memory_usage(model, batch_size, seq_length, dtype=torch.float32):
    def numel(model):
        return sum(p.numel() for p in model.parameters())

    model_params_memory = numel(model) * dtype.itemsize
    input_size = batch_size * seq_length
    activations_memory = input_size * model.encoder.config.hidden_size * dtype.itemsize * 2
    gradients_memory = model_params_memory
    optimizer_memory = model_params_memory * 2
    embedding_memory = batch_size * 2 * seq_length * model.encoder.config.hidden_size * dtype.itemsize
    attention_mask_memory = batch_size * 2 * seq_length * torch.bool.itemsize
    total_memory = (model_params_memory + activations_memory + gradients_memory + 
                    optimizer_memory + embedding_memory + attention_mask_memory)
    return total_memory / (1024 * 1024)

def main():
    dataset_size = 10000
    train_size = "train[:"+str(dataset_size)+"]"
    data = load_dataset("reddit", split=train_size, trust_remote_code=True)
    texts = data['content']
    authors = data['author']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    batch_size = 16  # Reduced batch size for larger model
    seq_length = 256
    
    tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-large')
    dataset = RedditAuthorshipDataset(texts, authors, tokenizer)

    model = ContrastiveAuthorshipModel(freeze_base=False).to(device)

    # Calculate expected resources
    estimated_memory = estimate_memory_usage(model, batch_size, seq_length)
    print(f"Estimated memory usage per batch: {estimated_memory:.2f} MB")

    dataset_size = 10000
    num_batches = dataset_size // batch_size
    total_estimated_memory = estimated_memory * num_batches
    print(f"Estimated total memory usage for one epoch: {total_estimated_memory:.2f} MB")

    n_splits = 2
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset), 1):
        print(f"Fold {fold}/{n_splits}")

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)

        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler)
        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_subsampler)

        optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)

        model = train(model, train_loader, val_loader, optimizer, device, epochs=2, patience=1)

    best_model = ContrastiveAuthorshipModel().to(device)
    best_model.load_state_dict(torch.load('best_authorship_model.pth'))

    text1 = """There were no Dark Ages! They didn't happen! Byzantium was happily being Byzantium..."""
    text2 = """I would associate the decline of the church largely to the loss of power..."""
    same_author, similarity = predict_authorship(best_model, tokenizer, text1, text2, device)
    print(f"Texts are by the same author: {same_author}")
    print(f"Similarity score: {similarity:.4f}")

if __name__ == "__main__":
    main()