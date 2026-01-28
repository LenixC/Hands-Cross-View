import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from pathlib import Path
import numpy as np
from PIL import Image
from typing import Tuple, List, Dict
import random
from tqdm import tqdm
import json
import pandas as pd


class ProjectionHead(nn.Module):
    """Projects ResNet features to embedding space: 2048 → 512 → 128"""
    def __init__(self, input_dim=2048, hidden_dim=512, output_dim=128):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.projection(x)


class HandMatchingModel(nn.Module):
    """Twin network with shared ResNet50 encoder + projection head"""
    def __init__(self, embedding_dim=128):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.projection = ProjectionHead(output_dim=embedding_dim)
        
    def forward(self, x):
        features = self.encoder(x)
        features = features.view(features.size(0), -1)
        embeddings = self.projection(features)
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings

class ContrastiveLoss(nn.Module):
    """Contrastive loss: pulls positive pairs together, pushes negatives apart"""
    def __init__(self, margin=2.0):
        super().__init__()
        self.margin = margin
    
    def forward(self, embedding1, embedding2, label):
        distance = torch.nn.functional.pairwise_distance(embedding1, embedding2)
        positive_loss = label * torch.pow(distance, 2)
        negative_loss = (1 - label) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
        loss = torch.mean(positive_loss + negative_loss)
        return loss

class HandPairDataset(Dataset):
    """Dataset for hand image pairs (dorsal-palmar matching)"""
    def __init__(self, data_root: Path, subject_ids: List[str], transform=None, 
                 positive_ratio=0.5):
        self.data_root = Path(data_root)
        self.subject_ids = set(subject_ids)
        self.transform = transform
        self.positive_ratio = positive_ratio
        
        metadata_path = self.data_root / "HandInfo.csv"
        self.metadata = pd.read_csv(metadata_path)
        self.metadata = self.metadata[self.metadata['id'].astype(str).isin(self.subject_ids)]
        
        self.subject_to_images = {}
        
        for subject_id in self.subject_ids:
            subject_data = self.metadata[self.metadata['id'].astype(str) == subject_id]
            dorsal_images = subject_data[subject_data['aspectOfHand'].str.contains('dorsal', case=False, na=False)]['imageName'].tolist()
            palmar_images = subject_data[subject_data['aspectOfHand'].str.contains('palmar', case=False, na=False)]['imageName'].tolist()
            
            if len(dorsal_images) > 0 and len(palmar_images) > 0:
                self.subject_to_images[subject_id] = {
                    'dorsal': dorsal_images,
                    'palmar': palmar_images
                }
        
        self.valid_subjects = list(self.subject_to_images.keys())
        print(f"Loaded {len(self.valid_subjects)} valid subjects with both dorsal and palmar images")
    
    def __len__(self):
        return len(self.valid_subjects) * 10
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, int]:
        is_positive = random.random() < self.positive_ratio
        
        if is_positive:
            subject = random.choice(self.valid_subjects)
            dorsal_img = random.choice(self.subject_to_images[subject]['dorsal'])
            palmar_img = random.choice(self.subject_to_images[subject]['palmar'])
            img1_path = self.data_root / 'Hands' / dorsal_img
            img2_path = self.data_root / 'Hands' / palmar_img
            label = 1
        else:
            subject1, subject2 = random.sample(self.valid_subjects, 2)
            view1 = random.choice(['dorsal', 'palmar'])
            view2 = random.choice(['dorsal', 'palmar'])
            img1 = random.choice(self.subject_to_images[subject1][view1])
            img2 = random.choice(self.subject_to_images[subject2][view2])
            img1_path = self.data_root / 'Hands' / img1
            img2_path = self.data_root / 'Hands' / img2
            label = 0
        
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        return img1, img2, label


class HandRetrievalDataset(Dataset):
    """Dataset for retrieval evaluation (query-gallery setup)"""
    def __init__(self, data_root: Path, subject_ids: List[str], 
                 query_view='dorsal', gallery_view='palmar', transform=None,
                 max_images_per_subject=None):
        self.data_root = Path(data_root)
        self.subject_ids = set(subject_ids)
        self.query_view = query_view
        self.gallery_view = gallery_view
        self.transform = transform
        self.max_images_per_subject = max_images_per_subject
        
        # Load metadata
        metadata_path = self.data_root / "HandInfo.csv"
        self.metadata = pd.read_csv(metadata_path)
        
        # Filter to only include subjects in our split
        self.metadata = self.metadata[self.metadata['id'].astype(str).isin(self.subject_ids)]
        
        # Build query and gallery lists - USE ALL IMAGES
        self.queries = []
        self.gallery = []
        
        for subject_id in self.subject_ids:
            subject_data = self.metadata[self.metadata['id'].astype(str) == subject_id]
            
            # Get ALL query images (e.g., dorsal)
            query_images = subject_data[subject_data['aspectOfHand'].str.contains(query_view, case=False, na=False)]['imageName'].tolist()
            
            # Get ALL gallery images (e.g., palmar)
            gallery_images = subject_data[subject_data['aspectOfHand'].str.contains(gallery_view, case=False, na=False)]['imageName'].tolist()
            
            # Limit number of images if specified
            if self.max_images_per_subject:
                query_images = query_images[:self.max_images_per_subject]
                gallery_images = gallery_images[:self.max_images_per_subject]
            
            # Add ALL images for this subject
            if len(query_images) > 0 and len(gallery_images) > 0:
                for query_img in query_images:
                    self.queries.append({
                        'path': self.data_root / 'Hands' / query_img, 
                        'subject': subject_id
                    })
                
                for gallery_img in gallery_images:
                    self.gallery.append({
                        'path': self.data_root / 'Hands' / gallery_img, 
                        'subject': subject_id
                    })
        
        print(f"Retrieval dataset: {len(self.queries)} queries ({len(self.subject_ids)} subjects), "
              f"{len(self.gallery)} gallery items ({len(self.subject_ids)} subjects)")
        print(f"Avg images per subject: Query={len(self.queries)/len(self.subject_ids):.1f}, "
              f"Gallery={len(self.gallery)/len(self.subject_ids):.1f}")
    
    def get_query_loader(self, batch_size=32):
        dataset = SingleImageDataset(self.queries, self.transform)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    def get_gallery_loader(self, batch_size=32):
        dataset = SingleImageDataset(self.gallery, self.transform)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)


class SingleImageDataset(Dataset):
    """Helper dataset for single images (used in retrieval)"""
    def __init__(self, image_list, transform=None):
        self.image_list = image_list
        self.transform = transform
    
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        item = self.image_list[idx]
        img = Image.open(item['path']).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img, item['subject']

def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    """Train for one epoch with optional mixed precision"""
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for img1, img2, labels in pbar:
        img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        if scaler is not None:
            with torch.amp.autocast(device_type='cuda'):
                emb1 = model(img1)
                emb2 = model(img2)
                loss = criterion(emb1, emb2, labels.float())
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            emb1 = model(img1)
            emb2 = model(img2)
            loss = criterion(emb1, emb2, labels.float())
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """Validate on validation set"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for img1, img2, labels in tqdm(dataloader, desc="Validation"):
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
            
            emb1 = model(img1)
            emb2 = model(img2)
            
            loss = criterion(emb1, emb2, labels.float())
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def extract_embeddings(model, dataloader, device):
    """Extract embeddings for all images in dataloader"""
    model.eval()
    embeddings = []
    subjects = []
    
    with torch.no_grad():
        for imgs, subj in tqdm(dataloader, desc="Extracting embeddings"):
            imgs = imgs.to(device)
            emb = model(imgs)
            embeddings.append(emb.cpu().numpy())
            subjects.extend(subj)
    
    embeddings = np.vstack(embeddings)
    return embeddings, subjects


def compute_retrieval_metrics(query_embeddings, query_subjects, 
                              gallery_embeddings, gallery_subjects):
    """Compute retrieval metrics with embedding aggregation per subject"""
    def aggregate_by_subject(embeddings, subjects):
        unique_subjects = []
        aggregated_embeddings = []
        
        subject_to_embeddings = {}
        for emb, subj in zip(embeddings, subjects):
            if subj not in subject_to_embeddings:
                subject_to_embeddings[subj] = []
            subject_to_embeddings[subj].append(emb)
        
        for subj in sorted(subject_to_embeddings.keys()):
            unique_subjects.append(subj)
            subject_embeddings = np.array(subject_to_embeddings[subj])
            avg_embedding = subject_embeddings.mean(axis=0)
            avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
            aggregated_embeddings.append(avg_embedding)
        
        return np.array(aggregated_embeddings), unique_subjects
    
    print("Aggregating query embeddings by subject...")
    query_agg, query_subjects_unique = aggregate_by_subject(query_embeddings, query_subjects)
    
    print("Aggregating gallery embeddings by subject...")
    gallery_agg, gallery_subjects_unique = aggregate_by_subject(gallery_embeddings, gallery_subjects)
    
    print(f"Aggregated to {len(query_subjects_unique)} unique query subjects "
          f"and {len(gallery_subjects_unique)} unique gallery subjects")
    
    similarities = query_agg @ gallery_agg.T
    
    ranks = []
    recall_at_k = {1: 0, 5: 0, 10: 0}
    
    for i, query_subject in enumerate(query_subjects_unique):
        scores = similarities[i]
        sorted_indices = np.argsort(scores)[::-1]
        
        correct_match_rank = None
        for rank, idx in enumerate(sorted_indices, start=1):
            if gallery_subjects_unique[idx] == query_subject:
                correct_match_rank = rank
                break
        
        if correct_match_rank is not None:
            ranks.append(correct_match_rank)
            
            for k in recall_at_k.keys():
                if correct_match_rank <= k:
                    recall_at_k[k] += 1
    
    mean_rank = np.mean(ranks)
    num_queries = len(query_subjects_unique)
    
    metrics = {
        'mean_rank': mean_rank,
        'recall@1': recall_at_k[1] / num_queries,
        'recall@5': recall_at_k[5] / num_queries,
        'recall@10': recall_at_k[10] / num_queries,
    }
    
    return metrics


def evaluate_retrieval(model, retrieval_dataset, device):
    """Evaluate retrieval performance"""
    query_loader = retrieval_dataset.get_query_loader(batch_size=32)
    gallery_loader = retrieval_dataset.get_gallery_loader(batch_size=32)
    
    query_embeddings, query_subjects = extract_embeddings(model, query_loader, device)
    gallery_embeddings, gallery_subjects = extract_embeddings(model, gallery_loader, device)
    
    metrics = compute_retrieval_metrics(
        query_embeddings, query_subjects,
        gallery_embeddings, gallery_subjects
    )
    
    return metrics

def main():
    config = {
        'data_root': './11k_hands_data',
        'embedding_dim': 128,
        'margin': 2.0,
        'batch_size': 64,
        'num_epochs': 30,
        'lr': 1e-4,
        'weight_decay': 1e-5,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'checkpoint_dir': './checkpoints',
        'seed': 42,
        'num_workers': 8,
        'prefetch_factor': 4,
    }
    
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    
    Path(config['checkpoint_dir']).mkdir(exist_ok=True, parents=True)
    
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    metadata_path = Path(config['data_root']) / "HandInfo.csv"
    metadata = pd.read_csv(metadata_path)
    
    all_subjects = metadata['id'].astype(str).unique().tolist()
    print(f"Found {len(all_subjects)} unique subjects in dataset")
    
    random.shuffle(all_subjects)
    
    n_subjects = len(all_subjects)
    n_train = int(0.68 * n_subjects)
    n_val = int(0.11 * n_subjects)
    
    train_subjects = all_subjects[:n_train]
    val_subjects = all_subjects[n_train:n_train + n_val]
    test_subjects = all_subjects[n_train + n_val:]
    
    print(f"Train: {len(train_subjects)} subjects")
    print(f"Val: {len(val_subjects)} subjects")
    print(f"Test: {len(test_subjects)} subjects")
    
    train_dataset = HandPairDataset(
        config['data_root'], train_subjects, transform=train_transform
    )
    val_dataset = HandPairDataset(
        config['data_root'], val_subjects, transform=val_transform
    )
    
    val_retrieval = HandRetrievalDataset(
        config['data_root'], val_subjects, transform=val_transform
    )
    test_retrieval = HandRetrievalDataset(
        config['data_root'], test_subjects, transform=val_transform
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], 
        shuffle=True, num_workers=config['num_workers'], 
        pin_memory=True, prefetch_factor=config['prefetch_factor'],
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'],
        shuffle=False, num_workers=config['num_workers'], 
        pin_memory=True, prefetch_factor=config['prefetch_factor'],
        persistent_workers=True
    )
    
    device = torch.device(config['device'])
    model = HandMatchingModel(embedding_dim=config['embedding_dim']).to(device)
    
    use_compile = False
    if use_compile and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model)
            print("Model compiled with torch.compile")
        except Exception as e:
            print(f"Could not compile model: {e}")
    
    criterion = ContrastiveLoss(margin=config['margin'])
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], 
                          weight_decay=config['weight_decay'])
    
    scaler = torch.amp.GradScaler('cuda') if config['device'] == 'cuda' else None
    if scaler:
        print("Using mixed precision training (AMP)")
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, 
    )
    
    # Training loop
    best_val_loss = float('inf')
    best_mean_rank = float('inf')
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        print(f"Train Loss: {train_loss:.4f}")
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        print(f"Val Loss: {val_loss:.4f}")
        
        # Evaluate retrieval (every 10 epochs to save time)
        if (epoch + 1) % 10 == 0:
            metrics = evaluate_retrieval(model, val_retrieval, device)
            print(f"Val Retrieval - Mean Rank: {metrics['mean_rank']:.2f}, "
                  f"R@1: {metrics['recall@1']:.3f}, "
                  f"R@5: {metrics['recall@5']:.3f}, "
                  f"R@10: {metrics['recall@10']:.3f}")
            
            # Save best model based on mean rank
            if metrics['mean_rank'] < best_mean_rank:
                best_mean_rank = metrics['mean_rank']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'mean_rank': best_mean_rank,
                    'config': config
                }, f"{config['checkpoint_dir']}/best_model.pth")
                print(f"Saved new best model (mean rank: {best_mean_rank:.2f})")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0 and val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'config': config
            }, f"{config['checkpoint_dir']}/checkpoint_epoch_{epoch+1}.pth")
    
    # Final evaluation on test set
    print("\n" + "="*50)
    print("Final Evaluation on Test Set")
    print("="*50)
    
    # Load best model
    checkpoint = torch.load(f"{config['checkpoint_dir']}/best_model.pth", weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = evaluate_retrieval(model, test_retrieval, device)
    print(f"Test Retrieval - Mean Rank: {test_metrics['mean_rank']:.2f}")
    print(f"Recall@1: {test_metrics['recall@1']:.3f}")
    print(f"Recall@5: {test_metrics['recall@5']:.3f}")
    print(f"Recall@10: {test_metrics['recall@10']:.3f}")
    
    # Save test results
    with open(f"{config['checkpoint_dir']}/test_results.json", 'w') as f:
        json.dump(test_metrics, f, indent=2)


if __name__ == "__main__":
    main()
