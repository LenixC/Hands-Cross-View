"""
Evaluation script for trained hand matching model
Loads a checkpoint and evaluates on test set
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from pathlib import Path
import numpy as np
from PIL import Image
from typing import List
from tqdm import tqdm
import json
import pandas as pd
import random



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
        resnet = models.resnet50(pretrained=True)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        
        self.projection = ProjectionHead(output_dim=embedding_dim)
        
    def forward(self, x):
        features = self.encoder(x)
        features = features.view(features.size(0), -1)
        embeddings = self.projection(features)
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings



class HandRetrievalDataset(Dataset):
    """Dataset for retrieval evaluation (query-gallery setup)"""
    def __init__(self, data_root: Path, subject_ids: List[str], 
                 query_view='dorsal', gallery_view='palmar', transform=None):
        self.data_root = Path(data_root)
        self.subject_ids = set(subject_ids)
        self.query_view = query_view
        self.gallery_view = gallery_view
        self.transform = transform
        
        metadata_path = self.data_root / "HandInfo.csv"
        self.metadata = pd.read_csv(metadata_path)
        
        self.metadata = self.metadata[self.metadata['id'].isin(self.subject_ids)]
        
        self.queries = []
        self.gallery = []
        
        for subject_id in self.subject_ids:
            subject_data = self.metadata[self.metadata['id'] == subject_id]
            
            query_images = subject_data[subject_data['aspectOfHand'].str.contains(query_view, case=False, na=False)]['imageName'].tolist()
            
            gallery_images = subject_data[subject_data['aspectOfHand'].str.contains(gallery_view, case=False, na=False)]['imageName'].tolist()
            
            if len(query_images) > 0 and len(gallery_images) > 0:
                query_img = query_images[0]
                gallery_img = gallery_images[0]
                
                self.queries.append({
                    'path': self.data_root / 'Hands' / query_img, 
                    'subject': subject_id
                })
                self.gallery.append({
                    'path': self.data_root / 'Hands' / gallery_img, 
                    'subject': subject_id
                })
        
        print(f"Retrieval dataset: {len(self.queries)} queries, {len(self.gallery)} gallery items")
    
    def get_query_loader(self, batch_size=32):
        """Get DataLoader for query images"""
        dataset = SingleImageDataset(self.queries, self.transform)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    def get_gallery_loader(self, batch_size=32):
        """Get DataLoader for gallery images"""
        dataset = SingleImageDataset(self.gallery, self.transform)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)


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
    """
    Compute retrieval metrics: mean rank and Recall@K
    """
    similarities = query_embeddings @ gallery_embeddings.T
    
    ranks = []
    recall_at_k = {1: 0, 5: 0, 10: 0}
    
    for i, query_subject in enumerate(query_subjects):
        scores = similarities[i]
        sorted_indices = np.argsort(scores)[::-1]
        
        correct_match_rank = None
        for rank, idx in enumerate(sorted_indices, start=1):
            if gallery_subjects[idx] == query_subject:
                correct_match_rank = rank
                break
        
        if correct_match_rank is not None:
            ranks.append(correct_match_rank)
            
            for k in recall_at_k.keys():
                if correct_match_rank <= k:
                    recall_at_k[k] += 1
    
    mean_rank = np.mean(ranks)
    num_queries = len(query_subjects)
    
    metrics = {
        'mean_rank': mean_rank,
        'median_rank': np.median(ranks),
        'recall@1': recall_at_k[1] / num_queries,
        'recall@5': recall_at_k[5] / num_queries,
        'recall@10': recall_at_k[10] / num_queries,
        'num_queries': num_queries,
        'all_ranks': ranks
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
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate hand matching model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file (e.g., checkpoints/best_model.pth)')
    parser.add_argument('--data_root', type=str, default='./11k_hands_data',
                       help='Path to data root directory')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for evaluation')
    parser.add_argument('--output', type=str, default='test_results.json',
                       help='Output file for results')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print(f"\nLoading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    if 'config' in checkpoint:
        embedding_dim = checkpoint['config'].get('embedding_dim', 128)
        print(f"Loaded config from checkpoint: embedding_dim={embedding_dim}")
    else:
        embedding_dim = 128
        print("No config in checkpoint, using default embedding_dim=128")
    
    model = HandMatchingModel(embedding_dim=embedding_dim).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    if 'mean_rank' in checkpoint:
        print(f"Best validation mean rank: {checkpoint['mean_rank']:.2f}")
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print(f"\nLoading test data from {args.data_root}")
    metadata_path = Path(args.data_root) / "HandInfo.csv"
    metadata = pd.read_csv(metadata_path)
    
    all_subjects = metadata['id'].unique().tolist()
    print(f"Found {len(all_subjects)} unique subjects")
    
    random.seed(42)
    random.shuffle(all_subjects)
    
    n_subjects = len(all_subjects)
    n_train = int(0.68 * n_subjects)
    n_val = int(0.11 * n_subjects)
    
    test_subjects = all_subjects[n_train + n_val:]
    print(f"Test set: {len(test_subjects)} subjects")
    
    test_dataset = HandRetrievalDataset(
        args.data_root, test_subjects, transform=transform
    )
    
    print("\n" + "="*60)
    print("Evaluating on Test Set")
    print("="*60)
    
    metrics = evaluate_retrieval(model, test_dataset, device)
    
    print(f"\n{'='*60}")
    print("Test Results")
    print(f"{'='*60}")
    print(f"Number of queries: {metrics['num_queries']}")
    print(f"Mean Rank: {metrics['mean_rank']:.2f}")
    print(f"Median Rank: {metrics['median_rank']:.2f}")
    print(f"Recall@1: {metrics['recall@1']:.3f} ({metrics['recall@1']*100:.1f}%)")
    print(f"Recall@5: {metrics['recall@5']:.3f} ({metrics['recall@5']*100:.1f}%)")
    print(f"Recall@10: {metrics['recall@10']:.3f} ({metrics['recall@10']*100:.1f}%)")
    print(f"{'='*60}\n")
    
    ranks = metrics['all_ranks']
    print("Rank Distribution:")
    print(f"  Top 1: {sum(1 for r in ranks if r == 1)} queries ({sum(1 for r in ranks if r == 1)/len(ranks)*100:.1f}%)")
    print(f"  Top 5: {sum(1 for r in ranks if r <= 5)} queries ({sum(1 for r in ranks if r <= 5)/len(ranks)*100:.1f}%)")
    print(f"  Top 10: {sum(1 for r in ranks if r <= 10)} queries ({sum(1 for r in ranks if r <= 10)/len(ranks)*100:.1f}%)")
    print(f"  Worst rank: {max(ranks)}")
    
    output_metrics = {
        'mean_rank': float(metrics['mean_rank']),
        'median_rank': float(metrics['median_rank']),
        'recall@1': float(metrics['recall@1']),
        'recall@5': float(metrics['recall@5']),
        'recall@10': float(metrics['recall@10']),
        'num_queries': int(metrics['num_queries']),
        'checkpoint': args.checkpoint
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_metrics, f, indent=2)
    
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
