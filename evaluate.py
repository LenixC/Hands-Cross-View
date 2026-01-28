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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


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
        
        metadata_path = self.data_root / "HandInfo.csv"
        self.metadata = pd.read_csv(metadata_path)
        
        self.metadata = self.metadata[self.metadata['id'].astype(str).isin(self.subject_ids)]
        
        self.queries = []
        self.gallery = []
        
        for subject_id in self.subject_ids:
            subject_data = self.metadata[self.metadata['id'].astype(str) == subject_id]
            
            query_images = subject_data[subject_data['aspectOfHand'].str.contains(query_view, case=False, na=False)]['imageName'].tolist()
            
            gallery_images = subject_data[subject_data['aspectOfHand'].str.contains(gallery_view, case=False, na=False)]['imageName'].tolist()
            
            if self.max_images_per_subject:
                query_images = query_images[:self.max_images_per_subject]
                gallery_images = gallery_images[:self.max_images_per_subject]
            
            if len(query_images) > 0 and len(gallery_images) > 0:
                for query_img in query_images:
                    self.queries.append({
                        'path': self.data_root / 'Hands' / query_img, 
                        'subject': str(subject_id)
                    })
                
                for gallery_img in gallery_images:
                    self.gallery.append({
                        'path': self.data_root / 'Hands' / gallery_img, 
                        'subject': str(subject_id)
                    })
        
        print(f"Retrieval dataset: {len(self.queries)} queries ({len(self.subject_ids)} subjects), "
              f"{len(self.gallery)} gallery items ({len(self.subject_ids)} subjects)")
        print(f"Avg images per subject: Query={len(self.queries)/len(self.subject_ids):.1f}, "
              f"Gallery={len(self.gallery)/len(self.subject_ids):.1f}")
    
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
    Compute retrieval metrics with embedding aggregation per subject.
    Multiple images per subject are averaged into a single embedding.
    """
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
    top_matches = []
    
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
            
            top_match_idx = sorted_indices[0]
            top_matches.append({
                'query_subject': query_subject,
                'predicted_subject': gallery_subjects_unique[top_match_idx],
                'similarity': scores[top_match_idx],
                'rank': correct_match_rank,
                'is_correct': correct_match_rank == 1
            })
    
    mean_rank = np.mean(ranks)
    num_queries = len(query_subjects_unique)
    
    metrics = {
        'mean_rank': mean_rank,
        'median_rank': np.median(ranks),
        'recall@1': recall_at_k[1] / num_queries,
        'recall@5': recall_at_k[5] / num_queries,
        'recall@10': recall_at_k[10] / num_queries,
        'num_queries': num_queries,
        'all_ranks': ranks,
        'top_matches': top_matches,
        'similarities': similarities,
        'query_subjects': query_subjects_unique,
        'gallery_subjects': gallery_subjects_unique
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
    
    metrics['query_embeddings'] = query_embeddings
    metrics['query_subjects_all'] = query_subjects
    metrics['gallery_embeddings'] = gallery_embeddings
    metrics['gallery_subjects_all'] = gallery_subjects
    
    return metrics


def visualize_results(metrics, output_dir='visualizations'):
    """Generate all visualizations"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    plot_rank_distribution(metrics['all_ranks'], output_dir)
    
    plot_cumulative_recall(metrics['all_ranks'], output_dir)
    
    plot_confusion_matrix(metrics, output_dir, top_k=10)
    
    plot_similarity_distribution(metrics, output_dir)
    
    plot_embeddings_tsne(metrics, output_dir)
    
    plot_performance_breakdown(metrics, output_dir)
    
    print(f"\nAll visualizations saved to {output_dir}/")


def plot_rank_distribution(ranks, output_dir):
    """Plot histogram of retrieval ranks"""
    plt.figure(figsize=(10, 6))
    
    max_rank = min(max(ranks), 50)
    bins = range(1, max_rank + 2)
    
    plt.hist(ranks, bins=bins, edgecolor='black', alpha=0.7, color='steelblue')
    plt.xlabel('Rank of Correct Match', fontsize=12)
    plt.ylabel('Number of Queries', fontsize=12)
    plt.title('Distribution of Retrieval Ranks', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    
    stats_text = f'Mean: {np.mean(ranks):.2f}\nMedian: {np.median(ranks):.0f}'
    plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rank_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_cumulative_recall(ranks, output_dir):
    """Plot cumulative recall at different k values"""
    plt.figure(figsize=(10, 6))
    
    max_k = min(max(ranks), 50)
    k_values = range(1, max_k + 1)
    recall_values = []
    
    for k in k_values:
        recall = sum(1 for r in ranks if r <= k) / len(ranks)
        recall_values.append(recall * 100)
    
    plt.plot(k_values, recall_values, linewidth=2.5, color='steelblue')
    plt.fill_between(k_values, recall_values, alpha=0.3, color='steelblue')
    
    for k in [1, 5, 10]:
        if k <= max_k:
            recall = recall_values[k-1]
            plt.plot(k, recall, 'ro', markersize=8)
            plt.annotate(f'R@{k}: {recall:.1f}%', 
                        xy=(k, recall), xytext=(10, -10),
                        textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.xlabel('k (Top-k Retrievals)', fontsize=12)
    plt.ylabel('Recall@k (%)', fontsize=12)
    plt.title('Cumulative Recall Curve', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xlim(1, max_k)
    plt.ylim(0, 105)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cumulative_recall.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(metrics, output_dir, top_k=10):
    """Plot confusion matrix for top-k most common subjects"""
    top_matches = metrics['top_matches']
    
    all_subjects = list(set([m['query_subject'] for m in top_matches]))
    subject_counts = {s: sum(1 for m in top_matches if m['query_subject'] == s) 
                     for s in all_subjects}
    top_subjects = sorted(subject_counts.keys(), key=lambda x: subject_counts[x], 
                         reverse=True)[:top_k]
    
    n = len(top_subjects)
    conf_matrix = np.zeros((n, n))
    
    for match in top_matches:
        if match['query_subject'] in top_subjects:
            i = top_subjects.index(match['query_subject'])
            if match['predicted_subject'] in top_subjects:
                j = top_subjects.index(match['predicted_subject'])
                conf_matrix[i, j] += 1
    
    row_sums = conf_matrix.sum(axis=1, keepdims=True)
    conf_matrix_norm = np.divide(conf_matrix, row_sums, 
                                 where=row_sums!=0, out=np.zeros_like(conf_matrix))
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(conf_matrix_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=[f'S{s}' for s in top_subjects],
                yticklabels=[f'S{s}' for s in top_subjects],
                cbar_kws={'label': 'Normalized Frequency'})
    
    plt.xlabel('Predicted Subject', fontsize=12)
    plt.ylabel('True Subject', fontsize=12)
    plt.title(f'Confusion Matrix (Top {top_k} Subjects)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_similarity_distribution(metrics, output_dir):
    """Plot distribution of similarity scores for correct vs incorrect matches"""
    top_matches = metrics['top_matches']
    
    correct_sims = [m['similarity'] for m in top_matches if m['is_correct']]
    incorrect_sims = [m['similarity'] for m in top_matches if not m['is_correct']]
    
    plt.figure(figsize=(10, 6))
    
    bins = np.linspace(0, 1, 50)
    plt.hist(correct_sims, bins=bins, alpha=0.6, label='Correct Matches (R@1)', 
             color='green', edgecolor='black')
    plt.hist(incorrect_sims, bins=bins, alpha=0.6, label='Incorrect Matches (R@1)', 
             color='red', edgecolor='black')
    
    plt.xlabel('Cosine Similarity', fontsize=12)
    plt.ylabel('Number of Queries', fontsize=12)
    plt.title('Similarity Score Distribution', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(axis='y', alpha=0.3)
    
    if correct_sims:
        plt.axvline(np.mean(correct_sims), color='green', linestyle='--', 
                   linewidth=2, label=f'Mean Correct: {np.mean(correct_sims):.3f}')
    if incorrect_sims:
        plt.axvline(np.mean(incorrect_sims), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean Incorrect: {np.mean(incorrect_sims):.3f}')
    
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / 'similarity_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_embeddings_tsne(metrics, output_dir, n_samples=500):
    """Visualize embeddings using t-SNE"""
    query_emb = metrics['query_embeddings']
    query_subj = metrics['query_subjects_all']
    gallery_emb = metrics['gallery_embeddings']
    gallery_subj = metrics['gallery_subjects_all']
    
    if len(query_emb) + len(gallery_emb) > n_samples:
        query_indices = np.random.choice(len(query_emb), 
                                        min(n_samples//2, len(query_emb)), 
                                        replace=False)
        gallery_indices = np.random.choice(len(gallery_emb), 
                                          min(n_samples//2, len(gallery_emb)), 
                                          replace=False)
        
        query_emb = query_emb[query_indices]
        query_subj = [query_subj[i] for i in query_indices]
        gallery_emb = gallery_emb[gallery_indices]
        gallery_subj = [gallery_subj[i] for i in gallery_indices]
    
    all_embeddings = np.vstack([query_emb, gallery_emb])
    
    pca = PCA(n_components=50)
    embeddings_pca = pca.fit_transform(all_embeddings)
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(embeddings_pca)
    
    query_2d = embeddings_2d[:len(query_emb)]
    gallery_2d = embeddings_2d[len(query_emb):]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    ax1.scatter(query_2d[:, 0], query_2d[:, 1], 
               alpha=0.6, s=30, c='blue', label='Query (dorsal)', edgecolors='k', linewidth=0.5)
    ax1.scatter(gallery_2d[:, 0], gallery_2d[:, 1], 
               alpha=0.6, s=30, c='red', label='Gallery (palmar)', edgecolors='k', linewidth=0.5)
    ax1.set_title('t-SNE: Query vs Gallery', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    unique_subjects = list(set(query_subj + gallery_subj))
    n_colors = min(20, len(unique_subjects))
    selected_subjects = np.random.choice(unique_subjects, n_colors, replace=False)
    
    colors = plt.cm.tab20(np.linspace(0, 1, n_colors))
    
    for i, subject in enumerate(selected_subjects):
        query_mask = [s == subject for s in query_subj]
        if any(query_mask):
            ax2.scatter(query_2d[query_mask, 0], query_2d[query_mask, 1],
                       alpha=0.7, s=50, c=[colors[i]], marker='o', 
                       edgecolors='k', linewidth=1)
        
        gallery_mask = [s == subject for s in gallery_subj]
        if any(gallery_mask):
            ax2.scatter(gallery_2d[gallery_mask, 0], gallery_2d[gallery_mask, 1],
                       alpha=0.7, s=50, c=[colors[i]], marker='s',
                       edgecolors='k', linewidth=1)
    
    ax2.set_title(f't-SNE: Sample of {n_colors} Subjects\n(circles=query, squares=gallery)', 
                  fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'embeddings_tsne.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_performance_breakdown(metrics, output_dir):
    """Plot performance metrics breakdown"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    recalls = {
        'R@1': metrics['recall@1'] * 100,
        'R@5': metrics['recall@5'] * 100,
        'R@10': metrics['recall@10'] * 100
    }
    
    bars = ax1.bar(recalls.keys(), recalls.values(), color=['#2E86AB', '#A23B72', '#F18F01'],
                   edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Recall (%)', fontsize=11)
    ax1.set_title('Recall Metrics', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 105)
    ax1.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    rank_stats = {
        'Mean Rank': metrics['mean_rank'],
        'Median Rank': metrics['median_rank'],
        'Min Rank': min(metrics['all_ranks']),
        'Max Rank': max(metrics['all_ranks'])
    }
    
    ax2.barh(list(rank_stats.keys()), list(rank_stats.values()), 
             color='steelblue', edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Rank', fontsize=11)
    ax2.set_title('Rank Statistics', fontsize=13, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    for i, (k, v) in enumerate(rank_stats.items()):
        ax2.text(v, i, f'  {v:.1f}', va='center', fontweight='bold')
    
    ranks = metrics['all_ranks']
    buckets = {
        'Rank 1': sum(1 for r in ranks if r == 1),
        'Ranks 2-5': sum(1 for r in ranks if 2 <= r <= 5),
        'Ranks 6-10': sum(1 for r in ranks if 6 <= r <= 10),
        'Ranks 11-20': sum(1 for r in ranks if 11 <= r <= 20),
        'Ranks >20': sum(1 for r in ranks if r > 20)
    }
    
    colors_pie = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
    wedges, texts, autotexts = ax3.pie(buckets.values(), labels=buckets.keys(), 
                                        autopct='%1.1f%%', colors=colors_pie,
                                        startangle=90, textprops={'fontsize': 9})
    ax3.set_title('Query Distribution by Rank', fontsize=13, fontweight='bold')
    
    ax4.axis('off')
    
    summary_data = [
        ['Total Queries', f"{metrics['num_queries']}"],
        ['Mean Rank', f"{metrics['mean_rank']:.2f}"],
        ['Median Rank', f"{metrics['median_rank']:.0f}"],
        ['', ''],
        ['Recall@1', f"{metrics['recall@1']*100:.1f}%"],
        ['Recall@5', f"{metrics['recall@5']*100:.1f}%"],
        ['Recall@10', f"{metrics['recall@10']*100:.1f}%"],
    ]
    
    table = ax4.table(cellText=summary_data, cellLoc='left',
                     colWidths=[0.6, 0.4], loc='center',
                     bbox=[0.1, 0.2, 0.8, 0.7])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    for i in range(len(summary_data)):
        cell = table[(i, 0)]
        cell.set_facecolor('#E8E8E8')
        cell.set_text_props(weight='bold')
        
        if i == 3:
            cell.set_facecolor('white')
            table[(i, 1)].set_facecolor('white')
    
    ax4.set_title('Performance Summary', fontsize=13, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_breakdown.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate hand matching model with visualizations')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file (e.g., checkpoints/best_model.pth)')
    parser.add_argument('--data_root', type=str, default='./11k_hands_data',
                       help='Path to data root directory')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for evaluation')
    parser.add_argument('--output', type=str, default='test_results.json',
                       help='Output file for results')
    parser.add_argument('--viz_dir', type=str, default='visualizations',
                       help='Directory to save visualizations')
    parser.add_argument('--no_viz', action='store_true',
                       help='Skip visualization generation')
    
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
    
    all_subjects = metadata['id'].astype(str).unique().tolist()
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
    
    if not args.no_viz:
        visualize_results(metrics, args.viz_dir)
    
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
    if not args.no_viz:
        print(f"Visualizations saved to {args.viz_dir}/")


if __name__ == "__main__":
    main()
