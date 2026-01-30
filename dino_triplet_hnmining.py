import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
import numpy as np
from PIL import Image
from typing import Tuple, List, Dict
import random
from tqdm import tqdm
import json
import pandas as pd


class DINOv2Encoder(nn.Module):
    """DINOv2 ViT-B/14 encoder with projection head"""

    def __init__(self, embedding_dim=256, freeze_backbone=False):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.projection = nn.Sequential(
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, embedding_dim),
        )

    def forward(self, x):
        with torch.set_grad_enabled(self.backbone.training):
            features = self.backbone(x)

        embeddings = self.projection(features)

        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings


class TripletLoss(nn.Module):
    """Triplet loss with online hard negative mining"""

    def __init__(self, margin=0.5, mining_strategy="hard"):
        super().__init__()
        self.margin = margin
        self.mining_strategy = mining_strategy

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: [batch_size, embedding_dim]
            labels: [batch_size] - subject IDs (list or tensor)
        """
        if not isinstance(labels, torch.Tensor):
            unique_labels = list(set(labels))
            label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
            label_indices = torch.tensor(
                [label_to_idx[label] for label in labels], device=embeddings.device
            )
        else:
            label_indices = labels

        distances = torch.cdist(embeddings, embeddings, p=2)

        batch_size = embeddings.size(0)
        triplet_loss = 0.0
        num_triplets = 0

        for i in range(batch_size):
            anchor_label = label_indices[i]

            positive_mask = (label_indices == anchor_label).clone()
            positive_mask[i] = False

            if not positive_mask.any():
                continue

            negative_mask = label_indices != anchor_label

            if not negative_mask.any():
                continue

            positive_distances = distances[i][positive_mask]
            negative_distances = distances[i][negative_mask]

            if self.mining_strategy == "hard":
                hardest_positive_dist = positive_distances.max()
                hardest_negative_dist = negative_distances.min()

                loss = torch.clamp(
                    hardest_positive_dist - hardest_negative_dist + self.margin, min=0.0
                )
                triplet_loss += loss
                num_triplets += 1

            elif self.mining_strategy == "semi-hard":
                hardest_positive_dist = positive_distances.max()

                semi_hard_negatives = negative_distances[
                    (negative_distances > hardest_positive_dist)
                    & (negative_distances < hardest_positive_dist + self.margin)
                ]

                if len(semi_hard_negatives) > 0:
                    hardest_negative_dist = semi_hard_negatives.min()
                else:
                    hardest_negative_dist = negative_distances.min()

                loss = torch.clamp(
                    hardest_positive_dist - hardest_negative_dist + self.margin, min=0.0
                )
                triplet_loss += loss
                num_triplets += 1

            else:
                for pos_dist in positive_distances:
                    for neg_dist in negative_distances:
                        loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0.0)
                        if loss > 0:
                            triplet_loss += loss
                            num_triplets += 1

        if num_triplets == 0:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        return triplet_loss / num_triplets


class HandTripletDataset(Dataset):
    """Dataset that returns triplets (anchor, positive, negative) for training"""

    def __init__(self, data_root: Path, subject_ids: List[str], transform=None):
        self.data_root = Path(data_root)
        self.subject_ids = set(subject_ids)
        self.transform = transform

        metadata_path = self.data_root / "HandInfo.csv"
        self.metadata = pd.read_csv(metadata_path)
        self.metadata = self.metadata[
            self.metadata["id"].astype(str).isin(self.subject_ids)
        ]

        self.subject_to_images = {}

        for subject_id in self.subject_ids:
            subject_data = self.metadata[self.metadata["id"].astype(str) == subject_id]
            dorsal_images = subject_data[
                subject_data["aspectOfHand"].str.contains(
                    "dorsal", case=False, na=False
                )
            ]["imageName"].tolist()
            palmar_images = subject_data[
                subject_data["aspectOfHand"].str.contains(
                    "palmar", case=False, na=False
                )
            ]["imageName"].tolist()

            if len(dorsal_images) > 0 and len(palmar_images) > 0:
                self.subject_to_images[subject_id] = {
                    "dorsal": dorsal_images,
                    "palmar": palmar_images,
                }

        self.valid_subjects = list(self.subject_to_images.keys())
        print(
            f"Loaded {len(self.valid_subjects)} valid subjects with both dorsal and palmar images"
        )

    def __len__(self):
        return len(self.valid_subjects) * 20

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        anchor_subject = random.choice(self.valid_subjects)

        anchor_view = random.choice(["dorsal", "palmar"])
        anchor_img = random.choice(self.subject_to_images[anchor_subject][anchor_view])

        positive_view = "palmar" if anchor_view == "dorsal" else "dorsal"
        positive_img = random.choice(
            self.subject_to_images[anchor_subject][positive_view]
        )

        negative_subject = random.choice(
            [s for s in self.valid_subjects if s != anchor_subject]
        )
        negative_view = random.choice(["dorsal", "palmar"])
        negative_img = random.choice(
            self.subject_to_images[negative_subject][negative_view]
        )

        anchor_path = self.data_root / "Hands" / anchor_img
        positive_path = self.data_root / "Hands" / positive_img
        negative_path = self.data_root / "Hands" / negative_img

        anchor = Image.open(anchor_path).convert("RGB")
        positive = Image.open(positive_path).convert("RGB")
        negative = Image.open(negative_path).convert("RGB")

        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return anchor, positive, negative, anchor_subject


class HandBatchDataset(Dataset):
    """Dataset for batch-based hard negative mining (more efficient)"""

    def __init__(
        self,
        data_root: Path,
        subject_ids: List[str],
        transform=None,
        samples_per_subject=4,
    ):
        self.data_root = Path(data_root)
        self.subject_ids = list(subject_ids)
        self.transform = transform
        self.samples_per_subject = samples_per_subject

        metadata_path = self.data_root / "HandInfo.csv"
        self.metadata = pd.read_csv(metadata_path)
        self.metadata = self.metadata[self.metadata["id"].astype(str).isin(subject_ids)]

        self.subject_to_images = {}

        for subject_id in self.subject_ids:
            subject_data = self.metadata[self.metadata["id"].astype(str) == subject_id]
            dorsal_images = subject_data[
                subject_data["aspectOfHand"].str.contains(
                    "dorsal", case=False, na=False
                )
            ]["imageName"].tolist()
            palmar_images = subject_data[
                subject_data["aspectOfHand"].str.contains(
                    "palmar", case=False, na=False
                )
            ]["imageName"].tolist()

            if len(dorsal_images) > 0 and len(palmar_images) > 0:
                self.subject_to_images[subject_id] = {
                    "dorsal": dorsal_images,
                    "palmar": palmar_images,
                    "all": dorsal_images + palmar_images,
                }

        self.valid_subjects = list(self.subject_to_images.keys())
        print(f"Loaded {len(self.valid_subjects)} valid subjects")

    def __len__(self):
        return len(self.valid_subjects) * 10

    def __getitem__(self, idx):
        subject_idx = idx % len(self.valid_subjects)
        subject_id = self.valid_subjects[subject_idx]

        all_images = self.subject_to_images[subject_id]["all"]

        dorsal_images = self.subject_to_images[subject_id]["dorsal"]
        palmar_images = self.subject_to_images[subject_id]["palmar"]

        samples = []
        n_per_view = self.samples_per_subject // 2

        selected_dorsal = random.sample(
            dorsal_images, min(n_per_view, len(dorsal_images))
        )
        selected_palmar = random.sample(
            palmar_images, min(n_per_view, len(palmar_images))
        )

        selected_images = selected_dorsal + selected_palmar

        while len(selected_images) < self.samples_per_subject and len(
            selected_images
        ) < len(all_images):
            remaining = [img for img in all_images if img not in selected_images]
            if remaining:
                selected_images.append(random.choice(remaining))

        images = []
        for img_name in selected_images:
            img_path = self.data_root / "Hands" / img_name
            img = Image.open(img_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            images.append(img)

        images = torch.stack(images)

        return images, subject_id


def collate_batch_fn(batch):
    """Custom collate function for batch-based training"""
    all_images = []
    all_labels = []

    for images, subject_id in batch:
        all_images.append(images)
        all_labels.extend([subject_id] * images.size(0))

    all_images = torch.cat(all_images, dim=0)

    return all_images, all_labels


class HandRetrievalDataset(Dataset):
    """Dataset for retrieval evaluation (query-gallery setup)"""

    def __init__(
        self,
        data_root: Path,
        subject_ids: List[str],
        query_view="dorsal",
        gallery_view="palmar",
        transform=None,
        max_images_per_subject=None,
    ):
        self.data_root = Path(data_root)
        self.subject_ids = set(subject_ids)
        self.query_view = query_view
        self.gallery_view = gallery_view
        self.transform = transform
        self.max_images_per_subject = max_images_per_subject

        metadata_path = self.data_root / "HandInfo.csv"
        self.metadata = pd.read_csv(metadata_path)
        self.metadata = self.metadata[
            self.metadata["id"].astype(str).isin(self.subject_ids)
        ]

        self.queries = []
        self.gallery = []

        for subject_id in self.subject_ids:
            subject_data = self.metadata[self.metadata["id"].astype(str) == subject_id]

            query_images = subject_data[
                subject_data["aspectOfHand"].str.contains(
                    query_view, case=False, na=False
                )
            ]["imageName"].tolist()
            gallery_images = subject_data[
                subject_data["aspectOfHand"].str.contains(
                    gallery_view, case=False, na=False
                )
            ]["imageName"].tolist()

            if self.max_images_per_subject:
                query_images = query_images[: self.max_images_per_subject]
                gallery_images = gallery_images[: self.max_images_per_subject]

            if len(query_images) > 0 and len(gallery_images) > 0:
                for query_img in query_images:
                    self.queries.append(
                        {
                            "path": self.data_root / "Hands" / query_img,
                            "subject": subject_id,
                        }
                    )

                for gallery_img in gallery_images:
                    self.gallery.append(
                        {
                            "path": self.data_root / "Hands" / gallery_img,
                            "subject": subject_id,
                        }
                    )

        print(
            f"Retrieval dataset: {len(self.queries)} queries, {len(self.gallery)} gallery items"
        )

    def get_query_loader(self, batch_size=32):
        dataset = SingleImageDataset(self.queries, self.transform)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)

    def get_gallery_loader(self, batch_size=32):
        dataset = SingleImageDataset(self.gallery, self.transform)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)


class SingleImageDataset(Dataset):
    """Helper dataset for single images"""

    def __init__(self, image_list, transform=None):
        self.image_list = image_list
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        item = self.image_list[idx]
        img = Image.open(item["path"]).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, item["subject"]


def train_epoch_batch(model, dataloader, criterion, optimizer, device, scaler=None):
    """Train for one epoch using batch-based hard negative mining"""
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc="Training")
    for images, labels in pbar:
        images = images.to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast(device_type="cuda"):
                embeddings = model(images)
                loss = criterion(embeddings, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            embeddings = model(images)
            loss = criterion(embeddings, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / num_batches


def extract_embeddings(model, dataloader, device):
    """Extract embeddings for all images"""
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


def compute_retrieval_metrics(
    query_embeddings, query_subjects, gallery_embeddings, gallery_subjects
):
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

    query_agg, query_subjects_unique = aggregate_by_subject(
        query_embeddings, query_subjects
    )
    gallery_agg, gallery_subjects_unique = aggregate_by_subject(
        gallery_embeddings, gallery_subjects
    )

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

    mean_rank = np.mean(ranks) if ranks else float("inf")
    num_queries = len(query_subjects_unique)

    metrics = {
        "mean_rank": mean_rank,
        "recall@1": recall_at_k[1] / num_queries,
        "recall@5": recall_at_k[5] / num_queries,
        "recall@10": recall_at_k[10] / num_queries,
    }

    return metrics


def evaluate_retrieval(model, retrieval_dataset, device):
    """Evaluate retrieval performance"""
    query_loader = retrieval_dataset.get_query_loader(batch_size=32)
    gallery_loader = retrieval_dataset.get_gallery_loader(batch_size=32)

    query_embeddings, query_subjects = extract_embeddings(model, query_loader, device)
    gallery_embeddings, gallery_subjects = extract_embeddings(
        model, gallery_loader, device
    )

    metrics = compute_retrieval_metrics(
        query_embeddings, query_subjects, gallery_embeddings, gallery_subjects
    )

    return metrics


def main():
    config = {
        "data_root": "./11k_hands_data",
        "embedding_dim": 256,
        "triplet_margin": 0.5,
        "mining_strategy": "hard",  # 'hard', 'semi-hard', or 'all'
        "batch_size": 8,
        "samples_per_subject": 2,
        "num_epochs": 30,
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "freeze_backbone": True,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "checkpoint_dir": "./checkpoints_dinov2",
        "seed": 42,
        "num_workers": 2,
        "prefetch_factor": 2,
    }

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )

    print(f"\nConfiguration:")
    print(f"  - Model: DINOv2 ViT-B/14")
    print(f"  - Loss: Triplet with {config['mining_strategy']} negative mining")
    print(f"  - Embedding dim: {config['embedding_dim']}")
    print(f"  - Margin: {config['triplet_margin']}")
    print(f"  - Backbone frozen: {config['freeze_backbone']}")

    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    Path(config["checkpoint_dir"]).mkdir(exist_ok=True, parents=True)

    train_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    metadata_path = Path(config["data_root"]) / "HandInfo.csv"
    metadata = pd.read_csv(metadata_path)

    all_subjects = metadata["id"].astype(str).unique().tolist()
    print(f"\nFound {len(all_subjects)} unique subjects in dataset")

    random.shuffle(all_subjects)

    n_subjects = len(all_subjects)
    n_train = int(0.68 * n_subjects)
    n_val = int(0.11 * n_subjects)

    train_subjects = all_subjects[:n_train]
    val_subjects = all_subjects[n_train : n_train + n_val]
    test_subjects = all_subjects[n_train + n_val :]

    print(f"Train: {len(train_subjects)} subjects")
    print(f"Val: {len(val_subjects)} subjects")
    print(f"Test: {len(test_subjects)} subjects")

    train_dataset = HandBatchDataset(
        config["data_root"],
        train_subjects,
        transform=train_transform,
        samples_per_subject=config["samples_per_subject"],
    )

    val_retrieval = HandRetrievalDataset(
        config["data_root"], val_subjects, transform=val_transform
    )
    test_retrieval = HandRetrievalDataset(
        config["data_root"], test_subjects, transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True,
        prefetch_factor=config["prefetch_factor"],
        persistent_workers=True if config["num_workers"] > 0 else False,
        collate_fn=collate_batch_fn,
    )

    device = torch.device(config["device"])
    print("\nLoading DINOv2 model...")
    model = DINOv2Encoder(
        embedding_dim=config["embedding_dim"], freeze_backbone=config["freeze_backbone"]
    ).to(device)
    print("Model loaded successfully!")

    criterion = TripletLoss(
        margin=config["triplet_margin"], mining_strategy=config["mining_strategy"]
    )

    if config["freeze_backbone"]:
        optimizer = optim.AdamW(
            model.projection.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"],
        )
        print("Optimizing projection head only (backbone frozen)")
    else:
        optimizer = optim.AdamW(
            model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
        )
        print("Optimizing full model")

    scaler = torch.amp.GradScaler("cuda") if config["device"] == "cuda" else None
    if scaler:
        print("Using mixed precision training (AMP)")

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["num_epochs"], eta_min=1e-6
    )

    best_mean_rank = float("inf")

    for epoch in range(config["num_epochs"]):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")

        train_loss = train_epoch_batch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        print(f"Train Loss: {train_loss:.4f}")

        if (epoch + 1) % 5 == 0 or epoch == 0:
            metrics = evaluate_retrieval(model, val_retrieval, device)
            print(
                f"Val Retrieval - Mean Rank: {metrics['mean_rank']:.2f}, "
                f"R@1: {metrics['recall@1']:.3f}, "
                f"R@5: {metrics['recall@5']:.3f}, "
                f"R@10: {metrics['recall@10']:.3f}"
            )

            if metrics["mean_rank"] < best_mean_rank:
                best_mean_rank = metrics["mean_rank"]
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "mean_rank": best_mean_rank,
                        "metrics": metrics,
                        "config": config,
                    },
                    f"{config['checkpoint_dir']}/best_model.pth",
                )
                print(f"✓ Saved new best model (mean rank: {best_mean_rank:.2f})")

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                },
                f"{config['checkpoint_dir']}/checkpoint_epoch_{epoch+1}.pth",
            )

    print("\n" + "=" * 70)
    print("Final Evaluation on Test Set")
    print("=" * 70)

    checkpoint = torch.load(
        f"{config['checkpoint_dir']}/best_model.pth", weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = evaluate_retrieval(model, test_retrieval, device)
    print(f"\nTest Results:")
    print(f"  Mean Rank: {test_metrics['mean_rank']:.2f}")
    print(f"  Recall@1:  {test_metrics['recall@1']:.3f}")
    print(f"  Recall@5:  {test_metrics['recall@5']:.3f}")
    print(f"  Recall@10: {test_metrics['recall@10']:.3f}")

    results = {
        "test_metrics": test_metrics,
        "val_metrics": checkpoint.get("metrics", {}),
        "config": config,
    }

    with open(f"{config['checkpoint_dir']}/test_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {config['checkpoint_dir']}/test_results.json")


if __name__ == "__main__":
    main()
