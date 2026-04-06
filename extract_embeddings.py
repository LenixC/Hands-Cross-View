import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
import numpy as np
from PIL import Image
import json
from tqdm import tqdm


class DINOv2Encoder(nn.Module):
    def __init__(self, embedding_dim=256):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
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


class AllImagesDataset(Dataset):
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
        return img, str(item["subject"]), str(item["path"]), item["view"]


def extract_all_embeddings(model, dataloader, device):
    model.eval()
    results = []

    with torch.no_grad():
        for imgs, subjects, paths, views in tqdm(dataloader, desc="Extracting"):
            imgs = imgs.to(device)
            emb = model(imgs).cpu().numpy()
            for i in range(len(subjects)):
                results.append(
                    {
                        "embedding": emb[i].tolist(),
                        "subject": subjects[i],
                        "path": str(paths[i]),
                        "view": views[i],
                    }
                )

    return results


def main():
    config = {
        "data_root": ".",
        "checkpoint": "./checkpoints_dinov2/best_model.pth",
        "embedding_dim": 256,
        "batch_size": 16,
        "output": "embeddings_demo.json",
    }

    print("Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DINOv2Encoder(embedding_dim=config["embedding_dim"]).to(device)

    checkpoint = torch.load(config["checkpoint"], weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print("Model loaded!")

    import pandas as pd

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    metadata = pd.read_csv(Path(config["data_root"]) / "HandInfo.csv")
    all_subjects = metadata["id"].astype(str).unique().tolist()

    import random

    random.seed(42)
    random.shuffle(all_subjects)

    n_subjects = len(all_subjects)
    n_train = int(0.68 * n_subjects)
    n_val = int(0.11 * n_subjects)
    test_subjects = all_subjects[n_train + n_val :]

    print(f"Using {len(test_subjects)} test subjects")

    test_metadata = metadata[metadata["id"].astype(str).isin(test_subjects)]

    dorsal_images = []
    palmar_images = []

    for _, row in test_metadata.iterrows():
        view = "dorsal" if "dorsal" in str(row["aspectOfHand"]).lower() else "palmar"
        item = {
            "path": Path(config["data_root"]) / "Hands" / row["imageName"],
            "subject": str(row["id"]),
            "view": view,
        }
        if view == "dorsal":
            dorsal_images.append(item)
        else:
            palmar_images.append(item)

    print(f"Found {len(dorsal_images)} dorsal, {len(palmar_images)} palmar test images")

    dorsal_dataset = AllImagesDataset(dorsal_images, transform)
    palmar_dataset = AllImagesDataset(palmar_images, transform)

    dorsal_loader = DataLoader(
        dorsal_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=2
    )
    palmar_loader = DataLoader(
        palmar_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=2
    )

    print("Extracting dorsal embeddings...")
    dorsal_results = extract_all_embeddings(model, dorsal_loader, device)

    print("Extracting palmar embeddings...")
    palmar_results = extract_all_embeddings(model, palmar_loader, device)

    demo_data = {
        "dorsal": dorsal_results,
        "palmar": palmar_results,
    }

    with open(config["output"], "w") as f:
        json.dump(demo_data, f)

    print(f"Saved embeddings to {config['output']}")
    print(f"Dorsal: {len(dorsal_results)}, Palmar: {len(palmar_results)}")


if __name__ == "__main__":
    main()
