# Dorsal-to-Palmar Hand Matching

Consider reading the [blog post](https://lenixc.github.io/2026/02/02/cross-view-hands.html). It gives a more narrative look into the problem
and why and how I made certain decisions. 

## Problem Statement

Given an image of the back (dorsal) side of a hand, can we identify which image of a palm (palmar) side belongs to the same person? This is a challenging cross-view biometric matching problem where we need to learn identity-preserving features that work across different hand orientations.

## Approach

We frame this as a metric learning problem: learn an embedding space where images of the same person's hands are close together, regardless of whether they show the dorsal or palmar view.

### Dataset
- **11k Hands** [1]: 11,076 hand images (1600 × 1200 pixels) from 190 subjects
- Ages 18-75 years old
- Each subject photographed: left/right hands, dorsal/palmar sides, fingers open/closed
- Metadata includes: subject ID, gender, age, skin color, accessories, nail polish
- Split: 68% train, 11% validation, 21% test (by subject)
- Evaluation: Query with dorsal images, retrieve matching palmar images from gallery

## Architectures

### 1. Baseline: ResNet50 + Contrastive Loss

**Architecture:**
- Encoder: ResNet50 [4] (ImageNet pretrained) → 2048-dim features
- Projection head: 2048 → 512 → 128-dim embeddings
- Twin network: same encoder processes both images

**Loss Function:**
```
L = y * d² + (1-y) * max(margin - d, 0)²
```
Where `d` is the distance between embeddings, `y=1` for same person, `y=0` for different people [5].

**Training Strategy:**
- Randomly sample positive pairs (dorsal-palmar from same person)
- Randomly sample negative pairs (any two images from different people)
- Margin = 2.0

**Why this approach:**
- Simple, proven architecture
- Works well as a baseline
- Fast to train (~1 hour on consumer GPU)

### 2. Advanced: DINOv2 + Triplet Loss + Hard Negative Mining

**Architecture:**
- Encoder: DINOv2 ViT-B/14 [2] (self-supervised pretrained) → 768-dim features
- Projection head: 768 → 512 → 256-dim embeddings
- L2-normalized embeddings

**Loss Function:**
```
L = max(d(a,p) - d(a,n) + margin, 0)
```
Where `a` is anchor, `p` is positive (same person), `n` is negative (different person) [3].

**Hard Negative Mining:**
For each anchor, we select:
- Hardest positive: most distant same-person sample
- Hardest negative: closest different-person sample

This forces the model to learn from the most difficult examples rather than easy ones [3].

**Why this approach:**
- DINOv2 learns superior self-supervised features (trained on 142M images) [2]
- Triplet loss directly optimizes ranking metric [3]
- Hard negative mining focuses learning on challenging distinctions
- Better sample efficiency than random sampling

## Results

| Metric | ResNet50 + Contrastive | DINOv2 + Triplet | Improvement |
|--------|------------------------|------------------|-------------|
| **Mean Rank** | 12.20 | 2.51 | **+79%** |
| **Recall@1** | 9.8% | 61.0% | **+525%** |
| **Recall@5** | 41.5% | 90.2% | **+118%** |
| **Recall@10** | 63.4% | 95.1% | **+50%** |

### Random Baseline
A random baseline (guessing) would achieve:
- Recall@1: ~0.5% (1 in ~200 subjects)
- Mean Rank: ~100

Both learned models significantly outperform random guessing, with DINOv2 achieving near-perfect retrieval in the top 10 results.

## Key Findings

1. **Hard negative mining matters**: By selecting challenging negatives rather than random ones, the model learns more discriminative features. This resulted in 6x improvement in top-1 accuracy.

2. **Self-supervised features transfer well**: DINOv2's pretraining on 142M images provides better feature representations than ImageNet-supervised ResNet50 for this cross-view matching task.

3. **Triplet loss for retrieval**: Directly optimizing for ranking (triplet loss) outperforms pair-based contrastive loss, achieving 95% top-10 accuracy.

## Limitations

### Computational Constraints

Due to mid-range consumer GPU limitations (~8GB VRAM), we had to make several compromises:

1. **Frozen backbone**: We froze the DINOv2 encoder and only trained the projection head. Full fine-tuning would likely improve results further but requires ~12-16GB VRAM.

2. **Small batch sizes**: Limited to batch size of 8 (with 2 images per subject). Larger batches (32-64) would provide more hard negatives per iteration and faster training.

3. **Limited mining scope**: Hard negative mining only considers negatives within each batch. A global mining strategy across the full dataset would be more effective but computationally expensive.

4. **Model size**: Used DINOv2 ViT-B/14 instead of the larger ViT-L/14 or ViT-g/14 variants which would provide better features but don't fit in memory.

### Potential Improvements

With more computational resources:
- Full fine-tuning of DINOv2 backbone
- Larger batch sizes for better hard negative mining
- Multi-crop training for more robust features
- Ensemble of multiple models
- Cross-batch hard negative mining
- Larger DINOv2 variants (ViT-L or ViT-g)

## Usage

**Train baseline:**
```bash
python resnet_contrastive.py
```

**Train DINOv2 model:**
```bash
python dino_triplet_hnmining.py
```

**Memory-saving config** (edit in dinov2_triplet_model.py):
```python
config = {
    'freeze_backbone': True,  # Only train projection head
    'batch_size': 8,          # Reduce if OOM
    'samples_per_subject': 2,
}
```

## Requirements

```bash
pip install torch torchvision tqdm pandas pillow numpy
```

Minimum: 8GB GPU VRAM (with frozen backbone)  
Recommended: 12-16GB GPU VRAM (for full fine-tuning)

## References

[1] Afifi, M. (2019). "11K Hands: Gender recognition and biometric identification using a large dataset of hand images." *Multimedia Tools and Applications*. https://doi.org/10.1007/s11042-019-7424-8

[2] Oquab, M., Darcet, T., Moutakanni, T., et al. (2023). "DINOv2: Learning Robust Visual Features without Supervision." *arXiv preprint arXiv:2304.07193*.

[3] Schroff, F., Kalenichenko, D., & Philbin, J. (2015). "FaceNet: A unified embedding for face recognition and clustering." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 815-823.

[4] He, K., Zhang, X., Ren, S., & Sun, J. (2016). "Deep residual learning for image recognition." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 770-778.

[5] Hadsell, R., Chopra, S., & LeCun, Y. (2006). "Dimensionality reduction by learning an invariant mapping." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, Vol. 2, 1735-1742.
