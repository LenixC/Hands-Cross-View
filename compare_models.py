import subprocess
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def run_experiment(script_name, experiment_name):
    """Run an experiment and return the results"""
    print(f"\n{'='*70}")
    print(f"Running: {experiment_name}")
    print(f"{'='*70}\n")
    
    try:
        result = subprocess.run(
            ['python', script_name],
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running {experiment_name}:")
        print(e.stderr)
        return False


def load_results(checkpoint_dir):
    """Load test results from a checkpoint directory"""
    results_path = Path(checkpoint_dir) / "test_results.json"
    if results_path.exists():
        with open(results_path, 'r') as f:
            return json.load(f)
    return None


def compare_results(baseline_results, dinov2_results):
    """Compare and visualize results from both models"""
    
    baseline_metrics = baseline_results if isinstance(baseline_results, dict) else baseline_results.get('test_metrics', {})
    dinov2_metrics = dinov2_results.get('test_metrics', {}) if isinstance(dinov2_results, dict) else dinov2_results
    
    metrics_names = ['mean_rank', 'recall@1', 'recall@5', 'recall@10']
    
    comparison_data = {
        'Metric': [],
        'ResNet50 + Contrastive': [],
        'DINOv2 + Triplet': [],
        'Improvement': []
    }
    
    for metric in metrics_names:
        baseline_val = baseline_metrics.get(metric, 0)
        dinov2_val = dinov2_metrics.get(metric, 0)
        
        if metric == 'mean_rank':
            improvement = ((baseline_val - dinov2_val) / baseline_val * 100) if baseline_val > 0 else 0
        else:
            improvement = ((dinov2_val - baseline_val) / baseline_val * 100) if baseline_val > 0 else 0
        
        comparison_data['Metric'].append(metric)
        comparison_data['ResNet50 + Contrastive'].append(baseline_val)
        comparison_data['DINOv2 + Triplet'].append(dinov2_val)
        comparison_data['Improvement'].append(f"{improvement:+.2f}%")
    
    df = pd.DataFrame(comparison_data)
    
    print("\n" + "="*70)
    print("RESULTS COMPARISON")
    print("="*70)
    print(df.to_string(index=False))
    print("="*70)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    recall_metrics = ['recall@1', 'recall@5', 'recall@10']
    x = range(len(recall_metrics))
    width = 0.35
    
    baseline_recalls = [baseline_metrics.get(m, 0) for m in recall_metrics]
    dinov2_recalls = [dinov2_metrics.get(m, 0) for m in recall_metrics]
    
    axes[0].bar([i - width/2 for i in x], baseline_recalls, width, 
                label='ResNet50 + Contrastive', alpha=0.8)
    axes[0].bar([i + width/2 for i in x], dinov2_recalls, width, 
                label='DINOv2 + Triplet', alpha=0.8)
    
    axes[0].set_xlabel('Metric')
    axes[0].set_ylabel('Recall Score')
    axes[0].set_title('Recall Comparison')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(['R@1', 'R@5', 'R@10'])
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    mean_ranks = [baseline_metrics.get('mean_rank', 0), 
                  dinov2_metrics.get('mean_rank', 0)]
    models = ['ResNet50\n+ Contrastive', 'DINOv2\n+ Triplet']
    
    bars = axes[1].bar(models, mean_ranks, alpha=0.8, 
                       color=['#1f77b4', '#ff7f0e'])
    axes[1].set_ylabel('Mean Rank (lower is better)')
    axes[1].set_title('Mean Rank Comparison')
    axes[1].grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: model_comparison.png")
    
    return df


def main():
    print("Hand Matching Model Comparison")
    print("="*70)
    print("\nThis script will run both models and compare their performance:")
    print("1. Baseline: ResNet50 + Contrastive Loss")
    print("2. Advanced: DINOv2 + Triplet Loss with Hard Negative Mining")
    print("\nNote: This will take a while to complete!")
    
    choice = input("\nOptions:\n1. Run both experiments\n2. Compare existing results only\nChoose (1/2): ").strip()
    
    if choice == '1':
        baseline_success = run_experiment('baseline_model.py', 'Baseline (ResNet50 + Contrastive)')
        
        dinov2_success = run_experiment('dinov2_triplet_model.py', 'DINOv2 + Triplet')
        
        if not (baseline_success and dinov2_success):
            print("\nError: One or both experiments failed!")
            return
    
    print("\n" + "="*70)
    print("Loading results...")
    print("="*70)
    
    baseline_results = load_results('./checkpoints')
    dinov2_results = load_results('./checkpoints_dinov2')
    
    if baseline_results is None:
        print("Error: Could not find baseline results in ./checkpoints/test_results.json")
        return
    
    if dinov2_results is None:
        print("Error: Could not find DINOv2 results in ./checkpoints_dinov2/test_results.json")
        return
    
    comparison_df = compare_results(baseline_results, dinov2_results)
    
    comparison_df.to_csv('model_comparison.csv', index=False)
    print(f"\nComparison table saved to: model_comparison.csv")
    
    print("\n✓ Comparison complete!")


if __name__ == "__main__":
    main()
