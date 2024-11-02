import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tfdiff.conditioning import ConditioningManager
from typing import Dict, List, Optional
import pandas as pd

class ConditioningVisualizer:
    def __init__(self, conditioning_manager: ConditioningManager):
        """Initialize visualizer with a conditioning manager"""
        self.cm = conditioning_manager
        
    def plot_conditioning_vector(self, cond_vector: torch.Tensor, 
                               title: Optional[str] = None,
                               figsize: tuple = (10, 6)):
        """Visualize a single conditioning vector"""
        # Convert to numpy if needed
        if isinstance(cond_vector, torch.Tensor):
            cond_vector = cond_vector.numpy()
            
        # Get human readable description
        description = self.cm.describe_vector(torch.tensor(cond_vector))
        
        plt.figure(figsize=figsize)
        
        # Plot vector values
        plt.subplot(2, 1, 1)
        plt.bar(range(len(cond_vector)), cond_vector)
        plt.title("Conditioning Vector Values")
        plt.xlabel("Dimension")
        plt.ylabel("Value")
        
        # Add text description
        plt.subplot(2, 1, 2)
        plt.axis('off')
        desc_text = "Conditioning Description:\n\n"
        for key, value in description.items():
            desc_text += f"{key}: {value}\n"
        plt.text(0.1, 0.5, desc_text, fontsize=12, family='monospace')
        
        if title:
            plt.suptitle(title)
        plt.tight_layout()
        return plt.gcf()
    
    def plot_conditioning_distribution(self, cond_vectors: List[torch.Tensor],
                                    figsize: tuple = (15, 10)):
        """Visualize distribution of conditioning values across multiple samples"""
        # Convert to numpy array
        vectors = np.stack([v.numpy() if isinstance(v, torch.Tensor) else v 
                          for v in cond_vectors])
        
        plt.figure(figsize=figsize)
        
        # Plot distribution for each dimension
        num_dims = vectors.shape[1]
        num_cols = min(3, num_dims)
        num_rows = (num_dims + num_cols - 1) // num_cols
        
        for i in range(num_dims):
            plt.subplot(num_rows, num_cols, i+1)
            sns.histplot(vectors[:, i], bins=30)
            plt.title(f"Dimension {i}")
            
        plt.tight_layout()
        return plt.gcf()
    
    def plot_correlation_matrix(self, cond_vectors: List[torch.Tensor],
                              figsize: tuple = (8, 8)):
        """Visualize correlation between different conditioning dimensions"""
        # Convert to numpy array
        vectors = np.stack([v.numpy() if isinstance(v, torch.Tensor) else v 
                          for v in cond_vectors])
        
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(vectors.T)
        
        plt.figure(figsize=figsize)
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   vmin=-1, vmax=1)
        plt.title("Conditioning Dimension Correlations")
        return plt.gcf()
    
    def analyze_dataset_conditioning(self, dataset, num_samples: int = 1000,
                                  save_path: Optional[str] = None):
        """Comprehensive analysis of conditioning in a dataset"""
        # Sample dataset
        samples = []
        for i in range(min(num_samples, len(dataset))):
            batch = dataset[i]
            samples.append(batch['cond'])
            
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Distribution plot
        plt.subplot(2, 2, 1)
        self.plot_conditioning_distribution(samples)
        plt.title("Conditioning Distributions")
        
        # 2. Correlation matrix
        plt.subplot(2, 2, 2)
        self.plot_correlation_matrix(samples)
        plt.title("Conditioning Correlations")
        
        # 3. Example vectors
        plt.subplot(2, 2, 3)
        self.plot_conditioning_vector(samples[0])
        plt.title("Example Vector 1")
        
        plt.subplot(2, 2, 4)
        self.plot_conditioning_vector(samples[-1])
        plt.title("Example Vector 2")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            
        return plt.gcf()
    
    def get_conditioning_stats(self, dataset) -> pd.DataFrame:
        """Get statistical summary of conditioning in dataset"""
        # Collect all conditioning vectors
        samples = []
        for i in range(len(dataset)):
            batch = dataset[i]
            samples.append(batch['cond'].numpy())
        samples = np.stack(samples)
        
        # Calculate statistics
        stats = {
            'mean': np.mean(samples, axis=0),
            'std': np.std(samples, axis=0),
            'min': np.min(samples, axis=0),
            'max': np.max(samples, axis=0),
            'median': np.median(samples, axis=0)
        }
        
        # Create DataFrame
        df = pd.DataFrame(stats).T
        df.columns = [f'dim_{i}' for i in range(samples.shape[1])]
        return df

# Example usage:
if __name__ == "__main__":
    from argparse import ArgumentParser
    
    parser = ArgumentParser(description='Analyze conditioning in dataset')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to model config')
    parser.add_argument('--output', type=str, default='conditioning_analysis.png',
                       help='Output path for analysis plot')
    args = parser.parse_args()
    
    # Load config and create dataset
    from tfdiff.params import load_config
    from tfdiff.dataset import from_path
    
    config = load_config(args.config)
    dataset = from_path(config)
    
    # Create visualizer and analyze
    vis = ConditioningVisualizer(dataset.conditioning_manager)
    vis.analyze_dataset_conditioning(dataset, save_path=args.output)
    
    # Print statistics
    print("\nConditioning Statistics:")
    print(vis.get_conditioning_stats(dataset))