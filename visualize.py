import matplotlib.pyplot as plt
import numpy as np

def visualize_comparison(original, refined, edges, guidance, save_path=None):
    
    '''Visualize the before/after results'''
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(original, cmap='viridis')
    axes[0, 0].set_title('Original Depth')
    
    axes[0, 1].imshow(refined, cmap='viridis')
    axes[0, 1].set_title('Refined Depth')
    
    axes[0, 2].imshow(edges, cmap='gray')
    axes[0, 2].set_title('SAM Edges')
    
    axes[1, 0].imshow(guidance, cmap='gray')
    axes[1, 0].set_title('Guiding Image')
    
    diff = np.abs(original.astype(float) - refined.astype(float))
    axes[1, 1].imshow(diff, cmap='hot')
    axes[1, 1].set_title('Difference Map')
    
    for ax in axes.flat:
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()