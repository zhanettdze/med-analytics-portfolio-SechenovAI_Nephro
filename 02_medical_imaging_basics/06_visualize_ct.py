import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Optional, List

def plot_ct_slices(
    ct_volume: np.ndarray, 
    mask_volume: Optional[np.ndarray] = None,
    slice_indices: Optional[List[int]] = None,
    cols: int = 4,
    figsize: tuple = (16, 4),
    cmap: str = 'gray',
    window_center: int = 40,
    window_width: int = 400
) -> None:
    total_slices = ct_volume.shape[0]
    
    if slice_indices is None:
        step = max(1, total_slices // (cols * 2))
        slice_indices = list(range(0, total_slices, step))[:cols]
    
    rows = (len(slice_indices) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if rows * cols > 1 else [axes]
    
    min_hu = window_center - window_width // 2
    max_hu = window_center + window_width // 2
    
    for idx, ax in enumerate(axes):
        if idx < len(slice_indices):
            slice_data = ct_volume[slice_indices[idx]]
            slice_data = np.clip(slice_data, min_hu, max_hu)
            ax.imshow(slice_data, cmap=cmap)
            
            if mask_volume is not None:
                mask_slice = mask_volume[slice_indices[idx]]
                ax.imshow(mask_slice, cmap='jet', alpha=0.4, vmin=0, vmax=2)
            
            ax.set_title(f"Slice {slice_indices[idx]}")
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    kits19_img = Path("data/kits19/data/case_00000/imaging.nii.gz")
    kits19_mask = Path("data/kits19/data/case_00000/segmentation.nii.gz")
    
    if kits19_img.exists():
        import nibabel as nib
        print(f"Loading KiTS19 data...")
        ct_volume = nib.load(str(kits19_img)).get_fdata()
        mask_volume = nib.load(str(kits19_mask)).get_fdata() if kits19_mask.exists() else None
        
        print(f"CT shape: {ct_volume.shape}")
        if mask_volume is not None:
            print(f"Mask shape: {mask_volume.shape}")
            print(f"Mask values: {np.unique(mask_volume)} (0=background, 1=kidney, 2=tumor)")
        
        plot_ct_slices(ct_volume, mask_volume, cols=5)
    else:
        print("Demo mode: synthetic data")
        fake_ct = np.random.rand(100, 256, 256)
        plot_ct_slices(fake_ct, cols=4)
        print("Tip: Run '../data/kits19/get_imaging.py' first to download KiTS19 data.")