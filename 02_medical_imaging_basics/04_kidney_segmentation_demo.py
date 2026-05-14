import numpy as np
from scipy import ndimage
from pathlib import Path
from typing import Tuple

def segment_kidney_by_hu(
    ct_hu: np.ndarray, 
    min_hu: int = 30, 
    max_hu: int = 200,
    min_volume_cc: float = 50.0,
    voxel_volume_ml: float = 0.001
) -> np.ndarray:
    mask = (ct_hu >= min_hu) & (ct_hu <= max_hu)
    mask = ndimage.binary_erosion(mask, iterations=1)
    mask = ndimage.binary_dilation(mask, iterations=2)
    mask = ndimage.binary_fill_holes(mask)
    
    labeled, num_features = ndimage.label(mask)
    if num_features > 1:
        sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
        largest = np.argmax(sizes) + 1
        mask = (labeled == largest)
    
    volume_ml = mask.sum() * voxel_volume_ml
    if volume_ml < min_volume_cc:
        print(f"Warning: Kidney volume {volume_ml:.1f}ml < {min_volume_cc}ml threshold")
    
    return mask.astype(np.uint8)

def get_kidney_bbox(mask: np.ndarray) -> Tuple[int, int, int, int, int, int]:
    slices = np.any(mask, axis=(1, 2))
    rows = np.any(mask, axis=(0, 2))
    cols = np.any(mask, axis=(0, 1))
    
    z_min, z_max = np.where(slices)[0][[0, -1]]
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    
    return z_min, z_max, y_min, y_max, x_min, x_max

if __name__ == "__main__":
    kits19_path = Path("data/kits19/data/case_00000/imaging.nii.gz")
    
    if kits19_path.exists():
        import nibabel as nib
        img = nib.load(str(kits19_path))
        ct_data = img.get_fdata()
        print(f"Loaded KiTS19 CT: {ct_data.shape}")
        mask = segment_kidney_by_hu(ct_data)
        bbox = get_kidney_bbox(mask)
        print(f"Kidney mask: {mask.sum()} voxels")
        print(f"BBox: z={bbox[0]}-{bbox[1]}, y={bbox[2]}-{bbox[3]}, x={bbox[4]}-{bbox[5]}")
    else:
        print("Demo mode: synthetic data")
        fake_ct = np.random.uniform(-100, 300, (100, 512, 512))
        mask = segment_kidney_by_hu(fake_ct)
        print(f"CT shape: {fake_ct.shape}, Mask sum: {mask.sum()}")