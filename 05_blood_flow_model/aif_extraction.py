import numpy as np
import nibabel as nib
from pathlib import Path

def extract_aif_simple(ct_volume, aorta_hu_range=(150, 400)):
    """
    Simple AIF extraction based on HU threshold.
    
    Parameters:
    - ct_volume: 4D CT volume (time, z, y, x)
    - aorta_hu_range: expected HU range for aorta
    
    Returns:
    - aif_curve: mean concentration in aorta over time
    - aorta_mask: binary mask of aorta
    """
    if len(ct_volume.shape) == 4:
        first_time_point = ct_volume[0]
    else:
        first_time_point = ct_volume
    
    aorta_mask = (first_time_point >= aorta_hu_range[0]) & (first_time_point <= aorta_hu_range[1])
    
    from scipy import ndimage
    aorta_mask = ndimage.binary_erosion(aorta_mask, iterations=2)
    aorta_mask = ndimage.binary_dilation(aorta_mask, iterations=3)
    
    labeled, num_features = ndimage.label(aorta_mask)
    if num_features > 0:
        sizes = ndimage.sum(aorta_mask, labeled, range(1, num_features + 1))
        largest = np.argmax(sizes) + 1
        aorta_mask = (labeled == largest)
    
    if len(ct_volume.shape) == 4:
        aif_curve = []
        for t in range(ct_volume.shape[0]):
            mean_hu = ct_volume[t][aorta_mask].mean()
            aif_curve.append(mean_hu)
        return np.array(aif_curve), aorta_mask
    else:
        return ct_volume[aorta_mask].mean(), aorta_mask

def extract_aif_peak_heuristic(ct_4d):
    """
    Extract AIF by finding the voxel with maximum peak enhancement.
    """
    time_max = ct_4d.max(axis=0)
    peak_voxel = np.unravel_index(np.argmax(time_max), time_max.shape)
    
    aif_curve = ct_4d[:, peak_voxel[0], peak_voxel[1], peak_voxel[2]]
    return aif_curve, peak_voxel

def normalize_aif(aif_curve, baseline_start=0, baseline_end=10):
    """
    Subtract baseline from AIF.
    """
    baseline = np.mean(aif_curve[baseline_start:baseline_end])
    return aif_curve - baseline

if __name__ == "__main__":
    print("AIF EXTRACTION MODULE")
    print("=" * 40)
    print("Functions available:")
    print("  - extract_aif_simple(ct_volume, aorta_hu_range)")
    print("  - extract_aif_peak_heuristic(ct_4d)")
    print("  - normalize_aif(aif_curve, baseline_start, baseline_end)")
    print("\nNote: Requires 4D CT data (time series).")