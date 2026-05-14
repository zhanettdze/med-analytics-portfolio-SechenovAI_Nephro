import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Optional

def load_nifti(file_path: str) -> np.ndarray:
    img = nib.load(file_path)
    return img.get_fdata().astype(np.float32)

def save_nifti(data: np.ndarray, file_path: str, affine: Optional[np.ndarray] = None) -> None:
    if affine is None:
        affine = np.eye(4)
    img = nib.Nifti1Image(data, affine)
    nib.save(img, file_path)

def get_nifti_affine(file_path: str) -> np.ndarray:
    img = nib.load(file_path)
    return img.affine

if __name__ == "__main__":
    import sys
    
    # Try to load KiTS19 sample if available
    kits19_path = Path("data/kits19/data/case_00000/imaging.nii.gz")
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        data = load_nifti(file_path)
        print(f"Loaded: {file_path}")
        print(f"Shape: {data.shape}")
        print(f"Range: [{data.min():.1f}, {data.max():.1f}]")
    elif kits19_path.exists():
        print(f"Demo mode: loading KiTS19 sample from {kits19_path}")
        data = load_nifti(str(kits19_path))
        print(f"Shape: {data.shape}")
        print(f"Range: [{data.min():.1f}, {data.max():.1f}]")
        print(f"Voxel spacing: {get_nifti_affine(str(kits19_path))}")
    else:
        print("NIfTI I/O for CT kidney")
        print("Usage: python 02_nifti_io.py /path/to/file.nii.gz")
        print("\nTip: Run '../data/kits19/get_imaging.py' first to download KiTS19 data.")