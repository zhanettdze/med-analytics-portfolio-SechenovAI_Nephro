import nibabel as nib
import numpy as np
from pathlib import Path

def save_segmentation_as_nifti(mask_3d, output_path, affine):
    img = nib.Nifti1Image(mask_3d.astype(np.uint8), affine)
    nib.save(img, output_path)
    print(f"Saved to {output_path}")

def extract_kidney_only(seg_path, output_path):
    seg_img = nib.load(str(seg_path))
    seg_data = seg_img.get_fdata()
    
    kidney_mask = (seg_data == 1).astype(np.uint8) * 255
    
    save_segmentation_as_nifti(kidney_mask, output_path, seg_img.affine)
    
    print(f"\nKidney mask extracted: {output_path}")
    print("To open in ITK-SNAP:")
    print(f"  File -> Open Main Image -> {output_path}")

if __name__ == "__main__":
    kits19_path = Path("data/kits19/data/case_00000/")
    seg_path = kits19_path / "segmentation.nii.gz"
    
    if seg_path.exists():
        extract_kidney_only(seg_path, "kidney_only.nii.gz")
    else:
        print("KiTS19 segmentation not found at data/kits19/")
        print("Run registration scripts first to generate data.")
