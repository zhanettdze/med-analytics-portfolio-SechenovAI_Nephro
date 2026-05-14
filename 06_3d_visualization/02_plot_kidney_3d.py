import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import nibabel as nib
from pathlib import Path

# Правильный путь к маске
mask_path = Path("/Users/zhanett/01_PROJECTS/Programming/med-analytics-portfolio/data/kits19/data/case_00000/segmentation.nii.gz")

print(f"Looking for mask at: {mask_path}")

if not mask_path.exists():
    print(f"Mask not found!")
    print("Checking alternative location...")
    alt_path = Path("/Users/zhanett/01_PROJECTS/Programming/med-analytics-portfolio/kits19/data/case_00000/segmentation.nii.gz")
    if alt_path.exists():
        mask_path = alt_path
        print(f"Found at: {mask_path}")
    else:
        print("Mask not found in any location")
        exit(1)

img = nib.load(str(mask_path))
mask = img.get_fdata()
print(f"Mask shape: {mask.shape}")

print("Extracting kidney voxels...")
kidney_mask = (mask == 1)
tumor_mask = (mask == 2)

kidney_coords = np.argwhere(kidney_mask)[::5]
tumor_coords = np.argwhere(tumor_mask)[::2]

print(f"Kidney voxels (sampled): {len(kidney_coords)}")
print(f"Tumor voxels (sampled): {len(tumor_coords)}")

# Plot 1: Kidney only
print("\nCreating kidney_3d.png...")
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

if len(kidney_coords) > 0:
    ax.scatter(kidney_coords[:, 2], kidney_coords[:, 1], kidney_coords[:, 0],
               c='red', s=1, alpha=0.3, label='Kidney')

ax.set_xlabel('X (voxels)')
ax.set_ylabel('Y (voxels)')
ax.set_zlabel('Z (slices)')
ax.set_title('Kidney 3D Model (KiTS19 case_00000)')
ax.legend()

plt.savefig('kidney_3d.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: kidney_3d.png")

# Plot 2: Kidney + Tumor
print("Creating kidney_tumor_3d.png...")
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

if len(kidney_coords) > 0:
    ax.scatter(kidney_coords[:, 2], kidney_coords[:, 1], kidney_coords[:, 0],
               c='red', s=1, alpha=0.3, label='Kidney')

if len(tumor_coords) > 0:
    ax.scatter(tumor_coords[:, 2], tumor_coords[:, 1], tumor_coords[:, 0],
               c='blue', s=3, alpha=0.7, label='Tumor')

ax.set_xlabel('X (voxels)')
ax.set_ylabel('Y (voxels)')
ax.set_zlabel('Z (slices)')
ax.set_title('Kidney and Tumor 3D Model (KiTS19 case_00000)')
ax.legend()

plt.savefig('kidney_tumor_3d.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: kidney_tumor_3d.png")

print("\nDone! Images saved in 06_3d_visualization/")
