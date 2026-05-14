"""
Overlay blood flow map on 3D kidney surface.
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import nibabel as nib
from pathlib import Path

def plot_flow_on_surface(mask_path, flow_path, title="Blood Flow Distribution"):
    """
    Plot kidney surface colored by blood flow values.
    """
    from skimage import measure
    
    print(f"Loading mask: {mask_path}")
    mask_img = nib.load(mask_path)
    mask = mask_img.get_fdata()
    kidney_mask = (mask == 1)
    
    print(f"Loading flow map: {flow_path}")
    flow_img = nib.load(flow_path)
    flow = flow_img.get_fdata()
    
    print("Extracting surface (this may take a moment)...")
    # Убираем step=2, просто используем marching_cubes
    verts, faces, _, _ = measure.marching_cubes(kidney_mask, level=0.5)
    
    if len(verts) == 0:
        print("No surface extracted")
        return
    
    print(f"Surface: {len(verts)} vertices, {len(faces)} faces")
    
    print("Sampling flow values on surface...")
    flow_values = []
    for v in verts:
        i, j, k = int(v[0]), int(v[1]), int(v[2])
        if 0 <= i < flow.shape[0] and 0 <= j < flow.shape[1] and 0 <= k < flow.shape[2]:
            flow_values.append(flow[i, j, k])
        else:
            flow_values.append(0)
    
    flow_values = np.array(flow_values)
    
    if len(flow_values) > 0:
        print(f"Flow range on surface: [{flow_values.min():.1f}, {flow_values.max():.1f}]")
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    
    if len(flow_values) > 0 and flow_values.max() > flow_values.min():
        norm = Normalize(vmin=flow_values.min(), vmax=flow_values.max())
        cmap = plt.cm.jet
    else:
        norm = Normalize(vmin=0, vmax=1)
        cmap = plt.cm.Reds
    
    mesh = Poly3DCollection(verts[faces], alpha=0.8, linewidths=0.1, edgecolors='gray')
    
    face_colors = []
    for face in faces:
        if len(flow_values) > 0:
            center_flow = np.mean([flow_values[idx] for idx in face])
            face_colors.append(cmap(norm(center_flow)))
        else:
            face_colors.append(cmap(0.5))
    
    mesh.set_facecolors(face_colors)
    ax.add_collection3d(mesh)
    
    ax.set_xlim(verts[:, 0].min(), verts[:, 0].max())
    ax.set_ylim(verts[:, 1].min(), verts[:, 1].max())
    ax.set_zlim(verts[:, 2].min(), verts[:, 2].max())
    
    if len(flow_values) > 0 and flow_values.max() > flow_values.min():
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array(flow_values)
        cbar = plt.colorbar(sm, ax=ax, shrink=0.6)
        cbar.set_label('Blood Flow (ml/min/100g)')
    
    ax.set_xlabel('X (voxels)')
    ax.set_ylabel('Y (voxels)')
    ax.set_zlabel('Z (slices)')
    ax.set_title(title)
    
    output_path = 'flow_on_kidney.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

if __name__ == "__main__":
    # Пути к файлам
    base_path = Path("/Users/zhanett/01_PROJECTS/Programming/med-analytics-portfolio")
    mask_path = base_path / "data/kits19/data/case_00000/segmentation.nii.gz"
    flow_path = base_path / "05_blood_flow_model/blood_flow_results/blood_flow.nii.gz"
    
    if not mask_path.exists():
        alt_mask = base_path / "kits19/data/case_00000/segmentation.nii.gz"
        if alt_mask.exists():
            mask_path = alt_mask
    
    if not flow_path.exists():
        print("Blood flow file not found. Creating simulated flow map...")
        # Создаем симулированную карту кровотока
        img = nib.load(str(mask_path))
        mask = img.get_fdata()
        simulated_flow = np.random.normal(300, 50, mask.shape)
        simulated_flow = np.clip(simulated_flow, 150, 450)
        flow_path = base_path / "06_3d_visualization/simulated_flow.nii.gz"
        nib.save(nib.Nifti1Image(simulated_flow.astype(np.float32), img.affine), str(flow_path))
        print(f"Created simulated flow map at {flow_path}")
    
    if mask_path.exists() and flow_path.exists():
        print(f"Mask: {mask_path}")
        print(f"Flow: {flow_path}")
        plot_flow_on_surface(str(mask_path), str(flow_path))
    else:
        print("Required files not found")
        print(f"Mask exists: {mask_path.exists()}")
        print(f"Flow exists: {flow_path.exists()}")
