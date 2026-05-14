"""
Extract 3D surface mesh from binary segmentation mask.
"""

import numpy as np
import nibabel as nib
from skimage import measure
from pathlib import Path

def extract_surface(mask_path, level=0.5, step=1):
    """
    Extract surface mesh from binary mask.
    
    Parameters:
    - mask_path: path to NIfTI mask file
    - level: isosurface level (0.5 for binary mask)
    - step: subsampling step (higher = coarser mesh)
    
    Returns:
    - verts: vertices coordinates
    - faces: face indices
    - normals: vertex normals
    """
    img = nib.load(mask_path)
    mask = img.get_fdata()
    
    if step > 1:
        mask = mask[::step, ::step, ::step]
    
    verts, faces, normals, _ = measure.marching_cubes(mask, level=level)
    
    return verts, faces, normals

def save_mesh_ply(verts, faces, output_path):
    """Save mesh as PLY file for external viewers (MeshLab, Blender)."""
    with open(output_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(verts)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        
        for v in verts:
            f.write(f"{v[0]} {v[1]} {v[2]}\n")
        
        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
    
    print(f"Mesh saved to {output_path}")

if __name__ == "__main__":
    kits19_path = Path("data/kits19/data/case_00000/")
    mask_path = kits19_path / "segmentation.nii.gz"
    
    if mask_path.exists():
        print("Extracting kidney surface from KiTS19 mask...")
        verts, faces, normals = extract_surface(str(mask_path), step=2)
        
        print(f"Vertices: {len(verts)}")
        print(f"Faces: {len(faces)}")
        
        output_path = Path("kidney_mesh.ply")
        save_mesh_ply(verts, faces, str(output_path))
        
        print(f"\nTo view: open {output_path} in MeshLab or Blender")
    else:
        print("KiTS19 mask not found")
        print("Creating sample sphere mesh for demonstration...")
        
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones_like(u), np.cos(v))
        
        verts = np.column_stack((x.flatten(), y.flatten(), z.flatten()))
        faces = []
        
        print(f"Sample sphere: {len(verts)} vertices")