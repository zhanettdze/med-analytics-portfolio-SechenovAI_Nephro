"""
Demons registration for CT kidney images.
Based on Thirion's demons algorithm (optical flow).
Suitable for soft tissue registration (kidney, liver).
"""

import time
import SimpleITK as sitk
from pathlib import Path

def resample_volume(image, scale=2):
    """Downsample image by given scale factor."""
    original_size = image.GetSize()
    new_size = [max(1, int(s / scale)) for s in original_size]
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing([image.GetSpacing()[i] * scale for i in range(3)])
    resampler.SetInterpolator(sitk.sitkLinear)
    return resampler.Execute(image)

def register_demons_fast(fixed_path, moving_path, output_path=None, downsample=4):
    """
    Fast Demons registration with downsampling.
    
    Args:
        fixed_path: path to fixed image
        moving_path: path to moving image  
        output_path: output path for registered image
        downsample: downsampling factor (default=4)
    """
    try:
        print("Loading images...")
        fixed_full = sitk.ReadImage(fixed_path)
        moving_full = sitk.ReadImage(moving_path)
        
        print(f"Downsampling by factor {downsample}...")
        fixed_down = resample_volume(fixed_full, downsample)
        moving_down = resample_volume(moving_full, downsample)
        
        print(f"Original size: {fixed_full.GetSize()} -> Downsampled: {fixed_down.GetSize()}")
        
        fixed_down = sitk.Normalize(fixed_down)
        moving_down = sitk.Normalize(moving_down)
        
        print("Performing Demons registration...")
        start = time.time()
        
        demons = sitk.DemonsRegistrationFilter()
        demons.SetNumberOfIterations(100)
        demons.SetStandardDeviations(1.0)
        demons.SetSmoothDisplacementField(True)
        
        displacement_field = demons.Execute(fixed_down, moving_down)
        elapsed = time.time() - start
        
        print(f"Registration completed in {elapsed:.2f} seconds")
        print(f"RMS change: {demons.GetRMSChange():.6f}")
        
        transform = sitk.DisplacementFieldTransform(displacement_field)
        
        resampler_full = sitk.ResampleImageFilter()
        resampler_full.SetReferenceImage(fixed_full)
        resampler_full.SetTransform(transform)
        resampler_full.SetInterpolator(sitk.sitkLinear)
        resampled = resampler_full.Execute(moving_full)
        
        resampled = sitk.Cast(resampled, sitk.sitkFloat32)
        
        displacement_field_full = sitk.TransformToDisplacementField(
            transform,
            sitk.sitkVectorFloat64,
            fixed_full.GetSize(),
            fixed_full.GetOrigin(),
            fixed_full.GetSpacing(),
            fixed_full.GetDirection()
        )
        displacement_field_full = sitk.Cast(displacement_field_full, sitk.sitkVectorFloat32)
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            sitk.WriteImage(resampled, str(output_path))
            print(f"Registered image saved to {output_path}")
            
            disp_path = output_path.parent / f"{output_path.stem}_displacement.nii.gz"
            sitk.WriteImage(displacement_field_full, str(disp_path))
            print(f"Displacement field saved to {disp_path}")
        
        return resampled, displacement_field_full, transform, elapsed
    
    except Exception as e:
        print(f"Error during Demons registration: {e}")
        return None, None, None, None

def compute_jacobian(displacement_field_path, output_path=None):
    """Compute Jacobian determinant from displacement field."""
    try:
        disp = sitk.ReadImage(displacement_field_path)
        jacobian = sitk.DisplacementFieldJacobianDeterminant(disp)
        jacobian = sitk.Cast(jacobian, sitk.sitkFloat32)
        
        if output_path:
            sitk.WriteImage(jacobian, output_path)
            print(f"Jacobian saved to {output_path}")
        
        data = sitk.GetArrayViewFromImage(jacobian)
        print(f"\nJacobian statistics:")
        print(f"  Min: {data.min():.4f}")
        print(f"  Max: {data.max():.4f}")
        print(f"  Mean: {data.mean():.4f}")
        print(f"  Std: {data.std():.4f}")
        
        expanded = (data > 1.0).sum()
        compressed = (data < 1.0).sum()
        total = data.size
        
        print(f"\nTissue deformation:")
        print(f"  Expanded (increased volume): {expanded/total*100:.2f}%")
        print(f"  Compressed (decreased volume): {compressed/total*100:.2f}%")
        
        return jacobian
    
    except Exception as e:
        print(f"Error computing Jacobian: {e}")
        return None

def print_method_comparison():
    """Print comparison of registration methods."""
    print("\n" + "=" * 70)
    print("REGISTRATION METHODS COMPARISON")
    print("=" * 70)
    print("""
Method      | Parameters | Speed | Local deformations
------------|------------|-------|-------------------
Rigid       | 6          | Fast  | No
Affine      | 12         | Fast  | No  
B-spline    | ~1000      | Medium| Yes (grid-based)
Demons      | ~1M        | Slow  | Yes (pixel-wise)

Demons characteristics:
- No manual grid configuration required
- Based on intensity gradients
- Suitable for soft tissues (kidney, liver)
- May produce unrealistic deformations with noisy data
""")

if __name__ == "__main__":
    import sys
    
    kits19_path = Path("data/kits19/data/case_00000/")
    img_path = kits19_path / "imaging.nii.gz"
    
    if len(sys.argv) > 2:
        fixed_path = sys.argv[1]
        moving_path = sys.argv[2]
        output_name = "registered_demons.nii.gz"
        print("=" * 60)
        print("DEMONS REGISTRATION (Clinical mode)")
        print("=" * 60)
        print(f"Fixed: {fixed_path}")
        print(f"Moving: {moving_path}\n")
        
        registered, disp_field, transform, elapsed = register_demons_fast(
            fixed_path, moving_path, output_name, downsample=4
        )
        
        if disp_field is not None:
            compute_jacobian("registered_demons_displacement.nii.gz", "jacobian_demons.nii.gz")
        
        print_method_comparison()
        
    elif img_path.exists():
        print("=" * 60)
        print("DEMONS REGISTRATION (Self-test mode)")
        print("=" * 60)
        print("Registering image with itself as performance test.\n")
        
        registered, disp_field, transform, elapsed = register_demons_fast(
            str(img_path), str(img_path), "registered_demons.nii.gz", downsample=4
        )
        
        print("\n" + "=" * 60)
        print("DEMONS SPECIFIC FEATURES")
        print("=" * 60)
        print("""
Demons registration:
- No mesh configuration needed (unlike B-spline)
- Better suited for soft tissue (kidney, liver)
- Produces smoother deformations
- Displacement field can be used for blood flow modeling

For SechenovAI_Nephro:
The displacement field from Demons can estimate local tissue
compression/expansion, which correlates with regional blood flow.
""")
        
        print_method_comparison()
        
    else:
        print("KiTS19 data not found at data/kits19/")
        print("\nUsage:")
        print("  python 04_demons_registration.py")
        print("  python 04_demons_registration.py fixed.nii moving.nii")