import time
import SimpleITK as sitk
from pathlib import Path

def resample_volume(image, scale=2):
    """Уменьшает разрешение в scale раз"""
    original_size = image.GetSize()
    new_size = [int(s / scale) for s in original_size]
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing([image.GetSpacing()[i] * scale for i in range(3)])
    resampler.SetInterpolator(sitk.sitkLinear)
    return resampler.Execute(image)

def register_rigid_fast(fixed_path, moving_path, output_path=None, downsample=4):
    fixed_full = sitk.ReadImage(fixed_path)
    moving_full = sitk.ReadImage(moving_path)
    
    # Даунсемплинг для ускорения
    fixed = resample_volume(fixed_full, downsample)
    moving = resample_volume(moving_full, downsample)
    
    print(f"Original size: {fixed_full.GetSize()} → Downsampled: {fixed.GetSize()}")
    
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.2)
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0,
        minStep=1e-4,
        numberOfIterations=50,
        gradientMagnitudeTolerance=1e-5
    )
    
    initial_transform = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    registration_method.SetInitialTransform(initial_transform)
    
    print("Performing rigid registration (downsampled)...")
    start = time.time()
    final_transform = registration_method.Execute(fixed, moving)
    elapsed = time.time() - start
    
    print(f"Registration completed in {elapsed:.2f} seconds")
    print(f"Final metric value: {registration_method.GetMetricValue():.4f}")
    
    # Применяем найденное преобразование к полному разрешению
    resampler_full = sitk.ResampleImageFilter()
    resampler_full.SetReferenceImage(fixed_full)
    resampler_full.SetTransform(final_transform)
    resampler_full.SetInterpolator(sitk.sitkLinear)
    resampled = resampler_full.Execute(moving_full)
    
    if output_path:
        sitk.WriteImage(resampled, output_path)
        print(f"Registered image saved to {output_path}")
    
    return resampled, final_transform, elapsed

if __name__ == "__main__":
    kits19_path = Path("data/kits19/data/case_00000/")
    img_path = kits19_path / "imaging.nii.gz"
    
    if img_path.exists():
        print("=" * 60)
        print("RIGID REGISTRATION (оптимизированная версия)")
        print("=" * 60)
        print("Note: Registration performed on downsampled volume for speed.\n")
        
        registered, transform, elapsed = register_rigid_fast(
            str(img_path), str(img_path), "registered_rigid.nii.gz", downsample=4
        )
        
        print("\n For real clinical use, apply the transform to original resolution.")
    else:
        print("KiTS19 data not found at data/kits19/")