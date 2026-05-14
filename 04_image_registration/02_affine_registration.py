import time
import SimpleITK as sitk
from pathlib import Path

def resample_volume(image, scale=2):
    """Уменьшает разрешение изображения в scale раз"""
    original_size = image.GetSize()
    new_size = [max(1, int(s / scale)) for s in original_size]
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing([image.GetSpacing()[i] * scale for i in range(3)])
    resampler.SetInterpolator(sitk.sitkLinear)
    return resampler.Execute(image)

def register_affine_fast(fixed_path, moving_path, output_path=None, downsample=4):
    """
    Быстрая аффинная регистрация с даунсемплингом
    
    Аффинная трансформация включает:
    - Поворот (3 параметра)
    - Сдвиг (3 параметра)  
    - Масштабирование (3 параметра)
    - Скос (3 параметра)
    Итого: 12 параметров (против 6 у жесткой)
    """
    try:
        print("Loading images...")
        fixed_full = sitk.ReadImage(fixed_path)
        moving_full = sitk.ReadImage(moving_path)
        
        print(f"Downsampling by factor {downsample}...")
        fixed = resample_volume(fixed_full, downsample)
        moving = resample_volume(moving_full, downsample)
        
        print(f"Original size: {fixed_full.GetSize()} -> Downsampled: {fixed.GetSize()}")
        
        registration_method = sitk.ImageRegistrationMethod()
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
        registration_method.SetMetricSamplingPercentage(0.2, seed=42)
        registration_method.SetInterpolator(sitk.sitkLinear)
        
        registration_method.SetOptimizerAsRegularStepGradientDescent(
            learningRate=1.0,
            minStep=1e-4,
            numberOfIterations=100,  
            gradientMagnitudeTolerance=1e-5
        )
        

        initial_transform = sitk.CenteredTransformInitializer(
            fixed, moving, sitk.AffineTransform(3), sitk.CenteredTransformInitializerFilter.GEOMETRY
        )
        registration_method.SetInitialTransform(initial_transform)
        

        registration_method.AddCommand(sitk.sitkIterationEvent, lambda: print(f"  Iteration: {registration_method.GetOptimizerIteration()}, Metric: {registration_method.GetMetricValue():.4f}"))
        
        print("Performing affine registration (more parameters, may take longer)...")
        start = time.time()
        final_transform = registration_method.Execute(fixed, moving)
        elapsed = time.time() - start
        
        print(f"\nRegistration completed in {elapsed:.2f} seconds")
        print(f"Final metric value: {registration_method.GetMetricValue():.4f}")
        
        transform_params = final_transform.GetParameters()
        print(f"Transform parameters (12): {[round(p, 3) for p in transform_params]}")
        
        resampler_full = sitk.ResampleImageFilter()
        resampler_full.SetReferenceImage(fixed_full)
        resampler_full.SetTransform(final_transform)
        resampler_full.SetInterpolator(sitk.sitkLinear)
        resampled = resampler_full.Execute(moving_full)
        
        resampled = sitk.Cast(resampled, sitk.sitkFloat32)
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            sitk.WriteImage(resampled, str(output_path))
            print(f"Registered image saved to {output_path}")
        
        return resampled, final_transform, elapsed
    
    except Exception as e:
        print(f"Error during affine registration: {e}")
        return None, None, None

def compare_fixed_and_moving(fixed_path, moving_path):
    """Сравнивает два изображения перед регистрацией"""
    fixed = sitk.ReadImage(fixed_path)
    moving = sitk.ReadImage(moving_path)
    
    print("\n" + "=" * 60)
    print("СРАВНЕНИЕ ИЗОБРАЖЕНИЙ ДО РЕГИСТРАЦИИ")
    print("=" * 60)
    print(f"Fixed image:  {fixed.GetSize()} | Spacing: {fixed.GetSpacing()}")
    print(f"Moving image: {moving.GetSize()} | Spacing: {moving.GetSpacing()}")
    
    matcher = sitk.ImageRegistrationMethod()
    matcher.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    initial_metric = matcher.MetricEvaluate(fixed, moving)
    print(f"\nInitial Mutual Information: {initial_metric:.4f}")
    return initial_metric

if __name__ == "__main__":
    import sys
    
    kits19_path = Path("data/kits19/data/case_00000/")
    img_path = kits19_path / "imaging.nii.gz"
    
    if len(sys.argv) > 2:
        fixed_path = sys.argv[1]
        moving_path = sys.argv[2]
        output_name = "registered_affine.nii.gz"
        print("=" * 60)
        print("AFFINE REGISTRATION (Real phases)")
        print("=" * 60)
        print(f"Fixed: {fixed_path}")
        print(f"Moving: {moving_path}\n")
        
        compare_fixed_and_moving(fixed_path, moving_path)
        register_affine_fast(fixed_path, moving_path, output_name, downsample=4)
        
    elif img_path.exists():
        print("=" * 60)
        print("AFFINE REGISTRATION (Optimized)")
        print("=" * 60)
        print("Registering image with itself as a performance test.\n")
        
        compare_fixed_and_moving(str(img_path), str(img_path))
        register_affine_fast(str(img_path), str(img_path), "registered_affine.nii.gz", downsample=4)
        
    else:
        print("KiTS19 data not found at data/kits19/")
        print("\nUsage:")
        print("  python 02_affine_registration.py                        # Self-test on KiTS19")
        print("  python 02_affine_registration.py fixed.nii moving.nii  # Register two phases")