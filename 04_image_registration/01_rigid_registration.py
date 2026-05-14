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

def register_rigid_fast(fixed_path, moving_path, output_path=None, downsample=4):
    """
    Быстрая жесткая регистрация с даунсемплингом
    
    Args:
        fixed_path: путь к фиксированному изображению
        moving_path: путь к движущемуся изображению
        output_path: путь для сохранения результата
        downsample: фактор уменьшения разрешения (4 = уменьшение в 4 раза)
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
        registration_method.SetMetricSamplingPercentage(0.2, seed=42)  # Фиксация seed для воспроизводимости
        registration_method.SetInterpolator(sitk.sitkLinear)
        registration_method.SetOptimizerAsRegularStepGradientDescent(
            learningRate=1.0,
            minStep=1e-4,
            numberOfIterations=50,
            gradientMagnitudeTolerance=1e-5
        )
        
        # Включение логирования метрик на каждой итерации
        registration_method.AddCommand(sitk.sitkIterationEvent, lambda: print(f"  Iteration: {registration_method.GetOptimizerIteration()}, Metric: {registration_method.GetMetricValue():.4f}"))
        
        initial_transform = sitk.CenteredTransformInitializer(
            fixed, moving, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY
        )
        registration_method.SetInitialTransform(initial_transform)
        
        print("Performing rigid registration...")
        start = time.time()
        final_transform = registration_method.Execute(fixed, moving)
        elapsed = time.time() - start
        
        print(f"\nRegistration completed in {elapsed:.2f} seconds")
        print(f"Final metric value: {registration_method.GetMetricValue():.4f}")
        
        # Применение трансформации к полному разрешению
        resampler_full = sitk.ResampleImageFilter()
        resampler_full.SetReferenceImage(fixed_full)
        resampler_full.SetTransform(final_transform)
        resampler_full.SetInterpolator(sitk.sitkLinear)
        resampled = resampler_full.Execute(moving_full)
        
        # Конвертация в float32 для совместимости с ITK-SNAP и другими инструментами
        resampled = sitk.Cast(resampled, sitk.sitkFloat32)
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            sitk.WriteImage(resampled, str(output_path))
            print(f"Registered image saved to {output_path}")
        
        return resampled, final_transform, elapsed
    
    except Exception as e:
        print(f"Error during registration: {e}")
        return None, None, None

def compare_fixed_and_moving(fixed_path, moving_path):
    """Сравнивает два изображения перед регистрацией"""
    fixed = sitk.ReadImage(fixed_path)
    moving = sitk.ReadImage(moving_path)
    
    print("\n" + "=" * 60)
    print("СРАВНЕНИЕ ИЗОБРАЖЕНИЙ ДО РЕГИСТРАЦИИ")
    print("=" * 60)
    print(f"Fixed image:  {fixed.GetSize()} | {fixed.GetSpacing()} | {fixed.GetPixelIDTypeAsString()}")
    print(f"Moving image: {moving.GetSize()} | {moving.GetSpacing()} | {moving.GetPixelIDTypeAsString()}")
    
    # Вычисление начального MI
    matcher = sitk.ImageRegistrationMethod()
    matcher.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    initial_mi = matcher.MetricEvaluate(fixed, moving)
    print(f"\nInitial Mutual Information: {initial_mi:.4f}")
    return initial_mi

if __name__ == "__main__":
    import sys
    
    kits19_path = Path("data/kits19/data/case_00000/")
    img_path = kits19_path / "imaging.nii.gz"
    
    # Поддержка аргументов командной строки для разных фаз
    if len(sys.argv) > 2:
        fixed_path = sys.argv[1]
        moving_path = sys.argv[2]
        output_name = "registered_phase.nii.gz"
        print("=" * 60)
        print("RIGID REGISTRATION (Real phases)")
        print("=" * 60)
        print(f"Fixed: {fixed_path}")
        print(f"Moving: {moving_path}\n")
        
        compare_fixed_and_moving(fixed_path, moving_path)
        register_rigid_fast(fixed_path, moving_path, output_name, downsample=4)
    elif img_path.exists():
        print("=" * 60)
        print("RIGID REGISTRATION (Optimized)")
        print("=" * 60)
        print("Registering image with itself as a performance test.\n")
        
        compare_fixed_and_moving(str(img_path), str(img_path))
        register_rigid_fast(str(img_path), str(img_path), "registered_rigid.nii.gz", downsample=4)
    else:
        print("KiTS19 data not found at data/kits19/")
        print("\nUsage:")
        print("  python 01_rigid_registration.py                    # Self-test on KiTS19")
        print("  python 01_rigid_registration.py fixed.nii moving.nii # Register two phases")