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

def register_bspline_fast(fixed_path, moving_path, output_path=None, downsample=4, mesh_size=4):
    """
    Быстрая B-spline (нелинейная) регистрация с даунсемплингом
    
    B-spline трансформация позволяет:
    - Локально деформировать изображение
    - Учитывать сжатие/растяжение тканей
    - Моделировать дыхательные движения
    
    Args:
        fixed_path: путь к фиксированному изображению
        moving_path: путь к движущемуся изображению
        output_path: путь для сохранения результата
        downsample: фактор уменьшения разрешения
        mesh_size: размер контрольной сетки B-spline (чем больше, тем более локальные деформации)
    """
    try:
        print("Loading images...")
        fixed_full = sitk.ReadImage(fixed_path)
        moving_full = sitk.ReadImage(moving_path)
        
        print(f"Downsampling by factor {downsample}...")
        fixed = resample_volume(fixed_full, downsample)
        moving = resample_volume(moving_full, downsample)
        
        print(f"Original size: {fixed_full.GetSize()} -> Downsampled: {fixed.GetSize()}")
        
        transform_domain_mesh_size = [mesh_size, mesh_size, mesh_size]
        print(f"B-spline mesh size: {transform_domain_mesh_size} (control points: {mesh_size}³)")
        
        initial_transform = sitk.BSplineTransformInitializer(fixed, transform_domain_mesh_size)
        
        registration_method = sitk.ImageRegistrationMethod()
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
        registration_method.SetMetricSamplingPercentage(0.2, seed=42)
        registration_method.SetInterpolator(sitk.sitkLinear)
        
        registration_method.SetOptimizerAsRegularStepGradientDescent(
            learningRate=2.0,
            minStep=1e-6,
            numberOfIterations=200,
            gradientMagnitudeTolerance=1e-6
        )
        
        registration_method.SetInitialTransform(initial_transform, inPlace=False)
        
        registration_method.AddCommand(sitk.sitkIterationEvent, lambda: print(f"  Iteration: {registration_method.GetOptimizerIteration()}, Metric: {registration_method.GetMetricValue():.4f}"))
        
        print("Performing B-spline (non-rigid) registration...")
        print("This may take 1-3 minutes due to many parameters...")
        start = time.time()
        final_transform = registration_method.Execute(fixed, moving)
        elapsed = time.time() - start
        
        print(f"\nRegistration completed in {elapsed:.2f} seconds")
        print(f"Final metric value: {registration_method.GetMetricValue():.4f}")
        print(f"Number of B-spline parameters: {final_transform.GetNumberOfParameters()}")
        
        resampler_full = sitk.ResampleImageFilter()
        resampler_full.SetReferenceImage(fixed_full)
        resampler_full.SetTransform(final_transform)
        resampler_full.SetInterpolator(sitk.sitkLinear)
        resampled = resampler_full.Execute(moving_full)
        
        displacement_field = sitk.TransformToDisplacementField(final_transform, 
                                                                sitk.sitkVectorFloat64,
                                                                fixed.GetSize(), 
                                                                fixed.GetOrigin(),
                                                                fixed.GetSpacing(), 
                                                                fixed.GetDirection())
        displacement_field = sitk.Cast(displacement_field, sitk.sitkVectorFloat32)
        
        resampled = sitk.Cast(resampled, sitk.sitkFloat32)
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            sitk.WriteImage(resampled, str(output_path))
            print(f"Registered image saved to {output_path}")
            
            disp_path = output_path.parent / f"{output_path.stem}_displacement.nii.gz"
            sitk.WriteImage(displacement_field, str(disp_path))
            print(f"Displacement field saved to {disp_path}")
        
        return resampled, final_transform, displacement_field, elapsed
    
    except Exception as e:
        print(f"Error during B-spline registration: {e}")
        return None, None, None, None

def compute_jacobian_determinant(displacement_field_path, output_path=None):
    """
    Вычисляет определитель Якобиана из поля деформации.
    det(J) > 1: растяжение ткани
    det(J) < 1: сжатие ткани
    det(J) = 1: сохранение объема
    """
    try:
        disp = sitk.ReadImage(displacement_field_path)
        
        jacobian = sitk.DisplacementFieldJacobianDeterminant(disp)
        jacobian = sitk.Cast(jacobian, sitk.sitkFloat32)
        
        if output_path:
            sitk.WriteImage(jacobian, output_path)
            print(f"Jacobian determinant saved to {output_path}")
        
        jacobian_array = sitk.GetArrayViewFromImage(jacobian)
        print(f"\nJacobian determinant statistics:")
        print(f"  Min: {jacobian_array.min():.4f}")
        print(f"  Max: {jacobian_array.max():.4f}")
        print(f"  Mean: {jacobian_array.mean():.4f}")
        print(f"  Std: {jacobian_array.std():.4f}")
        
        return jacobian
    
    except Exception as e:
        print(f"Error computing Jacobian: {e}")
        return None

if __name__ == "__main__":
    import sys
    
    kits19_path = Path("data/kits19/data/case_00000/")
    img_path = kits19_path / "imaging.nii.gz"
    
    if len(sys.argv) > 2:
        fixed_path = sys.argv[1]
        moving_path = sys.argv[2]
        output_name = "registered_bspline.nii.gz"
        print("=" * 60)
        print("B-SPLINE REGISTRATION (Real phases)")
        print("=" * 60)
        print(f"Fixed: {fixed_path}")
        print(f"Moving: {moving_path}\n")
        
        registered, transform, disp_field, elapsed = register_bspline_fast(
            fixed_path, moving_path, output_name, downsample=4, mesh_size=4
        )
        
        if disp_field is not None:
            compute_jacobian_determinant("registered_bspline_displacement.nii.gz", "jacobian.nii.gz")
        
    elif img_path.exists():
        print("=" * 60)
        print("B-SPLINE REGISTRATION (Optimized)")
        print("=" * 60)
        print("Registering image with itself as a performance test.\n")
        
        registered, transform, disp_field, elapsed = register_bspline_fast(
            str(img_path), str(img_path), "registered_bspline.nii.gz", downsample=4, mesh_size=4
        )
        
        print("\n" + "=" * 60)
        print("КЛЮЧЕВОЕ ОТЛИЧИЕ ОТ RIGID/AFFINE")
        print("=" * 60)
        print("""
B-spline регистрация позволяет:
- Деформировать изображение локально (не только поворачивать/масштабировать)
- Сохранить поле деформации для дальнейшего анализа
- Вычислить якобиан деформации (растяжение/сжатие ткани)
- Использовать результаты для модели кровотока (УГТ-2 → УГТ-3)
""")
        
    else:
        print("KiTS19 data not found at data/kits19/")
        print("\nUsage:")
        print("  python 03_bspline_registration.py                        # Self-test on KiTS19")
        print("  python 03_bspline_registration.py fixed.nii moving.nii  # Register two phases")