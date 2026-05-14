import time
import SimpleITK as sitk
from pathlib import Path
import argparse

def resample_volume(image, scale=2):
    original_size = image.GetSize()
    new_size = [max(1, int(s / scale)) for s in original_size]
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing([image.GetSpacing()[i] * scale for i in range(3)])
    resampler.SetInterpolator(sitk.sitkLinear)
    return resampler.Execute(image)

def register_rigid(fixed, moving, downsample=4):
    fixed_down = resample_volume(fixed, downsample)
    moving_down = resample_volume(moving, downsample)
    
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.2, seed=42)
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0, minStep=1e-4, numberOfIterations=50, gradientMagnitudeTolerance=1e-5
    )
    
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_down, moving_down, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    registration_method.SetInitialTransform(initial_transform)
    
    transform = registration_method.Execute(fixed_down, moving_down)
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetTransform(transform)
    resampler.SetInterpolator(sitk.sitkLinear)
    registered = resampler.Execute(moving)
    
    return registered, transform

def register_affine(fixed, moving, downsample=4):
    fixed_down = resample_volume(fixed, downsample)
    moving_down = resample_volume(moving, downsample)
    
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.2, seed=42)
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0, minStep=1e-4, numberOfIterations=100, gradientMagnitudeTolerance=1e-5
    )
    
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_down, moving_down, sitk.AffineTransform(3), sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    registration_method.SetInitialTransform(initial_transform)
    
    transform = registration_method.Execute(fixed_down, moving_down)
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetTransform(transform)
    resampler.SetInterpolator(sitk.sitkLinear)
    registered = resampler.Execute(moving)
    
    return registered, transform

def register_bspline(fixed, moving, downsample=4, mesh_size=4):
    fixed_down = resample_volume(fixed, downsample)
    moving_down = resample_volume(moving, downsample)
    
    transform_domain_mesh_size = [mesh_size, mesh_size, mesh_size]
    initial_transform = sitk.BSplineTransformInitializer(fixed_down, transform_domain_mesh_size)
    
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.2, seed=42)
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0, minStep=1e-6, numberOfIterations=200, gradientMagnitudeTolerance=1e-6
    )
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    
    transform = registration_method.Execute(fixed_down, moving_down)
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetTransform(transform)
    resampler.SetInterpolator(sitk.sitkLinear)
    registered = resampler.Execute(moving)
    
    displacement_field = sitk.TransformToDisplacementField(
        transform, sitk.sitkVectorFloat64, fixed.GetSize(),
        fixed.GetOrigin(), fixed.GetSpacing(), fixed.GetDirection()
    )
    
    return registered, transform, displacement_field

def register_demons(fixed, moving, downsample=4):
    fixed_down = resample_volume(fixed, downsample)
    moving_down = resample_volume(moving, downsample)
    
    fixed_down = sitk.Normalize(fixed_down)
    moving_down = sitk.Normalize(moving_down)
    
    demons = sitk.DemonsRegistrationFilter()
    demons.SetNumberOfIterations(100)
    demons.SetStandardDeviations(1.0)
    demons.SetSmoothDisplacementField(True)
    
    displacement_field = demons.Execute(fixed_down, moving_down)
    transform = sitk.DisplacementFieldTransform(displacement_field)
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetTransform(transform)
    resampler.SetInterpolator(sitk.sitkLinear)
    registered = resampler.Execute(moving)
    
    displacement_field_full = sitk.TransformToDisplacementField(
        transform, sitk.sitkVectorFloat64, fixed.GetSize(),
        fixed.GetOrigin(), fixed.GetSpacing(), fixed.GetDirection()
    )
    
    return registered, transform, displacement_field_full

def run_pipeline(fixed_path, moving_path, output_dir="registration_results", methods=None):
    if methods is None:
        methods = ["rigid", "affine", "bspline", "demons"]
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("REGISTRATION PIPELINE")
    print("=" * 60)
    print(f"Fixed: {fixed_path}")
    print(f"Moving: {moving_path}")
    print(f"Methods: {', '.join(methods)}\n")
    
    fixed = sitk.ReadImage(fixed_path)
    moving = sitk.ReadImage(moving_path)
    
    results = {}
    
    if "rigid" in methods:
        print("[1/4] Running rigid registration...")
        start = time.perf_counter()
        registered, transform = register_rigid(fixed, moving)
        elapsed = time.perf_counter() - start
        results["rigid"] = {"registered": registered, "transform": transform, "time": elapsed}
        sitk.WriteImage(registered, str(output_path / "registered_rigid.nii.gz"))
        print(f"  Completed in {elapsed:.2f}s\n")
    
    if "affine" in methods:
        print("[2/4] Running affine registration...")
        start = time.perf_counter()
        registered, transform = register_affine(fixed, moving)
        elapsed = time.perf_counter() - start
        results["affine"] = {"registered": registered, "transform": transform, "time": elapsed}
        sitk.WriteImage(registered, str(output_path / "registered_affine.nii.gz"))
        print(f"  Completed in {elapsed:.2f}s\n")
    
    if "bspline" in methods:
        print("[3/4] Running B-spline registration...")
        start = time.perf_counter()
        registered, transform, disp_field = register_bspline(fixed, moving)
        elapsed = time.perf_counter() - start
        results["bspline"] = {"registered": registered, "transform": transform, "displacement": disp_field, "time": elapsed}
        sitk.WriteImage(registered, str(output_path / "registered_bspline.nii.gz"))
        sitk.WriteImage(disp_field, str(output_path / "displacement_bspline.nii.gz"))
        print(f"  Completed in {elapsed:.2f}s\n")
    
    if "demons" in methods:
        print("[4/4] Running Demons registration...")
        start = time.perf_counter()
        registered, transform, disp_field = register_demons(fixed, moving)
        elapsed = time.perf_counter() - start
        results["demons"] = {"registered": registered, "transform": transform, "displacement": disp_field, "time": elapsed}
        sitk.WriteImage(registered, str(output_path / "registered_demons.nii.gz"))
        sitk.WriteImage(disp_field, str(output_path / "displacement_demons.nii.gz"))
        print(f"  Completed in {elapsed:.2f}s\n")
    
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for method, data in results.items():
        print(f"{method:10s}: {data['time']:.2f}s")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Complete registration pipeline")
    parser.add_argument("fixed", help="Path to fixed image")
    parser.add_argument("moving", help="Path to moving image")
    parser.add_argument("--output", "-o", default="registration_results", help="Output directory")
    parser.add_argument("--methods", "-m", nargs="+", default=["rigid", "affine", "bspline", "demons"],
                        choices=["rigid", "affine", "bspline", "demons"], help="Registration methods to run")
    
    args = parser.parse_args()
    
    run_pipeline(args.fixed, args.moving, args.output, args.methods)