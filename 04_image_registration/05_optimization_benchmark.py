import time
import SimpleITK as sitk
from pathlib import Path
import pandas as pd

def resample_volume(image, scale=2):
    original_size = image.GetSize()
    new_size = [max(1, int(s / scale)) for s in original_size]
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing([image.GetSpacing()[i] * scale for i in range(3)])
    resampler.SetInterpolator(sitk.sitkLinear)
    return resampler.Execute(image)

def benchmark_optimizer(fixed, moving, optimizer_type, learning_rate=1.0, iterations=50):
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.2, seed=42)
    registration_method.SetInterpolator(sitk.sitkLinear)
    
    initial_transform = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.AffineTransform(3), sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    registration_method.SetInitialTransform(initial_transform)
    
    if optimizer_type == "RegularStepGradientDescent":
        registration_method.SetOptimizerAsRegularStepGradientDescent(
            learningRate=learning_rate,
            minStep=1e-4,
            numberOfIterations=iterations,
            gradientMagnitudeTolerance=1e-5
        )
    elif optimizer_type == "GradientDescent":
        registration_method.SetOptimizerAsGradientDescentLineSearch(
            learningRate=learning_rate,
            numberOfIterations=iterations,
            convergenceMinimumValue=1e-5,
            convergenceWindowSize=10
        )
    elif optimizer_type == "ConjugateGradient":
        registration_method.SetOptimizerAsConjugateGradientLineSearch(
            learningRate=learning_rate,
            numberOfIterations=iterations,
            convergenceMinimumValue=1e-5,
            convergenceWindowSize=10
        )
    elif optimizer_type == "Exhaustive":
        registration_method.SetOptimizerAsExhaustive(
            numberOfSteps=[5, 5, 5, 2, 2, 2],
            stepLength=1.0
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    start = time.perf_counter()
    transform = registration_method.Execute(fixed, moving)
    elapsed = time.perf_counter() - start
    
    return {
        "optimizer": optimizer_type,
        "time_sec": elapsed,
        "metric_value": registration_method.GetMetricValue(),
        "iterations": registration_method.GetOptimizerIteration(),
        "learning_rate": learning_rate
    }

def run_benchmark(fixed_path, moving_path, downsample=4):
    print("Loading images...")
    fixed_full = sitk.ReadImage(fixed_path)
    moving_full = sitk.ReadImage(moving_path)
    
    fixed = resample_volume(fixed_full, downsample)
    moving = resample_volume(moving_full, downsample)
    
    print(f"Downsampled size: {fixed.GetSize()}\n")
    
    optimizers = ["RegularStepGradientDescent", "GradientDescent", "ConjugateGradient"]
    results = []
    
    for opt in optimizers:
        print(f"Testing {opt}...")
        result = benchmark_optimizer(fixed, moving, opt, learning_rate=1.0, iterations=50)
        results.append(result)
        print(f"  Time: {result['time_sec']:.2f}s, Metric: {result['metric_value']:.4f}, Iterations: {result['iterations']}")
    
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    
    print("\nRecommendation:")
    best_idx = df["metric_value"].idxmin() if df["metric_value"].min() < 0 else df["time_sec"].idxmin()
    print(f"  Best optimizer: {df.iloc[best_idx]['optimizer']}")
    
    return results

if __name__ == "__main__":
    kits19_path = Path("data/kits19/data/case_00000/")
    img_path = kits19_path / "imaging.nii.gz"
    
    if img_path.exists():
        print("=" * 60)
        print("OPTIMIZER BENCHMARK")
        print("=" * 60)
        print("Comparing different optimizers for affine registration.\n")
        
        run_benchmark(str(img_path), str(img_path), downsample=4)
    else:
        print("KiTS19 data not found at data/kits19/")