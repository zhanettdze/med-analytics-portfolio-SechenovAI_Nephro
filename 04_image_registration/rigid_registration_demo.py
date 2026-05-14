import SimpleITK as sitk
import matplotlib.pyplot as plt
from pathlib import Path

def rigid_registration_demo(fixed_path, moving_path, output_plot="registration_comparison.png", downsample=4):
    fixed_full = sitk.ReadImage(fixed_path)
    moving_full = sitk.ReadImage(moving_path)
    
    factor = downsample
    new_size = [int(s / factor) for s in fixed_full.GetSize()]
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing([fixed_full.GetSpacing()[i] * factor for i in range(3)])
    resampler.SetInterpolator(sitk.sitkLinear)
    
    fixed = resampler.Execute(fixed_full)
    moving = resampler.Execute(moving_full)
    
    print(f"Downsampled: {fixed.GetSize()} (from {fixed_full.GetSize()})")
    
    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(0.2, seed=42)
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0, minStep=1e-4, numberOfIterations=50
    )
    
    initial = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    registration.SetInitialTransform(initial)
    
    print("Performing rigid registration...")
    transform = registration.Execute(fixed, moving)
    mi_value = registration.GetMetricValue()
    print(f"Final Mutual Information: {mi_value:.4f}")
    
    resampler_full = sitk.ResampleImageFilter()
    resampler_full.SetReferenceImage(fixed_full)
    resampler_full.SetTransform(transform)
    resampler_full.SetInterpolator(sitk.sitkLinear)
    registered = resampler_full.Execute(moving_full)
    
    fixed_arr = sitk.GetArrayViewFromImage(fixed)
    moving_arr = sitk.GetArrayViewFromImage(moving)
    reg_arr = sitk.GetArrayViewFromImage(registered)
    
    slice_idx = fixed_arr.shape[0] // 2
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(fixed_arr[slice_idx], cmap='gray')
    axes[0].set_title('Fixed (Reference)')
    axes[0].axis('off')
    
    axes[1].imshow(moving_arr[slice_idx], cmap='gray')
    axes[1].set_title('Moving (Before)')
    axes[1].axis('off')
    
    axes[2].imshow(reg_arr[slice_idx], cmap='gray')
    axes[2].set_title('Registered (After)')
    axes[2].axis('off')
    
    plt.suptitle(f'Rigid Registration | MI = {mi_value:.4f}')
    plt.tight_layout()
    plt.savefig(output_plot, dpi=150)
    print(f"Saved: {output_plot}")
    plt.show()
    
    return transform, mi_value

if __name__ == "__main__":
    kits19_path = Path("data/kits19/data/case_00000/")
    img_path = kits19_path / "imaging.nii.gz"
    
    if img_path.exists():
        rigid_registration_demo(str(img_path), str(img_path))
    else:
        print("KiTS19 not found")
