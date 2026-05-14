import numpy as np
from pathlib import Path

def convert_to_hu(pixel_array: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    return pixel_array * slope + intercept

def hu_to_display(hu: np.ndarray, window_center: int = 40, window_width: int = 400) -> np.ndarray:
    min_hu = window_center - window_width // 2
    max_hu = window_center + window_width // 2
    display = np.clip(hu, min_hu, max_hu)
    display = (display - min_hu) / (max_hu - min_hu)
    return (display * 255).astype(np.uint8)

def get_kidney_hu_range(contrast_phase: str = "parenchymal") -> tuple:
    ranges = {
        "non_contrast": (30, 50),
        "arterial": (50, 150),
        "parenchymal": (100, 200),
        "excretory": (50, 100)
    }
    return ranges.get(contrast_phase, (30, 200))

if __name__ == "__main__":
    print("HU conversion for CT kidney")
    
    # Try to load KiTS19 data if available
    kits19_path = Path("data/kits19/data/case_00000/imaging.nii.gz")
    
    if kits19_path.exists():
        import nibabel as nib
        img = nib.load(str(kits19_path))
        data = img.get_fdata()
        print(f"\nReal KiTS19 data:")
        print(f"HU range: [{data.min():.1f}, {data.max():.1f}]")
        print(f"Mean HU: {data.mean():.1f}")
        print(f"Kidney HU range (parenchymal): {get_kidney_hu_range('parenchymal')}")
    else:
        print(f"\nDemo mode (synthetic data):")
        fake_dicom = np.random.randint(0, 2000, (512, 512))
        hu = convert_to_hu(fake_dicom, slope=1.0, intercept=-1024)
        print(f"HU range: [{hu.min():.0f}, {hu.max():.0f}]")
        print(f"Kidney HU range: {get_kidney_hu_range()}")