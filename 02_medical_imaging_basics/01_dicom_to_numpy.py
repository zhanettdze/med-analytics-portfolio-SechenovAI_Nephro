import numpy as np
import pydicom
from pathlib import Path
from typing import List, Tuple

def load_dicom_series(folder_path: str) -> Tuple[np.ndarray, dict]:
    files = sorted(Path(folder_path).glob("*.dcm"))
    if not files:
        raise FileNotFoundError(f"No DICOM files found in {folder_path}")
    
    slices: List[np.ndarray] = []
    metadata = {}
    
    for f in files:
        ds = pydicom.dcmread(f)
        slices.append(ds.pixel_array.astype(np.float32))
        if not metadata:
            metadata = {
                "pixel_spacing": ds.get("PixelSpacing", [1.0, 1.0]),
                "slice_thickness": ds.get("SliceThickness", 1.0),
                "slope": ds.get("RescaleSlope", 1.0),
                "intercept": ds.get("RescaleIntercept", 0.0)
            }
    
    volume = np.stack(slices, axis=0)
    return volume, metadata

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        folder = sys.argv[1]
        print(f"Loading DICOM from {folder}...")
        volume, meta = load_dicom_series(folder)
        print(f"Loaded {volume.shape[0]} slices, shape={volume.shape[1:]}")
    else:
        print("DICOM loader for CT kidney")
        print("Usage: python 01_dicom_to_numpy.py /path/to/dicom/folder")
        print("\nNote: KiTS19 dataset uses NIfTI format, not DICOM.")
        print("For KiTS19, use 02_nifti_io.py instead.")