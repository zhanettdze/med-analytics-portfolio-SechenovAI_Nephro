import numpy as np
import nibabel as nib
from pathlib import Path

def get_voxel_volume_corrected(affine: np.ndarray) -> float:
    spacing = np.abs(np.diag(affine[:3, :3]))
    if spacing[2] > 10:
        # Если аффинная матрица нестандартная, используем известные значения KiTS19
        return 0.5 * 0.5 * 3.0  # типичный воксель KiTS19: 0.5×0.5×3 мм = 0.375 мл
    return float(np.prod(spacing))

def calculate_volume_corrected(mask: np.ndarray, voxel_volume_ml: float) -> float:
    return mask.sum() * voxel_volume_ml

def get_actual_voxel_volume(img: nib.Nifti1Image) -> float:
    header = img.header
    zooms = header.get_zooms()
    spacing_z, spacing_y, spacing_x = zooms
    volume_ml = spacing_x * spacing_y * spacing_z / 1000
    return volume_ml

if __name__ == "__main__":
    kits19_path = Path("data/kits19/data/case_00000/")
    gt_path = kits19_path / "segmentation.nii.gz"
    img_path = kits19_path / "imaging.nii.gz"
    
    if gt_path.exists() and img_path.exists():
        gt_img = nib.load(str(gt_path))
        gt = gt_img.get_fdata()
        
        # Правильный способ получить объем вокселя
        voxel_volume_ml = get_actual_voxel_volume(gt_img)
        zooms = gt_img.header.get_zooms()
        
        print("=" * 60)
        print("ОБЪЕМНЫЕ МЕТРИКИ на данных KiTS19 (case_00000)")
        print("=" * 60)
        print(f"Voxel spacing (мм): z={zooms[0]:.2f}, y={zooms[1]:.2f}, x={zooms[2]:.2f}")
        print(f"Voxel volume: {voxel_volume_ml * 1000:.2f} мм³ = {voxel_volume_ml:.4f} мл\n")
        
        kidney_voxels = (gt == 1).sum()
        tumor_voxels = (gt == 2).sum()
        
        kidney_ml = kidney_voxels * voxel_volume_ml
        tumor_ml = tumor_voxels * voxel_volume_ml
        
        print("Объемы (из ground truth):")
        print(f"  Kidney: {kidney_ml:.1f} мл ({kidney_voxels:,} вокселей)")
        print(f"  Tumor:  {tumor_ml:.1f} мл ({tumor_voxels:,} вокселей)")
        print(f"  Total:  {kidney_ml + tumor_ml:.1f} мл")
        
        if tumor_ml > 0:
            tumor_percent = tumor_ml / kidney_ml * 100
            print(f"  Tumor/Kidney ratio: {tumor_percent:.1f}%")
        
        print("\n" + "=" * 60)
        print("КЛИНИЧЕСКАЯ ИНТЕРПРЕТАЦИЯ")
        print("=" * 60)
        
        if tumor_ml == 0:
            print("No tumor detected in ground truth.")
        elif tumor_ml < 10:
            print("Small tumor (<10 ml) — possible partial nephrectomy.")
        elif tumor_ml < 50:
            print("Medium tumor (10-50 ml) — standard resection.")
        else:
            print(f"Large tumor ({tumor_ml:.1f} ml >50 ml) — consider radical nephrectomy.")
        
        print("\n" + "=" * 60)
        print("СРАВНЕНИЕ С HU-СЕГМЕНТАЦИЕЙ")
        print("=" * 60)
        
        img_data = nib.load(str(img_path)).get_fdata()
        simple_pred = (img_data >= 100) & (img_data <= 200)
        simple_pred = simple_pred.astype(np.uint8)
        
        pred_kidney_voxels = simple_pred.sum()
        pred_kidney_ml = pred_kidney_voxels * voxel_volume_ml
        
        print(f"Ground truth kidney: {kidney_ml:.1f} мл")
        print(f"HU-based kidney:     {pred_kidney_ml:.1f} мл")
        print(f"Difference:          {abs(kidney_ml - pred_kidney_ml):.1f} мл")
        
        if abs(kidney_ml - pred_kidney_ml) > 50:
            print("\nHU-segmentation failed completely for this patient.")
            print("   The kidney HU range (100-200) is incorrect for this case.")
            print("   Need adaptive threshold or deep learning approach.")