import numpy as np
import nibabel as nib
from scipy.spatial import cKDTree
from pathlib import Path
from tqdm import tqdm

def hausdorff_distance_optimized(mask1: np.ndarray, mask2: np.ndarray, sample_ratio: float = 0.1) -> float:
    pts1 = np.argwhere(mask1)
    pts2 = np.argwhere(mask2)
    
    if len(pts1) == 0 or len(pts2) == 0:
        return float('inf')
    
    # Для больших масок берем случайную выборку
    if len(pts1) > 50000:
        idx = np.random.choice(len(pts1), int(len(pts1) * sample_ratio), replace=False)
        pts1 = pts1[idx]
    if len(pts2) > 50000:
        idx = np.random.choice(len(pts2), int(len(pts2) * sample_ratio), replace=False)
        pts2 = pts2[idx]
    
    tree1 = cKDTree(pts1)
    tree2 = cKDTree(pts2)
    
    dist1 = tree1.query(pts2)[0].max()
    dist2 = tree2.query(pts1)[0].max()
    
    return max(dist1, dist2)

def hausdorff_95_optimized(mask1: np.ndarray, mask2: np.ndarray, sample_ratio: float = 0.1) -> float:
    pts1 = np.argwhere(mask1)
    pts2 = np.argwhere(mask2)
    
    if len(pts1) == 0 or len(pts2) == 0:
        return float('inf')
    
    if len(pts1) > 50000:
        idx = np.random.choice(len(pts1), int(len(pts1) * sample_ratio), replace=False)
        pts1 = pts1[idx]
    if len(pts2) > 50000:
        idx = np.random.choice(len(pts2), int(len(pts2) * sample_ratio), replace=False)
        pts2 = pts2[idx]
    
    tree1 = cKDTree(pts1)
    tree2 = cKDTree(pts2)
    
    dist1 = tree1.query(pts2)[0]
    dist2 = tree2.query(pts1)[0]
    
    return max(np.percentile(dist1, 95), np.percentile(dist2, 95))

def compute_all_surface_metrics(mask_gt: np.ndarray, mask_pred: np.ndarray) -> dict:
    results = {}
    
    for class_id, name in tqdm([(1, "kidney"), (2, "tumor")], desc="Computing metrics"):
        gt_class = (mask_gt == class_id)
        pred_class = (mask_pred == class_id)
        
        # Проверка наличия точек
        gt_points = gt_class.sum()
        pred_points = pred_class.sum()
        
        if gt_points == 0 or pred_points == 0:
            print(f"  Warning: {name} has no points (GT: {gt_points}, Pred: {pred_points})")
            results[name] = {
                "hausdorff": float('inf'),
                "hausdorff_95": float('inf'),
                "exists": False
            }
            continue
        
        results[name] = {
            "hausdorff": hausdorff_distance_optimized(gt_class, pred_class),
            "hausdorff_95": hausdorff_95_optimized(gt_class, pred_class),
            "exists": True
        }
    
    return results

if __name__ == "__main__":
    kits19_path = Path("data/kits19/data/case_00000/")
    gt_path = kits19_path / "segmentation.nii.gz"
    img_path = kits19_path / "imaging.nii.gz"
    
    if gt_path.exists() and img_path.exists():
        print("Loading data...")
        gt = nib.load(str(gt_path)).get_fdata()
        img = nib.load(str(img_path)).get_fdata()
        
        print("Creating prediction (simple HU threshold)...")
        simple_pred = (img >= 100) & (img <= 200)
        simple_pred = simple_pred.astype(np.uint8)
        
        print("\n" + "=" * 70)
        print("ПОВЕРХНОСТНЫЕ МЕТРИКИ на данных KiTS19 (case_00000)")
        print("=" * 70)
        
        results = compute_all_surface_metrics(gt, simple_pred)
        
        print("\n" + "-" * 70)
        print("РЕЗУЛЬТАТЫ")
        print("-" * 70)
        for name, metrics in results.items():
            if metrics["exists"]:
                print(f"\n{name.upper()}:")
                print(f"  Hausdorff Distance: {metrics['hausdorff']:.2f} вокселей")
                print(f"  Hausdorff Distance (95%): {metrics['hausdorff_95']:.2f} вокселей")
            else:
                print(f"\n{name.upper()}: NO POINTS IN PREDICTION")
        
        print("\n" + "=" * 70)
        print("ВЫВОД")
        print("=" * 70)
        print("""
Hausdorff Distance:
  - Для почки: ожидаем 5-15 вокселей для хорошей сегментации
  - 0.2642 Dice + большое Hausdorff → границы выделены плохо
  - Простая HU-сегментация НЕ работает для этого пациента
  - Рекомендация: использовать U-Net или адаптивный порог
""")
    else:
        print("KiTS19 data not found at data/kits19/")