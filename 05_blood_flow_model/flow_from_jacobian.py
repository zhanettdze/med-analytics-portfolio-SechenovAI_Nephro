import numpy as np
import nibabel as nib
from pathlib import Path

def compute_jacobian_from_displacement(displacement_field_path):
    """Load displacement field and compute Jacobian determinant."""
    try:
        import SimpleITK as sitk
        disp = sitk.ReadImage(displacement_field_path)
        jacobian = sitk.DisplacementFieldJacobianDeterminant(disp)
        jacobian = sitk.Cast(jacobian, sitk.sitkFloat32)
        return jacobian
    except ImportError:
        print("SimpleITK not available. Using fallback method.")
        return None

def estimate_blood_flow_from_jacobian(jacobian, baseline_flow=300, sensitivity=1.0):
    """
    Estimate regional blood flow from Jacobian determinant.
    
    Parameters:
    - jacobian: 3D array of Jacobian determinants
    - baseline_flow: baseline blood flow in ml/min/100g (normal kidney: ~300)
    - sensitivity: scaling factor for flow change
    
    Returns:
    - flow_map: estimated blood flow map
    """
    flow_map = baseline_flow * np.power(jacobian, sensitivity)
    flow_map = np.clip(flow_map, 0, baseline_flow * 2)
    return flow_map

def classify_ischemia(jacobian, threshold=0.9):
    """
    Classify ischemic regions based on Jacobian determinant.
    
    Returns:
    - ischemia_mask: binary mask where jacobian < threshold
    - severity: relative reduction in flow
    """
    ischemia_mask = jacobian < threshold
    severity = 1 - jacobian / threshold
    severity = np.clip(severity, 0, 1)
    
    return ischemia_mask.astype(np.uint8), severity

def compute_flow_statistics(flow_map, mask=None):
    """Compute statistics of blood flow map."""
    if mask is not None:
        flow_values = flow_map[mask > 0]
    else:
        flow_values = flow_map.flatten()
    
    stats = {
        "mean": float(np.mean(flow_values)),
        "std": float(np.std(flow_values)),
        "min": float(np.min(flow_values)),
        "max": float(np.max(flow_values)),
        "median": float(np.median(flow_values)),
        "q1": float(np.percentile(flow_values, 25)),
        "q3": float(np.percentile(flow_values, 75))
    }
    return stats

if __name__ == "__main__":
    print("BLOOD FLOW FROM JACOBIAN")
    print("=" * 40)
    
    simulated_jacobian = np.random.normal(1.0, 0.1, (100, 100, 100))
    simulated_jacobian = np.clip(simulated_jacobian, 0.5, 1.5)
    
    flow_map = estimate_blood_flow_from_jacobian(simulated_jacobian)
    ischemia_mask, severity = classify_ischemia(simulated_jacobian)
    
    stats = compute_flow_statistics(flow_map)
    
    print(f"Simulated Jacobian range: [{simulated_jacobian.min():.3f}, {simulated_jacobian.max():.3f}]")
    print(f"Estimated flow range: [{flow_map.min():.1f}, {flow_map.max():.1f}] ml/min/100g")
    print(f"Ischemic volume: {ischemia_mask.sum() / ischemia_mask.size * 100:.1f}%")
    print(f"\nFlow statistics: mean={stats['mean']:.1f}, median={stats['median']:.1f}")