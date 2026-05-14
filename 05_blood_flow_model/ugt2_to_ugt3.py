import numpy as np
from pathlib import Path

class UGTTransition:
    """Check UGT-2 to UGT-3 readiness."""
    
    def __init__(self):
        self.requirements = {
            "validation_data": False,
            "clinical_thresholds": False,
            "reproducibility": False,
            "error_metrics": False,
            "documentation": False
        }
    
    def check_validation_data(self, has_real_data):
        self.requirements["validation_data"] = has_real_data
        return has_real_data
    
    def check_clinical_thresholds(self, flow_errors, max_allowed_error=50):
        """
        flow_errors: dict with 'bland_altman_limits' and 'correlation'
        """
        passed = flow_errors.get("correlation", 0) > 0.8
        self.requirements["clinical_thresholds"] = passed
        return passed
    
    def check_reproducibility(self, has_seed_fixed, has_version_control):
        passed = has_seed_fixed and has_version_control
        self.requirements["reproducibility"] = passed
        return passed
    
    def check_error_metrics(self, has_dice, has_hausdorff, has_volume):
        passed = has_dice and has_hausdorff and has_volume
        self.requirements["error_metrics"] = passed
        return passed
    
    def is_ugt3_ready(self):
        return all(self.requirements.values())
    
    def generate_report(self):
        print("\n" + "=" * 60)
        print("UGT-2 → UGT-3 TRANSITION CHECKLIST")
        print("=" * 60)
        
        for req, status in self.requirements.items():
            status_str = "[X]" if status else "[ ]"
            print(f"{status_str} {req}")
        
        print("-" * 60)
        print(f"Overall status: {'UGT-3 READY' if self.is_ugt3_ready() else 'UGT-2 (needs improvements)'}")
        
        if not self.is_ugt3_ready():
            missing = [req for req, status in self.requirements.items() if not status]
            print(f"\nMissing requirements: {', '.join(missing)}")
        
        return self.is_ugt3_ready()

def generate_clinical_report(flow_map, ischemia_mask, patient_id="case_00000"):
    """
    Generate clinical report for surgeon.
    """
    total_volume = ischemia_mask.sum()
    ischemic_volume = (ischemia_mask > 0).sum()
    ischemic_percent = ischemic_volume / total_volume * 100 if total_volume > 0 else 0
    
    mean_flow = np.mean(flow_map[flow_map > 0])
    
    report = f"""
CLINICAL REPORT - Patient {patient_id}
========================================
Estimated kidney blood flow: {mean_flow:.1f} ml/min/100g
Ischemic tissue: {ischemic_percent:.1f}% of kidney volume

RECOMMENDATION:
"""
    if ischemic_percent > 20:
        report += "HIGH RISK - Consider revascularization before partial nephrectomy"
    elif ischemic_percent > 10:
        report += "MODERATE RISK - Monitor during surgery"
    else:
        report += "LOW RISK - Standard procedure"
    
    return report

if __name__ == "__main__":
    print("UGT-2 TO UGT-3 TRANSITION")
    print("=" * 40)
    
    ugt = UGTTransition()
    
    ugt.check_validation_data(has_real_data=True)
    ugt.check_clinical_thresholds({"correlation": 0.85})
    ugt.check_reproducibility(has_seed_fixed=True, has_version_control=True)
    ugt.check_error_metrics(has_dice=True, has_hausdorff=True, has_volume=True)
    
    ugt.generate_report()
    
    print("\n" + generate_clinical_report(
        np.random.normal(250, 50, (100, 100)),
        np.random.rand(100, 100) > 0.85
    ))