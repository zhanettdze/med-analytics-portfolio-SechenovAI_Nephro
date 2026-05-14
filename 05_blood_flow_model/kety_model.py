import numpy as np
from scipy.optimize import curve_fit

def kety_model(t, blood_flow, extraction_fraction, delay=0):
    """
    Kety perfusion model.
    
    Parameters:
    - blood_flow: blood flow in ml/100g/min
    - extraction_fraction: extraction fraction (0-1)
    - delay: time delay in seconds
    
    Returns:
    - concentration curve over time
    """
    t_shifted = t - delay
    mask = t_shifted > 0
    result = np.zeros_like(t, dtype=np.float64)
    result[mask] = blood_flow * extraction_fraction * (1 - np.exp(-blood_flow * t_shifted[mask] / 100))
    return result

def fit_kety_model(time_series, concentration, p0=None):
    """
    Fit Kety model to time-concentration curve.
    
    Parameters:
    - time_series: time points in seconds
    - concentration: contrast concentration over time
    - p0: initial guess [blood_flow, extraction_fraction, delay]
    
    Returns:
    - fitted parameters and covariance
    """
    if p0 is None:
        p0 = [50.0, 0.5, 5.0]
    
    try:
        params, cov = curve_fit(kety_model, time_series, concentration, p0=p0, maxfev=5000)
        return params, cov
    except Exception as e:
        print(f"Fitting failed: {e}")
        return None, None

def calculate_perfusion(ct_curve, aif_curve, time_resolution=1.0):
    """
    Calculate perfusion using deconvolution method.
    
    Parameters:
    - ct_curve: tissue concentration curve
    - aif_curve: arterial input function
    - time_resolution: time between measurements in seconds
    
    Returns:
    - perfusion map
    """
    from scipy.fft import fft, ifft
    
    n = len(ct_curve)
    ct_fft = fft(ct_curve)
    aif_fft = fft(aif_curve)
    
    epsilon = 1e-3
    h_fft = ct_fft / (aif_fft + epsilon * np.max(np.abs(aif_fft)))
    impulse_response = np.real(ifft(h_fft))
    
    perfusion = np.max(impulse_response[:n//2])
    return perfusion

if __name__ == "__main__":
    t = np.linspace(0, 120, 61)
    
    true_params = [60.0, 0.6, 10.0]
    concentration = kety_model(t, *true_params)
    
    fitted_params, _ = fit_kety_model(t, concentration)
    
    print("KETY MODEL DEMONSTRATION")
    print("=" * 40)
    print(f"True parameters: BF={true_params[0]:.1f}, E={true_params[1]:.2f}, delay={true_params[2]:.1f}")
    if fitted_params is not None:
        print(f"Fitted parameters: BF={fitted_params[0]:.1f}, E={fitted_params[1]:.2f}, delay={fitted_params[2]:.1f}")