import numpy as np
from scipy.optimize import curve_fit
from scipy.special import wofz

# === Peak Functions ===
def gaussian(x, amp, cen, wid):
    return amp * np.exp(-(x - cen)**2 / (2 * wid**2))

def lorentzian(x, amp, cen, wid):
    return amp * (wid**2 / ((x - cen)**2 + wid**2))

def pseudo_voigt(x, amp, cen, wid, eta=0.5):
    g = gaussian(x, amp, cen, wid)
    l = lorentzian(x, amp, cen, wid)
    return eta * l + (1 - eta) * g

# === Main Fitting Function ===
def fit_peaks(x, y, peak_defs, eta=0.5):
    """
    Fit mixed model Raman peaks to the given spectrum.

    Parameters:
        x (array): Raman shift axis
        y (array): Intensity values
        peak_defs (list): List of tuples like (model, amp, center, width)
        eta (float): Fixed Voigt mixing parameter (default = 0.5)

    Returns:
        y_fit_total (array)
        fitted_peaks (list of (x, y) arrays for each peak)
        peak_params (list of dicts with peak info)
    """

    # === Build parameter guesses and bounds ===
    init = []
    bounds_lower = []
    bounds_upper = []

    for model, amp, cen, wid in peak_defs:
        init.extend([amp, cen, wid])
        bounds_lower.extend([0, cen - 25, 1])
        bounds_upper.extend([2*amp, cen + 25, 100])

    init.append(0.0)  # offset
    bounds_lower.append(-0.2)
    bounds_upper.append(0.2)

    # === Mixed model ===
    def mixed_model(x, *params):
        y_total = np.zeros_like(x)
        for i, (model, _, _, _) in enumerate(peak_defs):
            amp, cen, wid = params[3*i:3*i+3]
            if model == "gauss":
                y_total += gaussian(x, amp, cen, wid)
            elif model == "lorentz":
                y_total += lorentzian(x, amp, cen, wid)
            elif model == "pvoigt":
                y_total += pseudo_voigt(x, amp, cen, wid, eta)
            else:
                raise ValueError(f"[!] Unknown model: {model}")
        return y_total + params[-1]

    # === Fit ===
    popt, pcov = curve_fit(
        mixed_model, x, y,
        p0=init, bounds=(bounds_lower, bounds_upper),
        maxfev=200000
    )

    # === Generate total fit and individual components ===
    y_fit_total = mixed_model(x, *popt)
    fitted_peaks = []
    peak_params = []

    for i, (model, _, _, _) in enumerate(peak_defs):
        amp, cen, wid = popt[3*i:3*i+3]
        if model == "gauss":
            y_peak = gaussian(x, amp, cen, wid)
            fwhm = 2.3548 * abs(wid)
            area = amp * wid * np.sqrt(2 * np.pi)
        elif model == "lorentz":
            y_peak = lorentzian(x, amp, cen, wid)
            fwhm = 2 * wid
            area = amp * np.pi * wid
        elif model == "pvoigt":
            y_peak = pseudo_voigt(x, amp, cen, wid, eta)
            fwhm = 0.5346 * 2 * wid + np.sqrt(0.2166 * (2 * wid)**2 + (2.3548 * wid)**2)
            area = amp * wid * np.sqrt(2 * np.pi)  # approximate
        else:
            raise ValueError(f"[!] Unknown model: {model}")

        fitted_peaks.append((x, y_peak))
        peak_params.append({
            "peak": i + 1,
            "model": model,
            "mu": cen,
            "FWHM": fwhm,
            "Area": area,
            "Relative_Intensity": amp
        })

    return y_fit_total, fitted_peaks, peak_params
