# === Raman Spectrum Analysis Pipeline (Region-Based Fitting) ===
import os
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np

from preprocessing import preprocess
from curve_fitting import fit_peaks_regionwise
from analysis_plotting import plot_and_report, apply_pub_style, PUB_FIGSIZE, PUB_DPI

# === Size & Legend COnfig ===
FIG_WIDTH = 6      # inches
FIG_HEIGHT = 4.5   # inches
LEGEND_OUTSIDE = True

# === Region, Cropping & Baseline Definitions ===
cmin = 1000
cmax = 1800
baseorder = 1

# Format: (start, end, [ (model, amp, center, width), ... ])
REGIONS = [(1000, 2100, [("gauss", 0.3, 1080, 5), ("lorentz", 2, 1415, 50)])]

# === File Input Handling ===
def choose_file_dialog():
    root = tk.Tk()
    root.withdraw()
    return filedialog.askopenfilenames(
        title="Select Raman CSV File(s)",
        filetypes=[("CSV files", "*.csv")]
    )

# === Overlaying Multiple Spectra ===
def overlay_multiple_spectra(
    file_paths,
    crop_min=cmin, crop_max=cmax,
    scale_unirradiated=True,
    figsize=None, legend_outside=True
):
    fig, ax = plt.subplots(figsize=figsize or PUB_FIGSIZE, dpi=PUB_DPI)
    offset_step = 1.0  # always integer offset
    legend_handles, legend_labels = [], []

    # Step 1: Calculate ratios for each spectrum
    spectra_data = []
    ratios = []

    for file in file_paths:
        folder = os.path.basename(os.path.dirname(file))
        name = os.path.splitext(os.path.basename(file))[0]
        label = f"{folder} {name}"

        # Load full spectrum without cropping
        x_full, y_full = preprocess(
            input_path=file,
            crop_min=0,
            crop_max=99999,
            sg_window=11,
            sg_polyorder=10,
            imodpoly_order=baseorder,
            imodpoly_tol=1e-3,
            imodpoly_max_iter=100,
            normalisation="none",   # <-- No scaling here
            plot=False,
            save_path=None,
            alex_data=False
        )


        full_max = np.max(y_full)

        # Load cropped spectrum
        x_crop, y_crop = preprocess(
            input_path=file,
            crop_min=crop_min,
            crop_max=crop_max,
            sg_window=11,
            sg_polyorder=10,
            imodpoly_order=baseorder,
            imodpoly_tol=1e-3,
            imodpoly_max_iter=100,
            normalisation="none",
            plot=False,
            save_path=None,
            alex_data=False
        )

        cropped_max = np.max(y_crop)
        ratio = cropped_max / full_max
        ratios.append(ratio)

        spectra_data.append((label, x_crop, y_crop))

    # Step 2: Find the max ratio to normalise against
    max_ratio = max(ratios)

    # Step 3: Plot with scaled heights and integer offsets
    for i, (label, x_crop, y_crop) in enumerate(spectra_data):
        scale_factor = ratios[i] / max_ratio
        y_scaled = y_crop * scale_factor
        y_offset = y_scaled + i * offset_step

        h, = ax.plot(x_crop, y_offset, linewidth=1.2, label=label)
        legend_handles.append(h)
        legend_labels.append(label)

    # Step 4: Apply consistent style
    _ = apply_pub_style(
        ax,
        title="Overlay of Preprocessed Raman Spectra (Relative Cropped Height)",
        xlabel="Raman shift (cm$^{-1}$)",
        ylabel="Offset Intensity (a.u.)"
    )

    # Step 5: Legend outside by default
    ncol = 1 if len(legend_handles) <= 14 else 2
    if legend_outside:
        ax.legend(
            legend_handles, legend_labels,
            loc='upper left',
            bbox_to_anchor=(1.02, 1),
            borderaxespad=0,
            fontsize=6,
            frameon=False,
            ncol=ncol
        )
    else:
        ax.legend(legend_handles, legend_labels, loc='upper right', fontsize=6, frameon=False, ncol=ncol)

    plt.tight_layout()
    plt.show()



# === Main Execution ===
def main():
    input_files = choose_file_dialog()

    if not input_files:
        print("[!] No file(s) selected.")
        return

    figsize = (FIG_WIDTH, FIG_HEIGHT)

    if isinstance(input_files, (list, tuple)) and len(input_files) > 1:
        overlay_multiple_spectra(input_files, figsize=figsize, legend_outside=LEGEND_OUTSIDE)
        return

    input_file = input_files[0]
    if not os.path.isfile(input_file):
        print("[!] Invalid file selected.")
        return

    filename = os.path.splitext(os.path.basename(input_file))[0]
    print(f"[✓] Selected file: {filename}.csv")

    os.makedirs("output", exist_ok=True)

    x, y = preprocess(
        input_file,
        crop_min=cmin,
        crop_max=cmax,
        sg_window=11,
        sg_polyorder=10,
        imodpoly_order=baseorder,
        imodpoly_tol=1e-3,
        imodpoly_max_iter=100,
        normalisation="vector-0to1",
        plot=True,
        save_path=f"output/{filename}_processed.csv",
        alex_data=False
    )

    CENTER_SHIFT_LIMIT = 30
    y_fit_total, fitted_peaks, peak_params = fit_peaks_regionwise(x, y, REGIONS, center_tolerance=CENTER_SHIFT_LIMIT)

    plot_and_report(
        x, y,
        y_fit_total, fitted_peaks, peak_params,
        annotate=False,
        stagger_labels=True,
        font_size=9,
        label_offset=0.05,
        show_components=True,
        show_text_plot=True,
        save_curve_path=f"output/{filename}_fitted_curve.csv",
        save_params_path=f"output/{filename}_peak_parameters.csv",
        show=True,
        figsize=figsize,
        legend_outside=LEGEND_OUTSIDE,
        legend_ncol=1,
        legend_fontsize=6
    )

if __name__ == "__main__":
    main()
