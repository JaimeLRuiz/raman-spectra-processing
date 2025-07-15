#Multi Spectra Comparision

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import ramanspy as rp
from ramanspy import preprocessing

# === List all your input files here ===
file_paths = [
    "c:/Users/danie/OneDrive/Oxford 2025/03 Data/03 Processed Data/Refel_Ne_300_2.5/Dark Grey Focused.csv",
    "c:/Users/danie/OneDrive/Oxford 2025/03 Data/03 Processed Data/Refel_Si_300_2.5/Dark Grey Focused.csv",
    #"c:/Users/danie/OneDrive/Oxford 2025/03 Data/03 Processed Data/19C_Si-Cr_Si_300_0.25/Dark Grey.csv",
    #"c:/Users/danie/OneDrive/Oxford 2025/03 Data/03 Processed Data/19C_Si-Cr_Si_750_2.5/Dark Grey.csv",
    
    # Add more files below as needed
    # "path/to/another/spectrum.csv",
]

# === Define preprocessing pipeline ===
pipeline = preprocessing.Pipeline([
    preprocessing.denoise.SavGol(window_length=11, polyorder=3),
    preprocessing.baseline.IModPoly(poly_order=5),
    preprocessing.normalise.Vector()
])

# === Load + preprocess each file ===
def load_and_preprocess(filepath):
    data = pd.read_csv(filepath, encoding="latin1", sep=",")
    x = data["#Wave"].values
    y = data["#Intensity"].values
    spectrum = rp.Spectrum(y, x)
    return pipeline.apply(spectrum)

# === Plotting ===
plt.figure(figsize=(12, 6))
offset_step = 1.2  # spacing between each curve

for i, file in enumerate(file_paths):
    folder = os.path.basename(os.path.dirname(file))
    name = os.path.splitext(os.path.basename(file))[0]
    label = f"{folder} {name}"

    processed = load_and_preprocess(file)
    x, y = processed.spectral_axis, processed.spectral_data

    # Apply vertical offset to each curve
    # Scale offset to preserve spectral shape
    y_offset = y + i * (np.max(y) - np.min(y)) * offset_step


    plt.plot(x, y_offset, label=label, linewidth=1.5)

plt.xlabel("Raman Shift (cm⁻¹)")
plt.ylabel("Offset Intensity (a.u.)")
plt.title("Vertically Offset Overlay of Preprocessed Raman Spectra")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
