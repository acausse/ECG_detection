# ECG Processing and Feature Extraction Utilities

This repository contains a collection of Python functions for **electrocardiogram (ECG) signal processing**, **R-peak detection**, and **heart-rate–variability (HRV) feature extraction**.  
It also includes utility functions for **signal filtering**, **phase estimation** designed for exploratory research and offline analysis.

The core R-peak detection logic relies on **multi-algorithm agreement** using established ECG detectors.

---

## 📖 References

R-peak detection methods are provided by the **py-ecg-detectors** package:

- Silva, I., Moody, G., Behar, J., Johnson, A., & Clifford, G. D.  
  *An open-source toolbox for analysing and processing physiological signals in Python.*  
  https://pypi.org/project/py-ecg-detectors/

Algorithms implemented in that package include:
- Pan–Tompkins
- Hamilton
- Christov
- Stationary Wavelet Transform (SWT)
- Two-Average
- Engzee

---

## 📌 Features

### ECG Preprocessing
- **`filt_butter`**  
  Applies Butterworth filtering (high-pass, low-pass, or band-pass) with support for **causal** and **zero-phase** filtering.

---

### R-Peak Detection
- **`detect_ecg_R_peaks`**  
  Detects ECG R-peaks using multiple detectors and returns:
  - Raw detections per algorithm
  - Aligned peaks (local extrema refinement)
  - Thresholded detections
  - Final R-peaks based on **detector agreement**

  The final R-peak detection can be accessed via:
  ```python
  r_peaks['agree']
  ```
  
  This approach improves robustness to noise and morphology changes by requiring agreement across detectors.

### ECG Phase Estimation
- **`get_ecg_phases`**
   Computes a continuous ECG phase signal spanning 0 → 2π between consecutive R-peaks.


---

## 📦 Installation

Install the required dependencies using pip:

```bash
pip install numpy scipy py-ecg-detectors
```
  
