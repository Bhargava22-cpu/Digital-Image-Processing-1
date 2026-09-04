# Digital Image Processing

Implementation of fundamental digital image processing techniques covering image subsampling and interpolation, thresholding, and contrast enhancement.

## Contents

### 1. Image Subsampling and Interpolation

- **Image Shrinking** — Reduces image size by selecting every `d`-th pixel along the rows and columns.
- **Nearest-Neighbor Interpolation** — Enlarges an image by assigning each output pixel the value of its nearest input pixel.
- **Bilinear Interpolation** — Estimates pixel values using a weighted average of the four neighboring pixels.
- **Bicubic Interpolation** — Estimates pixel values using cubic interpolation over a larger neighborhood to produce smoother results.
- **Image Rotation** — Rotates an image using nearest-neighbor and bilinear interpolation while maintaining the original image size.
- **Downsampling and Upsampling** — Enlarges a subsampled CT image using nearest-neighbor, bilinear, and bicubic interpolation, and compares their reconstruction quality using difference images and RMSE.

### 2. Thresholding

- **Manual Thresholding** — Converts grayscale images into binary images using manually selected intensity thresholds.
- **Otsu Thresholding** — Automatically selects a global threshold by maximizing the separation between foreground and background intensity classes.
- **Adaptive Thresholding** — Computes a local threshold for each pixel using the Sauvola method, adapting to variations in illumination and local contrast.

### 3. Contrast Enhancement

- **Linear Contrast Stretching** — Expands the intensity range of an image using a piecewise linear transformation to improve contrast.
- **Histogram Equalization** — Redistributes image intensities using the cumulative histogram to enhance overall contrast.
- **CLAHE** — Performs local histogram equalization with histogram clipping to enhance local details while limiting excessive noise amplification.
- **Histogram Matching** — Transforms the intensity distribution of an image to match the histogram of a reference image.

## Repository Structure

```text
Digital-Image-Processing-1/
│
├── Image_Subsampling_and_Interpolation/
│   ├── image_shrinking.py
│   ├── nearest_neighbor_interpolation.py
│   ├── bilinear_interpolation.py
│   ├── bicubic_interpolation.py
│   ├── image_rotation.py
│   └── downsampling_upsampling.py
│
├── Thresholding/
│   ├── manual_thresholding.py
│   ├── otsu_thresholding.py
│   └── adaptive_thresholding.py
│
├── Contrast_Enhancement/
│   ├── linear_contrast_stretching.py
│   ├── histogram_equalization.py
│   ├── clahe.py
│   └── histogram_matching.py
│
├── data/
│   ├── interp/
│   ├── thresh/
│   └── hist/
│
├── report.pdf
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.x
- NumPy
- Matplotlib
- Pillow
- SciPy

Install the required libraries using:

```bash
pip install -r requirements.txt
```

Alternatively:

```bash
pip install numpy matplotlib pillow scipy
```

## Running the Code

Each algorithm is implemented as a separate Python script.

Navigate to the corresponding folder and run the desired script.

For example:

```bash
cd Thresholding
python adaptive_thresholding.py
```

The input images and data used by the implementations are provided in the `data/` directory.

## Report

The complete report contains the methodology, parameter selection, results, comparisons, RMSE values, and observations for the implemented techniques.

The report is available as:

`report.pdf`
