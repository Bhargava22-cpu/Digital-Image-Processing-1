import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys
import scipy.io

# ---------------------------------------------------------
# [Insert your 3 functions here: myNearestNeighborInterpolation, 
#  myBilinearInterpolation, myBicubicInterpolation]
# ---------------------------------------------------------

from q1b import myNearestNeighborInterpolation
from q1c import myBilinearInterpolation
from q1d import myBicubicInterpolation

def calculate_rmse(img1, img2):
    """Calculates the Root Mean Square Error between two images."""
    return np.sqrt(np.mean((img1.astype(float) - img2.astype(float)) ** 2))

# 1. Load the data
file_path = sys.argv[1] if len(sys.argv) > 1 else "../data/interp/ct.mat"
mat_data = scipy.io.loadmat(file_path)

# Extract images (assuming standard keys, adjust if your .mat keys differ)
img_original = mat_data['original']
img_subsampled = mat_data['subsampled']

# 2. Determine the scale factor dynamically
# Since R = scale * (M - 1) + 1, we can find scale as:
# scale = (R - 1) / (M - 1)
scale = (img_original.shape[0] - 1) // (img_subsampled.shape[0] - 1)

# 3. Perform Interpolations
img_nni = myNearestNeighborInterpolation(img_subsampled, scale=scale)
img_bilinear = myBilinearInterpolation(img_subsampled, scale=scale)
img_bicubic = myBicubicInterpolation(img_subsampled, scale=scale)

# 4. Compute and Report RMSE
rmse_nni = calculate_rmse(img_original, img_nni)
rmse_bilinear = calculate_rmse(img_original, img_bilinear)
rmse_bicubic = calculate_rmse(img_original, img_bicubic)

print("--- Root Mean Square Errors (RMSE) ---")
print(f"Nearest Neighbor: {rmse_nni:.4f}")
print(f"Bilinear:         {rmse_bilinear:.4f}")
print(f"Bicubic:          {rmse_bicubic:.4f}")

# 5. Compute Difference Images
diff_nni = img_original.astype(float) - img_nni.astype(float)
diff_bilinear = img_original.astype(float) - img_bilinear.astype(float)
diff_bicubic = img_original.astype(float) - img_bicubic.astype(float)

# ==========================================
# PLOTTING 1: Original and Enlarged Images
# ==========================================
# Use the same color limits based on the original image
vmin_img, vmax_img = img_original.min(), img_original.max()

fig1, axes1 = plt.subplots(1, 4, figsize=(20, 5))
images = [img_original, img_nni, img_bilinear, img_bicubic]
titles = ['Original Image', 'Nearest Neighbor', 'Bilinear', 'Bicubic']

for ax, img, title in zip(axes1, images, titles):
    im = ax.imshow(img, cmap='jet', vmin=vmin_img, vmax=vmax_img, aspect='equal')
    ax.set_title(f"{title}\nDimensions: {img.shape}")
    ax.set_xlabel("Pixel Column Units")
    ax.set_ylabel("Pixel Row Units")
    fig1.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig("./img/q1f_ct_enlarged_comparison.png")
# plt.show()

# ==========================================
# PLOTTING 2: Difference Images
# ==========================================
# Find global min and max across ALL difference images to ensure uniform limits
vmin_diff = min(diff_nni.min(), diff_bilinear.min(), diff_bicubic.min())
vmax_diff = max(diff_nni.max(), diff_bilinear.max(), diff_bicubic.max())

fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
diff_images = [diff_nni, diff_bilinear, diff_bicubic]
diff_titles = ['Diff: Nearest Neighbor', 'Diff: Bilinear', 'Diff: Bicubic']

for ax, diff, title in zip(axes2, diff_images, diff_titles):
    im = ax.imshow(diff, cmap='jet', vmin=vmin_diff, vmax=vmax_diff, aspect='equal')
    ax.set_title(title)
    ax.set_xlabel("Pixel Column Units")
    ax.set_ylabel("Pixel Row Units")
    fig2.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig("./img/q1f_ct_difference_images.png")
# plt.show()