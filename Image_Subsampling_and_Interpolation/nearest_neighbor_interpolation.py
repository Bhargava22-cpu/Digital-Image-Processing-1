import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys

def myNearestNeighborInterpolation(img, scale=300):
    """
    Resizes an image using Nearest-Neighbor interpolation.
    Outputs dimensions: R = scale*(M-1) + 1, C = scale*(N-1) + 1.
    """
    # Get original dimensions
    M, N = img.shape
    
    # Calculate new dimensions
    R = scale * (M - 1) + 1
    C = scale * (N - 1) + 1
    
    # Create coordinate grids for the new image
    r_indices = np.arange(R)
    c_indices = np.arange(C)
    
    # Map new indices back to original indices using standard rounding
    # np.floor(x + 0.5) is used for symmetric half-up rounding
    m_indices = np.floor(r_indices / scale + 0.5).astype(int)
    n_indices = np.floor(c_indices / scale + 0.5).astype(int)
    
    # Use NumPy's advanced indexing to pull the mapped pixels efficiently
    resized_img = img[np.ix_(m_indices, n_indices)]
    
    return resized_img

# 1. Load the data
file_path = sys.argv[1] if len(sys.argv) > 1 else "../data/interp/random.png"
img = Image.open(file_path)

# STRICT REQIREMENT for gray-scale images
img_array = np.array(img, dtype=np.float64)

# 2. Perform Nearest-Neighbor Interpolation
resized_image_array = myNearestNeighborInterpolation(img_array, scale=300)

img_nni = Image.fromarray(resized_image_array.astype(np.uint8))
img_nni.save("./img/q1b_nni.png")

# 3. Plotting
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Display Original Image
im1 = axes[0].imshow(img_array, cmap='jet', aspect='equal')
axes[0].set_title(f"Original Image\nDimensions: {img_array.shape}")
axes[0].set_xlabel("Pixel Column Units")
axes[0].set_ylabel("Pixel Row Units")
fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

# Display Resized Image
im2 = axes[1].imshow(resized_image_array, cmap='jet', aspect='equal')
axes[1].set_title(f"Resized Image\nDimensions: {resized_image_array.shape}")
axes[1].set_xlabel("Pixel Column Units")
axes[1].set_ylabel("Pixel Row Units")
fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig("./img/q1b.png")