import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys

def myBilinearInterpolation(img, scale=300):
    """
    Resizes an image using Bilinear interpolation.
    Outputs dimensions: R = scale*(M-1) + 1, C = scale*(N-1) + 1.
    """
    # Get original dimensions (assuming a 2D grayscale image based on template)
    M, N = img.shape
    
    # Calculate new dimensions
    R = scale * (M - 1) + 1
    C = scale * (N - 1) + 1
    
    # Create coordinate grids for the new image
    # We use indexing='ij' to keep r_mesh as rows (y) and c_mesh as columns (x)
    r_indices = np.arange(R) / scale
    c_indices = np.arange(C) / scale
    r_mesh, c_mesh = np.meshgrid(r_indices, c_indices, indexing='ij')
    
    # Find the bounding coordinates (x1, x2, y1, y2)
    # y maps to rows (r), x maps to columns (c)
    r1 = np.floor(r_mesh).astype(int)
    r2 = np.clip(r1 + 1, 0, M - 1)
    
    c1 = np.floor(c_mesh).astype(int)
    c2 = np.clip(c1 + 1, 0, N - 1)
    
    # Avoid division by zero at the extreme boundaries where x1 == x2 or y1 == y2
    dx = c2 - c1
    dx[dx == 0] = 1 
    dy = r2 - r1
    dy[dy == 0] = 1

    # Extract pixel values at the four corners for all grid points
    # f(Q11) = img[r1, c1], f(Q21) = img[r1, c2]
    # f(Q12) = img[r2, c1], f(Q22) = img[r2, c2]
    Q11 = img[r1, c1].astype(float)
    Q21 = img[r1, c2].astype(float)
    Q12 = img[r2, c1].astype(float)
    Q22 = img[r2, c2].astype(float)

    # 1. First, linearly interpolate along x axis
    # f(x, y1)
    fx_y1 = ((c2 - c_mesh) / dx) * Q11 + ((c_mesh - c1) / dx) * Q21
    # f(x, y2)
    fx_y2 = ((c2 - c_mesh) / dx) * Q12 + ((c_mesh - c1) / dx) * Q22

    # 2. Then, linearly interpolate along y axis
    # f(x, y)
    f_xy = ((r2 - r_mesh) / dy) * fx_y1 + ((r_mesh - r1) / dy) * fx_y2
    
    # Cast back to the original image data type (usually uint8)
    return f_xy.astype(np.float64)

# 1. Load the data
file_path = sys.argv[1] if len(sys.argv) > 1 else "../data/interp/random.png"
img = Image.open(file_path)
img_array = np.array(img, dtype=np.float64)

# 2. Perform Bilinear Interpolation
resized_image_array = myBilinearInterpolation(img_array, scale=300)

img_bili = Image.fromarray(resized_image_array.astype(np.uint8))
img_bili.save("./img/q1c_bili.png")

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
axes[1].set_title(f"Resized Image (Bilinear)\nDimensions: {resized_image_array.shape}")
axes[1].set_xlabel("Pixel Column Units")
axes[1].set_ylabel("Pixel Row Units")
fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig("./img/q1c.png")