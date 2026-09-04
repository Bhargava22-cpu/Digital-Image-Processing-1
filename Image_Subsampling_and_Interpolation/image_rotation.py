import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys

def myImageRotationUsingNearestNeighborInterp(img, angle):
    """
    Rotates an image by a given angle (in degrees) using Nearest-Neighbor interpolation.
    Keeps the output dimensions the same as the input image.
    """
    # Get original dimensions (assuming grayscale 2D image)
    M, N = img.shape[:2]
    
    # Output dimensions stay the same
    R, C = M, N
    
    # Calculate the center of the image to rotate around it
    cy, cx = M / 2.0, N / 2.0
    
    # Convert angle from degrees to radians
    theta = np.radians(angle)
    # theta = angle
    
    # Create coordinate grids for the NEW (rotated) image
    c_indices = np.arange(C)
    r_indices = np.arange(R)
    c_grid, r_grid = np.meshgrid(c_indices, r_indices) # c_grid is X, r_grid is Y
    
    # Shift coordinates so the origin is at the center of the image
    c_shifted = c_grid - cx
    r_shifted = r_grid - cy
    
    # Backward mapping: Rotate the coordinates back by -theta to find original pixels
    # Standard 2D rotation matrix for (X, Y)
    c_orig = c_shifted * np.cos(theta) + r_shifted * np.sin(theta)
    r_orig = -c_shifted * np.sin(theta) + r_shifted * np.cos(theta)
    
    # Shift coordinates back from the center to the top-left origin
    c_orig = c_orig + cx
    r_orig = r_orig + cy
    
    # Nearest-Neighbor: Map back to integer indices using standard rounding
    m_indices = np.floor(r_orig + 0.5).astype(int)
    n_indices = np.floor(c_orig + 0.5).astype(int)
    
    # Create a mask to filter out indices that fall outside the original image boundaries
    valid_mask = (m_indices >= 0) & (m_indices < M) & (n_indices >= 0) & (n_indices < N)
    
    # Initialize the rotated image with zeros (black background)
    rotated_img = np.zeros_like(img)
    
    # Use the mask to map only valid pixels from the original image to the rotated image
    rotated_img[valid_mask] = img[m_indices[valid_mask], n_indices[valid_mask]]
    
    return rotated_img


def myImageRotationUsingBilinearInterp(img, angle):
    """
    Rotates an image by a given angle (in degrees) using Bilinear interpolation.
    Keeps the output dimensions the same as the input image.
    """
    # Get original dimensions (assuming grayscale 2D image based on template)
    M, N = img.shape[:2]
    
    # Output dimensions stay the same
    R, C = M, N
    
    # Calculate the center of the image to rotate around it
    cy, cx = M / 2.0, N / 2.0
    
    # Convert angle from degrees to radians
    theta = np.radians(angle)
    
    # Create coordinate grids for the NEW (rotated) image
    c_indices = np.arange(C)
    r_indices = np.arange(R)
    c_grid, r_grid = np.meshgrid(c_indices, r_indices) # c_grid is X, r_grid is Y
    
    # Shift coordinates so the origin is at the center of the image
    c_shifted = c_grid - cx
    r_shifted = r_grid - cy
    
    # Backward mapping: Rotate coordinates back by -theta
    c_orig = c_shifted * np.cos(theta) + r_shifted * np.sin(theta)
    r_orig = -c_shifted * np.sin(theta) + r_shifted * np.cos(theta)
    
    # Shift coordinates back from the center to the top-left origin
    c_orig = c_orig + cx
    r_orig = r_orig + cy
    
    # Find the bounding coordinates (r1, r2, c1, c2)
    # y maps to rows (r), x maps to columns (c)
    r1 = np.floor(r_orig).astype(int)
    r2 = r1 + 1
    c1 = np.floor(c_orig).astype(int)
    c2 = c1 + 1
    
    # Clip coordinates to stay within image bounds for safe array indexing
    r1_safe = np.clip(r1, 0, M - 1)
    r2_safe = np.clip(r2, 0, M - 1)
    c1_safe = np.clip(c1, 0, N - 1)
    c2_safe = np.clip(c2, 0, N - 1)
    
    # Avoid division by zero at extreme boundaries
    dx = c2_safe - c1_safe
    dx[dx == 0] = 1 
    dy = r2_safe - r1_safe
    dy[dy == 0] = 1
    
    # Extract pixel values at the four corners for all grid points
    Q11 = img[r1_safe, c1_safe].astype(float)
    Q21 = img[r1_safe, c2_safe].astype(float)
    Q12 = img[r2_safe, c1_safe].astype(float)
    Q22 = img[r2_safe, c2_safe].astype(float)
    
    # 1. First, linearly interpolate along x axis (columns)
    # Matching f(x, y1) and f(x, y2) formulas from slides
    fx_y1 = ((c2_safe - c_orig) / dx)[:, :, np.newaxis] * Q11 + ((c_orig - c1_safe) / dx)[:, :, np.newaxis] * Q21
    fx_y2 = ((c2_safe - c_orig) / dx)[:, :, np.newaxis] * Q12 + ((c_orig - c1_safe) / dx)[:, :, np.newaxis] * Q22
    
    # 2. Then, linearly interpolate along y axis (rows)
    # Matching f(x, y) formula from slides
    f_xy = ((r2_safe - r_orig) / dy)[:, :, np.newaxis] * fx_y1 + ((r_orig - r1_safe) / dy)[:, :, np.newaxis] * fx_y2
    
    # Create a mask to filter out indices that fall outside the original image boundaries
    valid_mask = (r_orig >= 0) & (r_orig <= M - 1) & (c_orig >= 0) & (c_orig <= N - 1)
    
    # Initialize the rotated image with zeros (black background)
    rotated_img = np.zeros_like(img)
    
    # Use the mask to map only valid pixels from the interpolated grid to the rotated image
    rotated_img[valid_mask] = f_xy[valid_mask].astype(img.dtype)
    
    return rotated_img

# 1. Load the data
file_path = sys.argv[1] if len(sys.argv) > 1 else "../data/interp/main.png"
img = Image.open(file_path)
img_array = np.array(img)

# find the angle (from any image website)
angle = np.degrees(np.arctan(14/160))

# 2. Perform Bicubic Interpolation
nn_rotated_image_array = myImageRotationUsingNearestNeighborInterp(img_array, angle)
bili_rotated_image_array = myImageRotationUsingBilinearInterp(img_array, angle)

img_nni_rotated = Image.fromarray(nn_rotated_image_array)
img_nni_rotated.save("./img/q1e_nni_rot.png")
img_bili_rotated = Image.fromarray(bili_rotated_image_array)
img_bili_rotated.save("./img/q1e_bili_rot.png")

# 4. Plot the results side-by-side
plt.figure(figsize=(15, 6))

plt.subplot(1, 3, 1)
plt.imshow(img_array, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(nn_rotated_image_array, cmap='gray')
plt.title(f'Nearest Neighbor Rotation ({angle}°)')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(bili_rotated_image_array, cmap='gray')
plt.title(f'Bilinear Rotation ({angle}°)')
plt.axis('off')

plt.tight_layout()
plt.savefig("./img/q1e.png")