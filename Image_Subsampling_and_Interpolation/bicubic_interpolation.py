import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys

def myBicubicInterpolation(img, scale=300):
    """
    Resizes an image using Bicubic interpolation.
    Outputs dimensions: R = scale*(M-1) + 1, C = scale*(N-1) + 1.
    Strictly follows the 16-parameter formulation and finite differences 
    from the lecture slides.
    """
    M, N = img.shape
    
    R = scale * (M - 1) + 1
    C = scale * (N - 1) + 1
    
    # Cast to float for derivative calculations
    img_float = img.astype(float)
    
    # Pad the image by 1 pixel on all edges to handle boundary finite differences
    # Replicating the edge pixel ensures a derivative of 0 at the extreme boundaries.
    img_pad = np.pad(img_float, 1, mode='edge')
    
    # Pre-compute function values and derivatives for the entire image
    # Note mapping: x maps to column index, y maps to row index
    f_val = img_pad[1:-1, 1:-1]
    
    # f_x = 0.5 * ( f(x+1, y) - f(x-1, y) ) -> differences along columns
    f_x = 0.5 * (img_pad[1:-1, 2:] - img_pad[1:-1, :-2])
    
    # f_y = 0.5 * ( f(x, y+1) - f(x, y-1) ) -> differences along rows
    f_y = 0.5 * (img_pad[2:, 1:-1] - img_pad[:-2, 1:-1])
    
    # f_xy = 0.25 * ( [f(x+1,y+1) + f(x-1,y-1)] - [f(x-1,y+1) + f(x+1,y-1)] )
    f_xy = 0.25 * ((img_pad[2:, 2:] + img_pad[:-2, :-2]) - 
                   (img_pad[2:, :-2] + img_pad[:-2, 2:]))
    
    # Create coordinate grids for the new image
    r_indices = np.arange(R) / scale
    c_indices = np.arange(C) / scale
    r_mesh, c_mesh = np.meshgrid(r_indices, c_indices, indexing='ij')
    
    # Determine the top-left corner (r0, c0) of the unit square for each point
    r0 = np.clip(np.floor(r_mesh).astype(int), 0, M - 2)
    c0 = np.clip(np.floor(c_mesh).astype(int), 0, N - 2)
    
    # Calculate local fractional coordinates (dx, dy) within the [0, 1] unit square
    dx = c_mesh - c0
    dy = r_mesh - r0
    
    # Create the F matrix (4x4) for each pixel grid coordinate simultaneously
    # Dimensions of F_grid: (R, C, 4, 4)
    F_grid = np.zeros((R, C, 4, 4))
    
    # Top-left block: Function values f(x,y)
    F_grid[..., 0, 0] = f_val[r0, c0]         # f(0,0)
    F_grid[..., 0, 1] = f_val[r0+1, c0]       # f(0,1)  (y=1 -> r0+1)
    F_grid[..., 1, 0] = f_val[r0, c0+1]       # f(1,0)  (x=1 -> c0+1)
    F_grid[..., 1, 1] = f_val[r0+1, c0+1]     # f(1,1)
    
    # Top-right block: Partial derivative f_y
    F_grid[..., 0, 2] = f_y[r0, c0]           # f_y(0,0)
    F_grid[..., 0, 3] = f_y[r0+1, c0]         # f_y(0,1)
    F_grid[..., 1, 2] = f_y[r0, c0+1]         # f_y(1,0)
    F_grid[..., 1, 3] = f_y[r0+1, c0+1]       # f_y(1,1)
    
    # Bottom-left block: Partial derivative f_x
    F_grid[..., 2, 0] = f_x[r0, c0]           # f_x(0,0)
    F_grid[..., 2, 1] = f_x[r0+1, c0]         # f_x(0,1)
    F_grid[..., 3, 0] = f_x[r0, c0+1]         # f_x(1,0)
    F_grid[..., 3, 1] = f_x[r0+1, c0+1]       # f_x(1,1)
    
    # Bottom-right block: Cross derivative f_xy
    F_grid[..., 2, 2] = f_xy[r0, c0]          # f_xy(0,0)
    F_grid[..., 2, 3] = f_xy[r0+1, c0]        # f_xy(0,1)
    F_grid[..., 3, 2] = f_xy[r0, c0+1]        # f_xy(1,0)
    F_grid[..., 3, 3] = f_xy[r0+1, c0+1]      # f_xy(1,1)
    
    # The M matrix to map values/derivatives to polynomial coefficients
    M_mat = np.array([
        [ 1,  0,  0,  0],
        [ 0,  0,  1,  0],
        [-3,  3, -2, -1],
        [ 2, -2,  1,  1]
    ], dtype=float)
    
    # Calculate coefficient matrix A = M * F * M^T for all points efficiently
    # Using np.einsum solves this over the whole (R,C) grid without nested Python loops
    A_grid = np.einsum('ia, rcab, jb -> rcij', M_mat, F_grid, M_mat)
    
    # Polynomial variables X = [1, x, x^2, x^3] and Y = [1, y, y^2, y^3]
    X_vec = np.stack([np.ones_like(dx), dx, dx**2, dx**3], axis=-1)
    Y_vec = np.stack([np.ones_like(dy), dy, dy**2, dy**3], axis=-1)
    
    # Compute the final interpolated value: p(x,y) = X * A * Y^T
    interpolated = np.einsum('rci, rcij, rcj -> rc', X_vec, A_grid, Y_vec)
    
    # Clip values to max and min of img to ensure they remain inside valid
    interpolated = np.clip(interpolated, img.min(), img.max())
    
    return interpolated.astype(np.float64)

# 1. Load the data
file_path = sys.argv[1] if len(sys.argv) > 1 else "../data/interp/random.png"
img = Image.open(file_path)
img_array = np.array(img, dtype=np.float64)

# 2. Perform Bicubic Interpolation
resized_image_array = myBicubicInterpolation(img_array, scale=300)

img_bicu = Image.fromarray(resized_image_array.astype(np.uint8))
img_bicu.save("./img/q1d_bicu.png")

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
axes[1].set_title(f"Resized Image (Bicubic)\nDimensions: {resized_image_array.shape}")
axes[1].set_xlabel("Pixel Column Units")
axes[1].set_ylabel("Pixel Row Units")
fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig("./img/q1d.png")