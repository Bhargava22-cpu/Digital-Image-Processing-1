import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

#1. Load the image
file_path = sys.argv[1] if len(sys.argv) > 1 else "../data/interp/suit.png"
img = Image.open(file_path)
img_array = np.array(img)

#2. Shrink function
def myImageShrink(img_arr, d):
    sub_array = img_arr[::d, ::d]
    return sub_array

#3. Generate subsampled images
sub2_array = myImageShrink(img_array, 2)
sub3_array = myImageShrink(img_array, 3)

#4. Save sumsampled images to disk
img_sub2 = Image.fromarray(sub2_array)
img_sub3 = Image.fromarray(sub3_array)
img_sub2.save("./img/q1a_sub2.png")
img_sub3.save("./img/q1a_sub3.png")

#5. Plotting logic
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(img_array, aspect='equal')
axes[0].set_title(f"Original Image\nShape: {img_array.shape}")

# Subsampled d=2
axes[1].imshow(sub2_array, aspect='equal')
axes[1].set_title(f"Subsampled (d=2)\nShape: {sub2_array.shape}")

# Subsampled d=3
axes[2].imshow(sub3_array, aspect='equal')
axes[2].set_title(f"Subsampled (d=3)\nShape: {sub3_array.shape}")

# print("Shapes:", img_array.shape, sub2_array.shape, sub3_array.shape)

plt.tight_layout()
plt.savefig("./img/q1a.png")