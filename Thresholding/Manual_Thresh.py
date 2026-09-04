from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def manual_thresholding(image, T):
    original_image = np.array(image)
    gray_image = image.convert("L")
    image_array = np.array(gray_image)

    # Manual thresholding
    binary = np.where(image_array < T, 0, 255).astype(np.uint8)

    # Plot
    plt.figure(figsize=(12, 8))

    # Original image
    plt.subplot(2, 2, 1)
    plt.imshow(original_image)
    plt.title("Original Image")
    plt.axis("off")

    # Original histogram
    plt.subplot(2, 2, 2)
    plt.hist(image_array.flatten(), bins=256, range=(0, 255))
    plt.axvline(T, color='orange', linestyle="--", label=f"T = {T}")
    plt.title("Original Histogram")
    plt.xlabel("Intensity")
    plt.ylabel("Number of Pixels")
    plt.legend()

    # Thresholded image
    plt.subplot(2, 2, 3)
    plt.imshow(binary, cmap="gray")
    plt.title(f"Manual Thresholded Image (T={T})")
    plt.axis("off")
    plt.colorbar(label="Intensity")

    # Thresholded histogram
    plt.subplot(2, 2, 4)
    plt.hist(binary.flatten(), bins=256, range=(0, 255))
    plt.title("Manual Thresholded Histogram")
    plt.xlabel("Intensity")
    plt.ylabel("Number of Pixels")

    plt.tight_layout()
    plt.show()

    return binary


image = Image.open(
    "../data/thresh/qr.png")

binary = manual_thresholding(image, 110)

Image.fromarray(binary).save(
    "./Output Images/manual_thresh_qr.png")

"""
Thresholds
blackboard -> 65
lilavati -> 115
receipt -> 120
qr ->110

"""