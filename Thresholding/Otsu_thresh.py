from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def otsu_thresholding(image):
    original_image = np.array(image)
    gray_image = image.convert("L")
    image_array = np.array(gray_image)

    n = image_array.size
    max_variance = -1
    threshold = -1

    # Otsu thresholding
    for t in range(0, 256):
        class0 = image_array[image_array < t]
        class1 = image_array[image_array >= t]

        if len(class0) == 0 or len(class1) == 0:
            continue

        w0 = len(class0) / n
        w1 = len(class1) / n

        mu0 = np.mean(class0)
        mu1 = np.mean(class1)

        var = w0*w1*(mu0-mu1)**2

        if (var > max_variance):
            max_variance = var
            threshold = t

    binary = np.where(image_array < threshold, 0, 255).astype(np.uint8)

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
    plt.axvline(threshold, color='orange',
                linestyle="--", label=f"T = {threshold}")
    plt.title("Original Histogram")
    plt.xlabel("Intensity")
    plt.ylabel("Number of Pixels")
    plt.legend()

    # Thresholded image
    plt.subplot(2, 2, 3)
    plt.imshow(binary, cmap="gray")
    plt.title(f"Otsu Thresholded Image (T={threshold})")
    plt.axis("off")
    plt.colorbar(label="Intensity")

    # Thresholded histogram
    plt.subplot(2, 2, 4)
    plt.hist(binary.flatten(), bins=256, range=(0, 255))
    plt.title("Otsu Thresholded Histogram")
    plt.xlabel("Intensity")
    plt.ylabel("Number of Pixels")

    plt.tight_layout()
    plt.show()

    return binary, threshold


image = Image.open(
    "../data/thresh/lilavati.tif")

binary, threshold = otsu_thresholding(image)

Image.fromarray(binary).save(
    "./Output Images/otsu_thresh_lilavati.png")
