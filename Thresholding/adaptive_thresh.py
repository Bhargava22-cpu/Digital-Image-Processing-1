from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def adaptive_thresholding(image, window_size, k, method, R=128):

    original_image = np.array(image)

    gray_image = image.convert("L")
    image_array = np.array(gray_image, dtype=float)

    padding = (window_size - 1) // 2
    padded = np.pad(image_array, padding, mode="edge")

    threshold_map = np.zeros_like(image_array, dtype=float)

    # Adaptive thresholding
    for x in range(image_array.shape[0]):
        for y in range(image_array.shape[1]):

            window = padded[
                x:x + window_size,
                y:y + window_size
            ]

            local_mean = np.mean(window)
            local_std = np.std(window)

            if method.lower() == "niblack":

                threshold = local_mean + k * local_std

            elif method.lower() == "sauvola":

                threshold = local_mean * (
                    1 + k * (local_std / R - 1)
                )

            else:
                raise ValueError(
                    "method must be 'niblack' or 'sauvola'"
                )

            threshold_map[x, y] = threshold

    binary = np.where(
        image_array < threshold_map,
        0,
        255
    ).astype(np.uint8)

    # Plot
    plt.figure(figsize=(15, 5))

    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(original_image)
    plt.title("Original Image")
    plt.axis("off")

    # Threshold map
    plt.subplot(1, 3, 2)
    plt.imshow(threshold_map, cmap="gray")
    plt.title(f"{method} Threshold Map")
    plt.axis("off")

    # Binary image
    plt.subplot(1, 3, 3)
    plt.imshow(binary, cmap="gray")
    plt.title(
        f"{method} Thresholded Image "
        f"(k={k}), Window Size={window_size}"
    )
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    return binary, threshold_map


image = Image.open(
    "../data/thresh/receipt.png"
)

binary, threshold_map = adaptive_thresholding(
    image,
    51,
    0.1,
    "sauvola"
)

Image.fromarray(binary).save(
    "./Output Images/adaptive_thresh_receipt.png"
)

"""
blackboard-> 75,0.05, sauvola
qr -> 51, 0.03, sauvalo
lilavati -> 81, 0.1, sauvalo
receipt -> 51, 0.1, sauvalo
"""