from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def myCLAHE(image, window_size, num_bins, threshold):

    # Convert RGB image to YCbCr
    ycbcr_image = image.convert("YCbCr")
    ycbcr_array = np.array(ycbcr_image)

    Y = ycbcr_array[:, :, 0].astype(np.uint8)

    height, width = Y.shape

    # luminance image
    Y_clahe = np.zeros_like(Y, dtype=np.uint8)

    padding = window_size // 2

    for x in range(height):
        for y in range(width):

            # Cropping window at image boundaries
            x1 = max(0, x - padding)
            x2 = min(height, x + padding + 1)

            y1 = max(0, y - padding)
            y2 = min(width, y + padding + 1)

            window = Y[x1:x2, y1:y2]

            # Local histogram
            histogram = np.bincount(
                window.flatten(),
                minlength=num_bins
            ).astype(float)

            probability = histogram / window.size

            # Clip histogram
            excess = np.sum(
                np.maximum(probability - threshold, 0)
            )

            probability = np.minimum(
                probability,
                threshold
            )

            # Redistribute excess probability
            probability += excess / num_bins

            # Local CDF
            cdf = np.cumsum(probability)

            # Map current pixel using local CDF
            pixel_value = Y[x, y]

            Y_clahe[x, y] = np.clip(
                255 * cdf[pixel_value],0,255
            )

    # Replace luminance channel
    ycbcr_clahe = ycbcr_array.copy()
    ycbcr_clahe[:, :, 0] = Y_clahe

    # Convert back to RGB
    enhanced_image = Image.fromarray(
        ycbcr_clahe,
        mode="YCbCr"
    ).convert("RGB")

    # Plot
    plt.figure(figsize=(12, 8))

    # Original image
    plt.subplot(2, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")

    # Original luminance histogram
    plt.subplot(2, 2, 2)
    plt.hist(
        Y.flatten(),
        bins=num_bins,
        range=(0, 255)
    )
    plt.title("Original Luminance Histogram")
    plt.xlabel("Luminance")
    plt.ylabel("Number of Pixels")

    # CLAHE image
    plt.subplot(2, 2, 3)
    plt.imshow(enhanced_image)
    plt.title(
        f"CLAHE Image "
        f"(window={window_size}, bins={num_bins}, "
        f"threshold={threshold})"
    )
    plt.axis("off")

    # Enhanced luminance histogram
    plt.subplot(2, 2, 4)
    plt.hist(
        Y_clahe.flatten(),
        bins=num_bins,
        range=(0, 255)
    )
    plt.title("CLAHE Luminance Histogram")
    plt.xlabel("Luminance")
    plt.ylabel("Number of Pixels")

    plt.tight_layout()
    plt.show()

    return enhanced_image


image = Image.open(
    "../data/hist/retina.png"
)

enhanced_image = myCLAHE(
    image,
    window_size=71,
    num_bins=256,
    threshold=0.025
)

enhanced_image.save(
    "./Output Images/clahe_retina_ht.png"
)

"""
canyon -> 71, 256, 0.03
significantly larger window @canyon-> 151, 256, 0.03
significantly smaller window @canyon-> 31, 256, 0.03
half threshold @canyon-> 71, 256, 0.015

retina -> 71, 256, 0.05
sl @retina -> 151, 256, 0.05
ss @retina -> 31, 256, 0.05
ht @retina -> 71, 256, 0.025
"""