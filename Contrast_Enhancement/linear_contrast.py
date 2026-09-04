from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def myLinearContrastStretch(image, r1, s1):
    # Converting RGB image to YCbCr
    ycbcr_image = image.convert("YCbCr")
    ycbcr_array = np.array(ycbcr_image)

    Y = ycbcr_array[:, :, 0].astype(float)

    Y_min = np.min(Y)
    Y_max = np.max(Y)

    # Piecewise-linear contrast stretching
    Y_stretched = np.zeros_like(Y)

    # Y_min → r1
    mask1 = Y <= r1

    Y_stretched[mask1] = (
        (s1 - 0) / (r1 - Y_min)
    ) * (Y[mask1] - Y_min)

    # r1 → Y_max
    mask2 = Y > r1

    Y_stretched[mask2] = (
        (255 - s1) / (Y_max - r1)
    ) * (Y[mask2] - r1) + s1

    # Replacing the luminance component
    ycbcr_stretched = ycbcr_array.astype(float)
    ycbcr_stretched[:, :, 0] = Y_stretched

    # Convert back to RGB
    enhanced_image = Image.fromarray(
        ycbcr_stretched.astype(np.uint8),
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
        bins=256,
        range=(0, 255)
    )
    plt.title("Original Luminance Histogram")
    plt.xlabel("Luminance")
    plt.ylabel("Number of Pixels")

    # Contrast-enhanced image
    plt.subplot(2, 2, 3)
    plt.imshow(enhanced_image)
    plt.title("Contrast Enhanced Image")
    plt.axis("off")

    # Enhanced luminance histogram
    plt.subplot(2, 2, 4)
    plt.hist(
        Y_stretched.flatten(),
        bins=256,
        range=(0, 255)
    )
    plt.title("Enhanced Luminance Histogram")
    plt.xlabel("Luminance")
    plt.ylabel("Number of Pixels")

    plt.tight_layout()
    plt.show()

    return enhanced_image


image = Image.open(
    "../data/hist/leh.png"
)

enhanced_image = myLinearContrastStretch(
    image,
    r1=50,
    s1=200
)

enhanced_image.save(
    "./Output Images/linear_contrast_leh.png"
)