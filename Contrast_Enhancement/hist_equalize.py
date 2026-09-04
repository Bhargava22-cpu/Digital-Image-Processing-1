from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def myHistEqualize(image):
    # Converting RGB image to YCbCr
    ycbcr_image = image.convert("YCbCr")
    ycbcr_array = np.array(ycbcr_image)

    # Extract luminance component
    Y = ycbcr_array[:, :, 0]

    # Histogram of luminance
    histogram = np.bincount(
        Y.flatten(),
        minlength=256
    )

    # Probability distribution
    probability = histogram / Y.size

    # Cumulative distribution function (CDF)
    cdf = np.cumsum(probability)

    # Histogram equalization mapping
    Y_equalized = (
        255 * cdf[Y]
    ).astype(np.uint8)

    # Replacing luminance component
    ycbcr_equalized = ycbcr_array.copy()
    ycbcr_equalized[:, :, 0] = Y_equalized

    enhanced_image = Image.fromarray(
        ycbcr_equalized,
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

    # Histogram equalized image
    plt.subplot(2, 2, 3)
    plt.imshow(enhanced_image)
    plt.title("Histogram Equalized Image")
    plt.axis("off")

    # Equalized luminance histogram
    plt.subplot(2, 2, 4)
    plt.hist(
        Y_equalized.flatten(),
        bins=256,
        range=(0, 255)
    )
    plt.title("Equalized Luminance Histogram")
    plt.xlabel("Luminance")
    plt.ylabel("Number of Pixels")

    plt.tight_layout()
    plt.show()

    return enhanced_image


image = Image.open(
    "../data/hist/leh.png"
)

enhanced_image = myHistEqualize(image)

enhanced_image.save(
    "./Output Images/hist_equalized_leh.png"
)