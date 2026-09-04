from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def myHistMatch(image, reference, num_bins):

    # Convert source and reference images to YCbCr
    ycbcr_image = image.convert("YCbCr")
    ycbcr_reference = reference.convert("YCbCr")

    ycbcr_array = np.array(ycbcr_image)
    ycbcr_ref_array = np.array(ycbcr_reference)

    # Convert to RGB for creating foreground masks
    rgb_array = np.array(image.convert("RGB"))
    rgb_ref_array = np.array(reference.convert("RGB"))

    # Ignore black background
    mask = np.any(rgb_array > 10, axis=2)
    ref_mask = np.any(rgb_ref_array > 10, axis=2)

    # Output image
    ycbcr_matched = ycbcr_array.copy()

    bin_width = 256 / num_bins

    # Histogram matching for Y, Cb and Cr
    for channel in range(3):

        # Source channel
        source = ycbcr_array[:, :, channel]

        # Reference channel
        target = ycbcr_ref_array[:, :, channel]

        # Source foreground pixels
        source_values = source[mask]

        # Reference foreground pixels
        reference_values = target[ref_mask]

        source_bins = (source_values / bin_width).astype(int)

        reference_bins = (reference_values / bin_width).astype(int)

        source_histogram = np.bincount(
            source_bins,
            minlength=num_bins
        ).astype(float)

        reference_histogram = np.bincount(
            reference_bins,
            minlength=num_bins
        ).astype(float)

        # Source probability distribution
        source_probability = (
            source_histogram / source_values.size
        )

        # Reference probability distribution
        reference_probability = (
            reference_histogram / reference_values.size
        )

        # Source CDF
        source_cdf = np.cumsum(source_probability)

        # Reference CDF
        reference_cdf = np.cumsum(reference_probability)

        # Histogram matching lookup table
        mapping = np.zeros(num_bins, dtype=np.uint8)

        for i in range(num_bins):

            index = np.argmin(
                np.abs(
                    reference_cdf - source_cdf[i]
                )
            )

            mapping[i] = int ((index + 0.5) * bin_width)

        # Apply mapping only to foreground
        source_bin_image = (source / bin_width).astype(int)

        ycbcr_matched[:, :, channel][mask] = (
            mapping[source_bin_image[mask]]
        )

    # Convert matched YCbCr image back to RGB
    enhanced_image = Image.fromarray(
        ycbcr_matched,
        mode="YCbCr"
    ).convert("RGB")

    # Plot
    plt.figure(figsize=(12, 8))

    # Original image
    plt.subplot(2, 2, 1)
    plt.imshow(image)
    plt.title("Original Retina")
    plt.axis("off")

    # Reference image
    plt.subplot(2, 2, 2)
    plt.imshow(reference)
    plt.title("Reference Retina")
    plt.axis("off")

    # Matched image
    plt.subplot(2, 2, 3)
    plt.imshow(enhanced_image)
    plt.title(f"Histogram Matched Image (num_bins={num_bins})")
    plt.axis("off")

    # Show luminance histograms
    plt.subplot(2, 2, 4)

    source_Y = ycbcr_array[:, :, 0]
    reference_Y = ycbcr_ref_array[:, :, 0]
    matched_Y = ycbcr_matched[:, :, 0]

    plt.hist(
        source_Y[mask],
        bins=num_bins,
        range=(0, 255),
        alpha=0.5,
        label="Original"
    )

    plt.hist(
        reference_Y[ref_mask],
        bins=num_bins,
        range=(0, 255),
        alpha=0.5,
        label="Reference"
    )

    plt.hist(
        matched_Y[mask],
        bins=num_bins,
        range=(0, 255),
        alpha=0.5,
        label="Matched"
    )

    plt.title("Luminance Histograms")
    plt.xlabel("Luminance")
    plt.ylabel("Number of Pixels")
    plt.legend()

    plt.tight_layout()
    plt.show()

    return enhanced_image


# Source image
image = Image.open(
    "../data/hist/retina.png"
)

# Reference image
reference = Image.open(
    "../data/hist/retinaRef.png"
)

# Histogram matching
enhanced_image = myHistMatch(
    image,
    reference, 256
)

# Save result
enhanced_image.save(
    "./Output Images/hist_matched_retina_sh.png"
)

"""
fixed - > 128
sl -> 32
sh -> 256
"""

