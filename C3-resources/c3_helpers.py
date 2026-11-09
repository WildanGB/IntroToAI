"""
C3 Helper Data — Computer Vision

FIT1061 Introduction to Artificial Intelligence

This module provides images, kernels, and plotting utilities for C3.
It builds on your understanding of NumPy arrays (W8) and dot products.

    from c3_helpers import (SMILEY, TEST_IMAGE, KERNELS,
                            make_face_image, clip_image,
                            show_image, show_image_grid,
                            show_kernel_results, show_edge_comparison,
                            HAND_TRACE_IMAGE, HAND_TRACE_KERNEL)

All images are 2D NumPy arrays (grayscale). Values range from 0 (black)
to 255 (white).
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------

# 8x8 smiley face — small enough to read as numbers
SMILEY = np.array([
    [  0,   0,   0,   0,   0,   0,   0,   0],
    [  0, 255, 255,   0,   0, 255, 255,   0],
    [  0, 255, 255,   0,   0, 255, 255,   0],
    [  0,   0,   0,   0,   0,   0,   0,   0],
    [  0,   0,   0,   0,   0,   0,   0,   0],
    [  0, 255,   0,   0,   0,   0, 255,   0],
    [  0,   0, 255, 255, 255, 255,   0,   0],
    [  0,   0,   0,   0,   0,   0,   0,   0],
], dtype=np.uint8)

# Larger test image with varied geometric features (deterministic)
# Includes shapes, gradient, and texture so kernels produce visually distinct results
_rng = np.random.RandomState(42)
_test = np.full((64, 64), 35.0)
_test += np.linspace(15, 0, 64).reshape(-1, 1)  # subtle vertical gradient
_test[8:24, 8:28] = 200            # bright rectangle (upper-left)
_test[36:54, 5:22] = 150           # medium rectangle (lower-left)
_y, _x = np.ogrid[:64, :64]
_circle = ((_x - 48)**2 + (_y - 18)**2) <= 8**2
_test[_circle] = 220               # bright circle (upper-right)
for _i in range(40, 60):           # checkerboard patch (lower-right)
    for _j in range(40, 60):
        if (_i // 3 + _j // 3) % 2 == 0:
            _test[_i, _j] = 185
_test += _rng.normal(0, 5, _test.shape)
TEST_IMAGE = np.clip(_test, 0, 255).astype(np.float64)


# ---------------------------------------------------------------------------
# Hand-trace data for convolution
# ---------------------------------------------------------------------------

# 5x5 image with one bright pixel in the centre
HAND_TRACE_IMAGE = np.array([
    [10, 10, 10, 10, 10],
    [10, 10, 10, 10, 10],
    [10, 10, 200, 10, 10],
    [10, 10, 10, 10, 10],
    [10, 10, 10, 10, 10],
], dtype=np.float64)

# 3x3 box blur kernel (average the neighbourhood)
HAND_TRACE_KERNEL = np.ones((3, 3)) / 9.0

# Expected output for the centre pixel:
# patch = [[10, 10, 10], [10, 200, 10], [10, 10, 10]]
# sum of element-wise multiply with kernel (all 1/9):
# (10*8 + 200*1) / 9 = 280/9 = 31.1111...
HAND_TRACE_CENTRE_EXPECTED = 280.0 / 9.0  # ≈ 31.11


# ---------------------------------------------------------------------------
# Standard kernels
# ---------------------------------------------------------------------------

KERNELS = {
    'Box blur': np.ones((3, 3)) / 9.0,

    'Gaussian blur': np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1],
    ]) / 16.0,

    'Sharpen': np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0],
    ], dtype=np.float64),

    'Sobel (horiz)': np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1],
    ], dtype=np.float64),

    'Sobel (vert)': np.array([
        [-1,  0,  1],
        [-2,  0,  2],
        [-1,  0,  1],
    ], dtype=np.float64),
}


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def clip_image(img):
    """Clip pixel values to valid range [0, 255] and convert to uint8.

    Args:
        img: NumPy array (any numeric type).

    Returns:
        NumPy array with values clipped to [0, 255], dtype uint8.
    """
    return np.clip(img, 0, 255).astype(np.uint8)


def make_face_image(face_value, bg_value, size=48):
    """Create a simple 'face' with features on a uniform background.

    The face has eyes, nose, and mouth whose contrast scales with
    the face-to-background contrast.  This makes low-contrast faces
    lose visible features — demonstrating why edge-based classifiers
    struggle with certain skin tones.

    Args:
        face_value: Pixel intensity for the face (0--255).
        bg_value: Pixel intensity for the background (0--255).
        size: Image dimension (square).

    Returns:
        2D NumPy array of shape (size, size), dtype float64.
    """
    rng = np.random.RandomState(7)
    img = np.full((size, size), bg_value, dtype=np.float64)
    y, x = np.ogrid[:size, :size]
    centre = size // 2

    # Oval face shape
    oval = ((x - centre)**2 / 12**2 + (y - centre)**2 / 16**2) <= 1
    img[oval] = face_value

    # Feature intensity scales with face-bg contrast
    contrast = abs(face_value - bg_value)
    feat_val = face_value - contrast * 0.4

    # Eyes (two small circles)
    for eye_x in [centre - 5, centre + 5]:
        eye_mask = ((x - eye_x)**2 + (y - (centre - 6))**2) <= 2.0**2
        img[eye_mask] = feat_val

    # Nose (small vertical mark)
    img[centre - 1:centre + 2, centre] = feat_val

    # Mouth (short horizontal line)
    img[centre + 6, centre - 4:centre + 5] = feat_val
    img[centre + 7, centre - 3:centre + 4] = feat_val

    # Slight noise for realism
    img += rng.normal(0, 3, img.shape)
    return np.clip(img, 0, 255)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def show_image(image, title="Image", cmap='gray', vmin=0, vmax=255,
               show_values=False):
    """Display a single grayscale image.

    Args:
        image: 2D NumPy array.
        title: Plot title.
        cmap: Colour map (default 'gray').
        vmin, vmax: Display range.
        show_values: If True, overlay pixel values (best for small images).
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
    ax.set_title(title, fontweight='bold', fontsize=12)

    if show_values:
        h, w = image.shape[:2]
        for i in range(h):
            for j in range(w):
                val = int(image[i, j]) if image[i, j] == int(image[i, j]) \
                    else f"{image[i, j]:.1f}"
                colour = 'white' if image[i, j] < 128 else 'black'
                ax.text(j, i, str(val), ha='center', va='center',
                        fontsize=max(6, 48 // max(h, w)), color=colour)

    ax.axis('off')
    plt.tight_layout()
    plt.show()


def show_image_grid(images, titles, suptitle="", cols=3, cmap='gray',
                    vmin=0, vmax=255):
    """Display multiple images in a grid.

    Args:
        images: List of 2D NumPy arrays.
        titles: List of title strings.
        suptitle: Overall figure title.
        cols: Number of columns.
        cmap: Colour map.
        vmin, vmax: Display range.
    """
    n = len(images)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for i, (img, title) in enumerate(zip(images, titles)):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontweight='bold', fontsize=10)
        ax.axis('off')

    # Hide unused axes
    for i in range(n, rows * cols):
        r, c = divmod(i, cols)
        axes[r, c].axis('off')

    if suptitle:
        plt.suptitle(suptitle, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()


def show_kernel_results(image, convolve_fn, kernels=None,
                        title="Kernel Comparison"):
    """Apply multiple kernels to an image and display results.

    Args:
        image: 2D NumPy array (grayscale).
        convolve_fn: Student's convolve2d function.
        kernels: Dict of {name: kernel_array}. Defaults to KERNELS.
        title: Overall figure title.
    """
    if kernels is None:
        kernels = KERNELS

    n = len(kernels) + 1  # +1 for original
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.atleast_1d(axes).flatten()

    axes[0].imshow(image, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title('Original', fontweight='bold')
    axes[0].axis('off')

    for i, (name, kernel) in enumerate(kernels.items(), 1):
        result = convolve_fn(image, kernel)
        if 'Sobel' in name:
            display = np.abs(result)
        else:
            display = np.clip(result, 0, 255)
        axes[i].imshow(display, cmap='gray')
        axes[i].set_title(name, fontweight='bold', fontsize=10)
        axes[i].axis('off')

    for j in range(n, len(axes)):
        axes[j].axis('off')

    plt.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()


def show_edge_comparison(convolve_fn, edge_fn):
    """Compare edge detection on faces with different contrast levels.

    Demonstrates that edge detection strength depends on contrast,
    which is one mechanism behind the Gender Shades disparity.

    Args:
        convolve_fn: Student's convolve2d function (used by edge_fn).
        edge_fn: Student's edge_magnitude function.
    """
    faces = [
        (make_face_image(200, 40), 'High contrast\n(light face, dark bg)'),
        (make_face_image(140, 80), 'Medium contrast'),
        (make_face_image(80, 60),  'Low contrast\n(dark face, dark bg)'),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    edge_strengths = []
    for i, (face, label) in enumerate(faces):
        edges = edge_fn(face)
        strength = np.mean(edges)
        edge_strengths.append(strength)

        axes[0, i].imshow(face, cmap='gray', vmin=0, vmax=255)
        axes[0, i].set_title(label, fontweight='bold', fontsize=10)
        axes[0, i].axis('off')

        axes[1, i].imshow(edges, cmap='hot')
        axes[1, i].set_title(f'Edge strength: {strength:.1f}', fontsize=10)
        axes[1, i].axis('off')

    plt.suptitle('Edge Detection Depends on Contrast — A Fairness Problem',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()

    print(f"Edge strength — High contrast: {edge_strengths[0]:.1f}, "
          f"Medium: {edge_strengths[1]:.1f}, "
          f"Low: {edge_strengths[2]:.1f}")
    ratio = edge_strengths[0] / max(edge_strengths[2], 0.01)
    print(f"Ratio (high / low): {ratio:.1f}x")
    print()
    print("Same shape, same algorithm — but the low-contrast face")
    print("produces weaker edge features. A classifier trained on")
    print("mostly high-contrast faces will struggle with low-contrast ones.")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("C3 Helpers — Computer Vision")
    print(f"  SMILEY shape: {SMILEY.shape}, dtype: {SMILEY.dtype}")
    print(f"  TEST_IMAGE shape: {TEST_IMAGE.shape}, range: "
          f"[{TEST_IMAGE.min():.1f}, {TEST_IMAGE.max():.1f}]")
    print(f"  HAND_TRACE_IMAGE shape: {HAND_TRACE_IMAGE.shape}")
    print(f"  HAND_TRACE_CENTRE_EXPECTED: {HAND_TRACE_CENTRE_EXPECTED:.4f}")
    print(f"  Kernels: {list(KERNELS.keys())}")
    print(f"  make_face_image(200, 40) shape: {make_face_image(200, 40).shape}")
