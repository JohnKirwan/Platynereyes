from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image, ImageOps

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None  # plotting optional


__all__ = [
    "sharpness_glimpse",
    "remove_red_channel",
]


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _rescale_01_per_channel(arr: np.ndarray) -> np.ndarray:
    """
    Match plotrix::rescale to [0,1] *per channel*, like the R code.
    arr: float64, shape (H, W, 3)
    """
    out = np.empty_like(arr, dtype=np.float64)
    for c in range(3):
        v = arr[..., c]
        vmin = np.min(v)
        vmax = np.max(v)
        if vmax > vmin:
            out[..., c] = (v - vmin) / (vmax - vmin)
        else:
            out[..., c] = 0.0
    return out


def _srgb_to_linear(srgb: np.ndarray) -> np.ndarray:
    """Piecewise sRGB -> linear (inverse EOTF), expects [0,1]."""
    a = 0.055
    out = np.empty_like(srgb, dtype=np.float64)
    mask = srgb <= 0.04045
    out[mask] = srgb[mask] / 12.92
    out[~mask] = ((srgb[~mask] + a) / (1.0 + a)) ** 2.4
    return out


def _linear_to_srgb(lin: np.ndarray) -> np.ndarray:
    """Piecewise linear -> sRGB (EOTF), returns nominal [0,1] (may exceed slightly)."""
    a = 0.055
    out = np.empty_like(lin, dtype=np.float64)
    mask = lin < 0.0031308
    out[mask] = lin[mask] * 12.92
    out[~mask] = (1.0 + a) * np.power(np.maximum(lin[~mask], 0.0), 1.0 / 2.4) - a
    return out


def _rescale_to_cap_at_one(channel: np.ndarray) -> np.ndarray:
    """
    Mirrors the R code's 'if (max > 1) rescale(x, newrange=c(min(x), 1))'.
    This leaves the channel's min as-is and compresses the dynamic range so max becomes 1.
    """
    vmax = np.max(channel)
    if vmax <= 1.0:
        return channel
    vmin = np.min(channel)
    if vmax == vmin:
        return np.minimum(channel, 1.0)
    return ((channel - vmin) / (vmax - vmin)) * (1.0 - vmin) + vmin


def _build_blur_kernel(width_px: int,
                       pixels_per_degree: float,
                       eye_res_x: float,
                       eye_res_y: float) -> np.ndarray:
    """
    Vectorized blur (MTF) kernel, centered (DC at the center) to be used with fftshifted spectra.
    Follows the R code's math exactly, including the anisotropic effective resolution.
    """
    # Grid with center at zero
    center = width_px // 2
    xs = np.arange(width_px, dtype=np.float64) - center
    ys = np.arange(width_px, dtype=np.float64) - center
    X, Y = np.meshgrid(xs, ys, indexing="ij")

    R = np.sqrt(X * X + Y * Y)  # radius in pixels
    # Frequency in cycles/degree, with the odd 'round' from the R code
    freq = np.round(R) / width_px * pixels_per_degree

    # Avoid divide-by-zero warnings; set sin/cos to 0 at center (R code effectively does this)
    with np.errstate(invalid="ignore", divide="ignore"):
        mySin = np.where(R > 0, Y / R, 0.0)
        myCos = np.where(R > 0, X / R, 0.0)
        denom = np.sqrt((eye_res_y * myCos) ** 2 + (eye_res_x * mySin) ** 2)
        # eyeResolution as in R:
        eye_res_eff = (eye_res_x * eye_res_y) / denom  # inf at center; handled below

        blur = np.exp(-3.56 * (eye_res_eff * freq) ** 2)

    # Explicitly set the center to 1.0 (R code does this)
    blur[center, center] = 1.0
    # Replace any NaNs produced at center (inf*0) with 1.0
    blur = np.nan_to_num(blur, nan=1.0, posinf=1.0, neginf=1.0)

    return blur.astype(np.float64)


def _ensure_rgb_3ch(img: Image.Image) -> Image.Image:
    """
    Apply EXIF orientation (if present) and ensure the image is 3-channel RGB.
    """
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        # Some formats lack EXIF; ignore and continue.
        pass
    if img.mode != "RGB":
        # Strip alpha or convert from L/other.
        img = img.convert("RGB")
    return img


def remove_red_channel(photo: Union[str, Path, Image.Image],
                       output: Optional[Union[str, Path]] = None,
                       quality: int = 95) -> Image.Image:
    """
    Zero the red channel of an image. Accepts file paths or PIL Images.

    Parameters
    ----------
    photo : path | PIL.Image
        Source image (any mode); converted to RGB with EXIF orientation applied.
    output : str or Path, optional
        If provided, save a JPEG to this path ('.jpg' appended if missing).
    quality : int
        JPEG quality when saving (default 95).

    Returns
    -------
    PIL.Image
        The red-removed RGB image.
    """
    if isinstance(photo, Image.Image):
        img = _ensure_rgb_3ch(photo)
    else:
        img = _ensure_rgb_3ch(Image.open(str(photo)))

    arr = np.array(img, dtype=np.uint8, copy=True)
    arr[..., 0] = 0
    out = Image.fromarray(arr, mode="RGB")

    if output is not None:
        out_path = str(output)
        if not out_path.lower().endswith((".jpg", ".jpeg")):
            out_path = f"{out_path}.jpg"
        out.save(out_path, format="JPEG", quality=int(quality), optimize=True, progressive=True)

    return out


def sharpness_glimpse(photo: Union[str, Path, np.ndarray, Image.Image] = None,
                distance: float = 2.0,
                real_width: float = 2.0,
                eye_resolution_x: float = 0.2,
                eye_resolution_y: Optional[float] = None,
                plot: bool = True,
                output_matrix: bool = False,
                output: str = "test.jpg") -> np.ndarray:
    """
    Python port of AcuityView::AcuityView (R).

    Parameters
    ----------
    photo : path | np.ndarray(H,W,3) | PIL.Image
        The photo to alter. Must be 3-channel RGB, square, power-of-two side length.
    distance : float
        Distance from viewer to object (same units as real_width).
    real_width : float
        Real width of the scene corresponding to image width (same units as distance).
    eye_resolution_x : float
        Eye resolution (deg) in X.
    eye_resolution_y : float or None
        Eye resolution (deg) in Y; defaults to eye_resolution_x if None.
    plot : bool
        If True, show before/after using matplotlib (if available).
    output : str
        Filename to save the result (.png, .bmp, .jpeg, .jpg).

    Returns
    -------
    np.ndarray
        The processed image as uint8 array (H, W, 3).
    """
    if eye_resolution_y is None:
        eye_resolution_y = eye_resolution_x

    if distance <= 0 or real_width <= 0:
        raise ValueError("distance and real_width must be > 0")

    # --- Load image ---
    if isinstance(photo, (str, Path)):
        img = Image.open(str(photo))
        img = _ensure_rgb_3ch(img)
        arr = np.asarray(img, dtype=np.float64)
    elif isinstance(photo, Image.Image):
        img = _ensure_rgb_3ch(photo)
        arr = np.asarray(img, dtype=np.float64)
    elif isinstance(photo, np.ndarray):
        if photo.ndim != 3 or photo.shape[2] != 3:
            raise ValueError("NumPy input must have shape (H, W, 3)")
        arr = photo.astype(np.float64, copy=False)
    else:
        raise ValueError("photo must be a file path, PIL.Image, or np.ndarray (H,W,3)")

    H, W, C = arr.shape
    if C != 3:
        raise ValueError("Input image must be 3-channel (RGB).")
    if H != W:
        raise ValueError("Image must be square. Got %dx%d." % (H, W))
    if not _is_power_of_two(W):
        raise ValueError("Image side length must be a power of 2. Got %d." % W)

    # --- Geometry: width in degrees & pixels/degree ---
    width_in_degrees = np.degrees(2.0 * np.arctan(real_width / (2.0 * distance)))
    pixels_per_degree = W / width_in_degrees

    # --- Build blur (centered) ---
    blur = _build_blur_kernel(W, pixels_per_degree, eye_resolution_x, eye_resolution_y)

    # --- Color pipeline: rescale->linearize ---
    # Match R's per-channel rescale to [0,1] before linearization
    arr01 = _rescale_01_per_channel(arr)
    lin = _srgb_to_linear(arr01)

    # --- Frequency-domain filtering per channel ---
    out_lin = np.empty_like(lin, dtype=np.float64)
    for c in range(3):
        plane = lin[..., c]

        # FFT, shift to center, multiply blur, unshift, IFFT
        F = np.fft.fft2(plane)
        Fc = np.fft.fftshift(F)
        Fc_blur = Fc * blur
        F_un = np.fft.ifftshift(Fc_blur)
        img_c = np.fft.ifft2(F_un)

        # R code uses Mod(ifft) -> magnitude; here real part is fine. Use abs() for safety.
        out_lin[..., c] = np.abs(img_c)

    # --- Back to sRGB ---
    srgb = _linear_to_srgb(out_lin)

    # If any channel exceeds 1, rescale that channel like the R code
    for c in range(3):
        srgb[..., c] = _rescale_to_cap_at_one(srgb[..., c])

    # Clip to [0,1] and convert to uint8
    srgb = np.clip(srgb, 0.0, 1.0)
    out_uint8 = (srgb * 255.0 + 0.5).astype(np.uint8)

    # --- Save ---
    ext = Path(output).suffix.lower()
    if ext not in {".png", ".bmp", ".jpeg", ".jpg"}:
        raise ValueError('Output must be one of .png, .bmp, .jpeg, .jpg')
    Image.fromarray(out_uint8, mode="RGB").save(output)

    # --- Plot (optional) ---
    if plot and plt is not None:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(arr.astype(np.uint8))
        plt.axis("off")
        plt.title("Before")
        plt.subplot(1, 2, 2)
        plt.imshow(out_uint8)
        plt.axis("off")
        plt.title("After")
        plt.tight_layout()
        plt.show()

    if output_matrix:
        warnings.warn(
            "output_matrix is deprecated; sharpness_glimpse now always returns the processed array.",
            DeprecationWarning,
            stacklevel=2,
        )
    return out_uint8
