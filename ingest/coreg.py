"""Sub-pixel image shift by phase correlation (numpy FFT only; AROSICS is not installed). PS 2.2.3: imperfect
co-registration is a confounder, and DESIGN.md §7 puts shift_px into the quality score q.

shift(ref, img) -> (dy, dx) in pixels such that img ~ ref shifted by (dy, dx). Integer peak of the normalised
cross-power spectrum, refined by a locally upsampled DFT (Guizar-Sicairos et al. 2008, the method scikit-image
uses) to 1/UPSAMPLE pixel. Invalid pixels (NaN) are filled with the image mean before the FFT.
"""

import numpy as np
from numpy.typing import NDArray

UPSAMPLE = 20  # 0.05 px resolution


def _prep(a: NDArray) -> NDArray[np.float64]:
    a = np.asarray(a, dtype=np.float64)
    a = np.where(np.isfinite(a), a, np.nanmean(a))
    return a - a.mean()


def _upsampled_dft(data: NDArray, size: int, factor: int, offsets: tuple[float, float]) -> NDArray:
    """DFT of `data` evaluated on a size x size grid, 1/factor pixel spacing, starting at `offsets`."""
    h, w = data.shape
    kc = np.fft.ifftshift(np.arange(w)) - w // 2
    kr = np.fft.ifftshift(np.arange(h)) - h // 2
    col = np.exp(-2j * np.pi / (w * factor) * np.outer(kc, np.arange(size) - offsets[1]))
    row = np.exp(-2j * np.pi / (h * factor) * np.outer(np.arange(size) - offsets[0], kr))
    return row @ data @ col


def shift(ref: NDArray, img: NDArray) -> tuple[float, float]:
    """(dy, dx) of img relative to ref, in pixels (sub-pixel)."""
    f = np.fft.fft2(_prep(ref))
    g = np.fft.fft2(_prep(img))
    r = np.conj(f) * g
    r /= np.abs(r) + 1e-12
    c = np.fft.ifft2(r)
    h, w = c.shape
    iy, ix = np.unravel_index(np.argmax(np.abs(c)), c.shape)
    dy0 = iy - h if iy > h // 2 else iy
    dx0 = ix - w if ix > w // 2 else ix
    size = int(np.ceil(UPSAMPLE * 1.5))
    centre = size // 2
    up = _upsampled_dft(np.conj(r), size, UPSAMPLE, (centre - dy0 * UPSAMPLE, centre - dx0 * UPSAMPLE))
    uy, ux = np.unravel_index(np.argmax(np.abs(up)), up.shape)
    return (float(dy0 + (uy - centre) / UPSAMPLE), float(dx0 + (ux - centre) / UPSAMPLE))


def magnitude(dy: float, dx: float) -> float:
    return float(np.hypot(dy, dx))
