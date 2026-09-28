import numpy as np
from scipy.ndimage import shift as nd_shift

from ingest.coreg import magnitude, shift


def _scene(n=128, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n, n))
    # smooth texture like a 10 m scene (fields, roads), not white noise
    k = np.outer(np.hanning(9), np.hanning(9))
    return np.real(np.fft.ifft2(np.fft.fft2(base) * np.fft.fft2(k / k.sum(), s=base.shape)))


def test_integer_and_subpixel_shifts_are_recovered():
    ref = _scene()
    for dy, dx in ((0.0, 0.0), (3.0, -2.0), (0.4, 0.0), (-0.3, 0.6)):
        img = nd_shift(ref, (dy, dx), order=3, mode="wrap")
        gy, gx = shift(ref, img)
        assert abs(gy - dy) < 0.15 and abs(gx - dx) < 0.15, (dy, dx, gy, gx)


def test_nan_pixels_do_not_break_the_estimate():
    ref = _scene(seed=1)
    img = nd_shift(ref, (1.0, 0.5), order=3, mode="wrap")
    img[:20, :20] = np.nan  # cloud / nodata patch
    gy, gx = shift(ref, img)
    assert abs(gy - 1.0) < 0.25 and abs(gx - 0.5) < 0.25
    assert magnitude(3, 4) == 5.0
