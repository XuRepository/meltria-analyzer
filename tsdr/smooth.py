from typing import cast

import numpy as np
import scipy.ndimage as ndimg
from tsmoothie.smoother import BinnerSmoother


def moving_average(x: np.ndarray, window_size: int) -> np.ndarray:
    out = ndimg.uniform_filter1d(input=x, size=window_size, mode="constant", origin=-(window_size // 2))[
        : -(window_size - 1)
    ]
    return cast(np.ndarray, out)


def binner(x: np.ndarray, window_size: int) -> np.ndarray:
    """Smooth time series with binner method."""
    smoother: BinnerSmoother = BinnerSmoother(n_knots=int(x.size / window_size), copy=True)
    smoother.smooth(x)
    return cast(np.ndarray, smoother.smooth_data[0])
