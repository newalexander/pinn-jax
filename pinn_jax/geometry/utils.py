import numpy as np

from functools import wraps
from typing import Tuple, Callable


def vectorize(**kwargs):
    """numpy.vectorize wrapper that works with instance methods.

    References:

    - https://numpy.org/doc/stable/reference/generated/numpy.vectorize.html
    - https://stackoverflow.com/questions/48981501/is-it-possible-to-numpy-vectorize-an-instance-method
    - https://github.com/numpy/numpy/issues/9477
    """

    def decorator(fn):
        vectorized = np.vectorize(fn, **kwargs)

        @wraps(fn)
        def wrapper(*args):
            return vectorized(*args)

        return wrapper

    return decorator


def get_close_to_line(endpoints: Tuple[np.ndarray, np.ndarray], n_spatial_dimensions=2) -> Callable:
    def close_to_line(x: np.ndarray, _):
        """decide if `x` is (numerically) close to a line segment with endpoints `endpoints`
        see here: https://stackoverflow.com/questions/910882/how-can-i-tell-if-a-point-is-nearby-a-certain-line
        basic idea is to compute the right angle distance between `x` and the line, then we look for points that
        are within a circle of radius ``dist(mid_point, start_point) + `eps``
        """
        r0, r1 = endpoints
        midpt = (r0 + r1) / 2.

        d = np.linalg.norm(np.subtract(midpt, x[:n_spatial_dimensions]))
        # warning: eps may cause some trouble numerically.
        eps = np.finfo(np.float32).eps
        tmp = np.linalg.norm(np.subtract(midpt, r0)) + eps
        if d < tmp:
            t0 = (r1[0] - r0[0]) * (r0[1] - x[1])
            t1 = (r0[0] - x[0]) * (r1[1] - r0[1])
            distance = np.abs(t0 - t1) / (2 * tmp)
            return np.isclose(distance, 0.0)
        else:
            return False

    return close_to_line
