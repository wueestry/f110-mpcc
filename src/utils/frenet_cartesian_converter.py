import numpy as np

from utils.splinify import SplineTrack


def convert_frenet_to_cartesian(spline, frenet_coords) -> np.array:

    co, si = np.cos(-np.pi / 2), np.sin(-np.pi / 2)
    R = np.array(((co, -si), (si, co)))

    if frenet_coords.shape[0] > 2:
        return np.array(
            [
                spline.get_coordinate(s) + d * spline.get_derivative(s).reshape(1, 2) @ R
                for s, d in frenet_coords
            ]
        ).reshape(-1, 2)

    else:

        s = frenet_coords[0]
        d = frenet_coords[1]

        return spline.get_coordinate(s) + d * spline.get_derivative(s) @ R
