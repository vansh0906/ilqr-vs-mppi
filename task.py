"""The driving task: a winding 4 m wide road with obstacles; the cost is a sum of squared residuals,
so iLQR can use a Gauss-Newton Hessian and MPPI can evaluate the very same cost on 1000 samples."""
import numpy as np

AMP, WAVELEN, HALF_WIDTH, GOAL_X, V_REF = 1.5, 25.0, 2.0, 40.0, 5.0
CAR_R = 0.25          # car footprint radius (m)
MARGIN = 0.15         # extra clearance the planner aims for (m)
W = dict(v=1.0, lat=0.3, head=1.0, obs=400.0, wall=400.0, steer=0.5, acc=0.05)


def centre(x):
    return AMP * np.sin(2 * np.pi * x / WAVELEN)


def centre_slope(x):
    return AMP * 2 * np.pi / WAVELEN * np.cos(2 * np.pi * x / WAVELEN)


def make_obstacles(seed, n=7):
    rng = np.random.default_rng(seed)
    xs = np.linspace(6, 36, n) + rng.uniform(-1.0, 1.0, n)
    lat = rng.uniform(-1.0, 1.0, n)
    r = rng.uniform(0.35, 0.55, n)
    return np.column_stack([xs, centre(xs) + lat, r])


def residuals(x, u, obs):
    """Batch residual vector r(x, u); stage cost = sum(r^2)."""
    X, Y, yaw, vx = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    lat = Y - centre(X)
    head = np.arctan2(np.sin(yaw - np.arctan(centre_slope(X))), np.cos(yaw - np.arctan(centre_slope(X))))
    res = [np.sqrt(W["v"]) * (vx - V_REF), np.sqrt(W["lat"]) * lat, np.sqrt(W["head"]) * head,
           np.sqrt(W["wall"]) * np.maximum(0.0, np.abs(lat) - (HALF_WIDTH - CAR_R))]
    for ox, oy, orad in obs:
        dist = np.hypot(X - ox, Y - oy)
        res.append(np.sqrt(W["obs"]) * np.maximum(0.0, orad + CAR_R + MARGIN - dist))
    res += [np.sqrt(W["steer"]) * u[..., 0], np.sqrt(W["acc"]) * u[..., 1]]
    return np.stack(res, axis=-1)


def cost(x, u, obs):
    return np.sum(residuals(x, u, obs) ** 2, axis=-1)


def clearance(x, obs):
    """Distance between the car footprint and the nearest obstacle (negative = collision)."""
    return min(np.hypot(x[0] - ox, x[1] - oy) - orad - CAR_R for ox, oy, orad in obs)


def off_road(x):
    return abs(x[1] - centre(x[0])) > HALF_WIDTH - CAR_R + 1e-9
