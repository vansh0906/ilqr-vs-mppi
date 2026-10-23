"""1/10-scale race car: dynamic bicycle model with Pacejka-style tyres, vectorised over a batch.

state x = [X, Y, yaw, vx, vy, r]   (world position, heading, body velocities, yaw rate)
input u = [steer (rad), longitudinal acceleration (m/s^2)]
All functions accept arrays with any leading batch shape (..., 6) and (..., 2).
"""
import numpy as np

M, IZ, LF, LR, G = 3.5, 0.05, 0.16, 0.17, 9.81
B_TYRE, C_TYRE = 6.0, 1.6
U_MIN = np.array([-0.40, -4.0])
U_MAX = np.array([0.40, 3.0])
DT = 0.05            # control period (s); integrated with 2 RK4 sub-steps


def f(x, u, mu=1.0):
    X, Y, yaw, vx, vy, r = np.moveaxis(x, -1, 0)
    d, a = np.moveaxis(u, -1, 0)
    vx = np.maximum(vx, 0.3)                       # the tyre model is singular at standstill
    Fzf, Fzr = M * G * LR / (LF + LR), M * G * LF / (LF + LR)
    af = d - np.arctan2(vy + LF * r, vx)
    ar = -np.arctan2(vy - LR * r, vx)
    Fyf = mu * Fzf * np.sin(C_TYRE * np.arctan(B_TYRE * af))
    Fyr = mu * Fzr * np.sin(C_TYRE * np.arctan(B_TYRE * ar))
    dx = vx * np.cos(yaw) - vy * np.sin(yaw)
    dy = vx * np.sin(yaw) + vy * np.cos(yaw)
    dvx = a - Fyf * np.sin(d) / M + vy * r
    dvy = (Fyr + Fyf * np.cos(d)) / M - vx * r
    dr = (LF * Fyf * np.cos(d) - LR * Fyr) / IZ
    return np.stack([dx, dy, r, dvx, dvy, dr], axis=-1)


def step(x, u, mu=1.0, dt=DT, sub=2):
    u = np.clip(u, U_MIN, U_MAX)
    h = dt / sub
    for _ in range(sub):
        k1 = f(x, u, mu)
        k2 = f(x + 0.5 * h * k1, u, mu)
        k3 = f(x + 0.5 * h * k2, u, mu)
        k4 = f(x + h * k3, u, mu)
        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x
