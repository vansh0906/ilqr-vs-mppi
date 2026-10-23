"""Quick self-checks. Run: python tests.py"""
import numpy as np

import car
import task
from planners import ILQR, MPPI, rollout

# straight driving: no steering keeps the car on a straight line, acceleration changes speed as commanded
x = np.array([0, 0, 0, 2.0, 0, 0.0])
for _ in range(20):
    x = car.step(x, np.array([0.0, 1.0]))
assert abs(x[1]) < 1e-9 and abs(x[3] - 3.0) < 1e-6, x
print("straight-line dynamics ... ok")

# steady left turn turns left, and the batch version equals the single version
x1 = np.array([0, 0, 0, 3.0, 0, 0.0])
for _ in range(20):
    x1 = car.step(x1, np.array([0.2, 0.0]))
assert x1[1] > 0 and x1[2] > 0
xb = car.step(np.tile([0, 0, 0, 3.0, 0, 0.0], (4, 1)), np.tile([0.2, 0.0], (4, 1)))
assert np.allclose(xb[0], car.step(np.array([0, 0, 0, 3.0, 0, 0.0]), np.array([0.2, 0.0])))
print("turning and batched dynamics ... ok")

# less grip -> less yaw rate for the same steering at speed
xa = xb_ = np.array([0, 0, 0, 4.0, 0, 0.0])
for _ in range(10):
    xa = car.step(xa, np.array([0.3, 0.0]), mu=1.0)
    xb_ = car.step(xb_, np.array([0.3, 0.0]), mu=0.5)
assert abs(xb_[5]) < abs(xa[5])
print("tyre grip limits yaw rate ... ok")

# both optimisers lower the cost of the initial (do-nothing) plan
obs = task.make_obstacles(0)
x0 = np.array([0.0, 0.0, np.arctan(task.centre_slope(0.0)), 2.0, 0.0, 0.0])
for P in (ILQR, MPPI):
    p = P()
    J0 = rollout(x0, p.U.copy(), obs)[1]
    for _ in range(3):
        p.plan(x0, obs)
    p.U = np.vstack([p.U[-1:], p.U[:-1]])      # undo the warm-start shift to score the plan from x0
    J1 = rollout(x0, p.U, obs)[1]
    assert J1 < J0, (P.name, J0, J1)
    print(f"{P.name} lowers the plan cost from {J0:.0f} to {J1:.0f} ... ok")
print("all checks passed")
