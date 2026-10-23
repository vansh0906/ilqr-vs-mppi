"""iLQR vs MPPI as model-predictive controllers for a 1/10-scale car on a winding road with obstacles.

python main.py              (10 random obstacle layouts x 2 planners x 2 grip cases, in parallel;
                             about 3-5 minutes on an M-series Mac)
python main.py --layouts 5  (faster)
"""
import argparse
import os
import time
from multiprocessing import Pool

import numpy as np

import car
import task
from planners import ILQR, MPPI

T_MAX = 20.0


def drive(planner, obs, mu_plant=1.0, log=False):
    x = np.array([0.0, 0.0, np.arctan(task.centre_slope(0.0)), 2.0, 0.0, 0.0])
    xs, plans, comp, steer = [x.copy()], [], [], []
    min_clear, crashed, off = np.inf, False, False
    t = 0.0
    while t < T_MAX and x[0] < task.GOAL_X:
        t0 = time.perf_counter()
        u, X = planner.plan(x, obs)
        comp.append(time.perf_counter() - t0)
        steer.append(float(np.clip(u[0], car.U_MIN[0], car.U_MAX[0])))
        x = car.step(x, u, mu_plant)
        t += car.DT
        c = task.clearance(x, obs)
        min_clear = min(min_clear, c)
        crashed |= c < 0
        off |= task.off_road(x)
        if log:
            xs.append(x.copy()); plans.append(X)
    done = x[0] >= task.GOAL_X
    return dict(success=done and not crashed and not off, crashed=crashed, off=off, time=t if done else np.nan,
                clear=min_clear, ms=1000 * np.mean(comp), xs=np.array(xs), plans=plans,
                steer_rate=float(np.degrees(np.mean(np.abs(np.diff(steer)))) / car.DT))


def _job(args):
    seed, mu, which = args
    obs = task.make_obstacles(seed)
    planner = ILQR() if which == "iLQR" else MPPI()
    r = drive(planner, obs, mu, log=(seed == 0 and mu == 1.0))
    if not (seed == 0 and mu == 1.0):
        r["xs"], r["plans"] = None, None
    return seed, mu, which, obs, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layouts", type=int, default=10)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    print("1/10-scale car (dynamic bicycle, Pacejka-style tyres), 20 Hz MPC, 1.25 s horizon")
    print(f"Winding 4 m road, {task.GOAL_X:.0f} m long, 7 random obstacles; target speed {task.V_REF} m/s; "
          f"{a.layouts} random layouts, {a.workers} parallel workers\n")
    cases = (("model = reality", 1.0), ("40% less grip than model", 0.6))
    jobs = [(s, mu, w) for _, mu in cases for w in ("iLQR", "MPPI") for s in range(a.layouts)]
    with Pool(a.workers) as pool:
        out = pool.map(_job, jobs)
    print(f"{'case':<26}{'planner':<8}{'success':>9}{'collisions':>12}{'off-road':>10}{'time':>9}"
          f"{'min clearance':>15}{'steer rate':>12}{'compute/step':>14}")
    keep = {}
    for case, mu in cases:
        for name in ("iLQR", "MPPI"):
            res = [r for s, m, w, obs, r in out if m == mu and w == name]
            for s, m, w, obs, r in out:
                if s == 0 and m == 1.0 and w == name:
                    keep[name] = (obs, r)
            succ = sum(r["success"] for r in res)
            col = sum(r["crashed"] for r in res)
            off = sum(r["off"] for r in res)
            tm = np.nanmean([r["time"] for r in res])
            cl = np.mean([r["clear"] for r in res])
            sr = np.mean([r["steer_rate"] for r in res])
            ms = np.mean([r["ms"] for r in res])
            print(f"{case:<26}{name:<8}{succ:>5}/{a.layouts:<3}{col:>10}{off:>10}{tm:>7.2f} s{cl * 100:>12.0f} cm"
                  f"{sr:>8.0f} °/s{ms:>11.0f} ms")
        print()
    print("min clearance = mean over layouts of the closest approach to any obstacle (negative = hit)")
    print("steer rate = mean |change in steering| per second (lower = smoother)")
    print(f"(ran in {time.time() - t0:.0f} s)")
    if not a.no_plot:
        plot(keep)


def plot(keep):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 1, figsize=(13, 4.4), sharex=True)
    xx = np.linspace(-1, task.GOAL_X + 1, 400)
    for ax, (name, (obs, r)) in zip(axs, keep.items()):
        c = task.centre(xx)
        ax.plot(xx, c + task.HALF_WIDTH, "k", lw=1); ax.plot(xx, c - task.HALF_WIDTH, "k", lw=1)
        ax.plot(xx, c, "k:", lw=0.6)
        for ox, oy, orad in obs:
            ax.add_patch(plt.Circle((ox, oy), orad, color="tab:red", alpha=.6))
        for X in r["plans"][::8]:
            ax.plot(X[:, 0], X[:, 1], color="tab:gray", lw=0.6, alpha=.6)
        sc = ax.scatter(r["xs"][:, 0], r["xs"][:, 1], c=r["xs"][:, 3], cmap="viridis", s=6, vmin=0, vmax=5)
        ax.set_aspect("equal"); ax.set_ylim(-4, 4)
        ax.set_title(f"{name}: {'reached the goal' if r['success'] else 'failed'} in {r['time']:.1f} s, "
                     f"closest approach {r['clear'] * 100:.0f} cm (grey = predicted plans)")
    fig.colorbar(sc, ax=axs, label="speed (m/s)", shrink=0.8)
    axs[1].set_xlabel("x (m)")
    fig.savefig("ilqr_vs_mppi.png", dpi=120, bbox_inches="tight")
    print("saved ilqr_vs_mppi.png")


if __name__ == "__main__":
    main()
