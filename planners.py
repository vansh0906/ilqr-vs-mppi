"""Two receding-horizon trajectory optimisers on the same model and the same cost.

iLQR  (Li & Todorov 2004; Tassa et al. 2012): derivative-based. Linearise the dynamics, build a
      Gauss-Newton quadratic model of the cost, solve the LQR backward pass, line-search forward.
MPPI  (Williams et al. 2017): sampling-based. Roll out ~1000 noisy control sequences in parallel and
      average them, weighted by exp(-cost / lambda). No derivatives needed.
"""
import numpy as np

import car
import task

NX, NU = 6, 2


def rollout(x0, U, obs, mu=1.0):
    X = np.zeros((len(U) + 1, NX))
    X[0] = x0
    for k, u in enumerate(U):
        X[k + 1] = car.step(X[k], u, mu)
    J = task.cost(X[:-1], U, obs).sum() + task.cost(X[-1], np.zeros(NU), obs)
    return X, J


class ILQR:
    name = "iLQR"

    def __init__(self, horizon=25, iters=5, mu_model=1.0):
        self.N, self.iters, self.mu = horizon, iters, mu_model
        self.U = np.tile([0.0, 0.0], (horizon, 1))

    def _jacobians(self, X, U, obs):
        """Batched central differences of the dynamics and of the residuals for every step."""
        N, eps = len(U), 1e-5
        Z = np.concatenate([X[:-1], U], axis=1)                    # (N, 8)
        P = np.repeat(Z[:, None, :], 16, axis=1)                   # (N, 16, 8)
        for i in range(8):
            P[:, 2 * i, i] += eps
            P[:, 2 * i + 1, i] -= eps
        xs, us = P[..., :NX], P[..., NX:]
        F = car.step(xs, us, self.mu)                                # (N, 16, 6)
        R = task.residuals(xs, us, obs)                             # (N, 16, nr)
        dF = (F[:, 0::2] - F[:, 1::2]) / (2 * eps)                  # (N, 8, 6)
        dR = (R[:, 0::2] - R[:, 1::2]) / (2 * eps)                  # (N, 8, nr)
        A = np.transpose(dF[:, :NX], (0, 2, 1))
        B = np.transpose(dF[:, NX:], (0, 2, 1))
        r = task.residuals(X[:-1], U, obs)                          # (N, nr)
        Jx = np.transpose(dR[:, :NX], (0, 2, 1))                    # (N, nr, 6)
        Ju = np.transpose(dR[:, NX:], (0, 2, 1))                    # (N, nr, 2)
        # terminal residual Jacobian (state only)
        zN = np.repeat(X[-1][None], 12, axis=0)
        for i in range(NX):
            zN[2 * i, i] += eps
            zN[2 * i + 1, i] -= eps
        RN = task.residuals(zN, np.zeros((12, NU)), obs)
        JN = ((RN[0::2] - RN[1::2]) / (2 * eps)).T                  # (nr, 6)
        rN = task.residuals(X[-1], np.zeros(NU), obs)
        return A, B, r, Jx, Ju, rN, JN

    def plan(self, x0, obs):
        U = self.U
        X, J = rollout(x0, U, obs, self.mu)
        reg = 1e-3
        for _ in range(self.iters):
            A, B, r, Jx, Ju, rN, JN = self._jacobians(X, U, obs)
            Vx, Vxx = 2 * JN.T @ rN, 2 * JN.T @ JN
            ks, Ks = np.zeros((self.N, NU)), np.zeros((self.N, NU, NX))
            for k in range(self.N - 1, -1, -1):
                lx, lu = 2 * Jx[k].T @ r[k], 2 * Ju[k].T @ r[k]
                lxx, luu, lux = 2 * Jx[k].T @ Jx[k], 2 * Ju[k].T @ Ju[k], 2 * Ju[k].T @ Jx[k]
                Qx = lx + A[k].T @ Vx
                Qu = lu + B[k].T @ Vx
                Qxx = lxx + A[k].T @ Vxx @ A[k]
                Quu = luu + B[k].T @ Vxx @ B[k] + reg * np.eye(NU)
                Qux = lux + B[k].T @ Vxx @ A[k]
                ks[k] = -np.linalg.solve(Quu, Qu)
                Ks[k] = -np.linalg.solve(Quu, Qux)
                Vx = Qx + Ks[k].T @ Quu @ ks[k] + Ks[k].T @ Qu + Qux.T @ ks[k]
                Vxx = Qxx + Ks[k].T @ Quu @ Ks[k] + Ks[k].T @ Qux + Qux.T @ Ks[k]
                Vxx = 0.5 * (Vxx + Vxx.T)
            improved = False
            for alpha in (1.0, 0.5, 0.25, 0.1, 0.03):
                Xn, Un = np.zeros_like(X), np.zeros_like(U)
                Xn[0] = x0
                for k in range(self.N):
                    Un[k] = np.clip(U[k] + alpha * ks[k] + Ks[k] @ (Xn[k] - X[k]), car.U_MIN, car.U_MAX)
                    Xn[k + 1] = car.step(Xn[k], Un[k], self.mu)
                Jn = task.cost(Xn[:-1], Un, obs).sum() + task.cost(Xn[-1], np.zeros(NU), obs)
                if Jn < J:
                    improved = J - Jn > 1e-4 * J
                    X, U, J = Xn, Un, Jn
                    reg = max(1e-6, reg / 3)
                    break
            else:
                reg = min(1e3, reg * 10)
            if not improved and reg >= 1e3:
                break
        self.U = np.vstack([U[1:], U[-1:]])          # warm start for the next control step
        return U[0], X


class MPPI:
    name = "MPPI"

    def __init__(self, horizon=25, samples=800, lam=2.0, sigma=(0.12, 1.5), mu_model=1.0, seed=0):
        self.N, self.K, self.lam, self.mu = horizon, samples, lam, mu_model
        self.sigma = np.array(sigma)
        self.U = np.tile([0.0, 0.0], (horizon, 1))
        self.rng = np.random.default_rng(seed)

    def plan(self, x0, obs):
        eps = self.rng.normal(0.0, 1.0, (self.K, self.N, NU)) * self.sigma
        eps[0] = 0.0                                              # always keep the current plan as a sample
        Us = np.clip(self.U[None] + eps, car.U_MIN, car.U_MAX)
        x = np.repeat(x0[None], self.K, axis=0)
        S = np.zeros(self.K)
        for k in range(self.N):
            S += task.cost(x, Us[:, k], obs)
            x = car.step(x, Us[:, k], self.mu)
        S += task.cost(x, np.zeros((self.K, NU)), obs)
        w = np.exp(-(S - S.min()) / self.lam)
        w /= w.sum()
        self.U = np.clip(self.U + np.einsum("k,knu->nu", w, Us - self.U[None]), car.U_MIN, car.U_MAX)
        u0 = self.U[0].copy()
        X, _ = rollout(x0, self.U, obs, self.mu)
        self.U = np.vstack([self.U[1:], self.U[-1:]])
        return u0, X
