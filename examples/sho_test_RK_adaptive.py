import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.integrate import solve_ivp
import odesolvers
import numba as nb

# SHO test
@nb.njit
def f(t, yv):
    return np.array([yv[1], -yv[0]])

y0 = np.array([1.0, 0.0])
t0 = 0.0
t_end = 10.0

# DOP tableau
A_dp = np.array([
    [0,         0,         0,        0,         0,       0,     0],
    [1/5,       0,         0,        0,         0,       0,     0],
    [3/40,      9/40,      0,        0,         0,       0,     0],
    [44/45,    -56/15,     32/9,     0,         0,       0,     0],
    [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0, 0],
    [9017/3168,  -355/33, 46732/5247, 49/176, -5103/18656, 0, 0],
    [35/384,     0,       500/1113, 125/192, -2187/6784, 11/84, 0]
])
b_dp = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0])
b_star_dp = np.array([5179/57600, 0, 7571/16695, 393/640,
                      -92097/339200, 187/2100, 1/40])
c_dp = np.array([0, 1/5, 3/10, 4/5, 8/9, 1.0, 1.0])

rk = odesolvers.AdaptiveERK(A_dp, b_dp, b_star_dp, c_dp, order=5, t0=t0)
# # compile the Numba functions
rk.integrate(f, y0, 0.1)
rk.t = t0

# Benchmark 100 runtimes
n_runs = 100
times_custom = []
times_scipy = []

for _ in range(n_runs):
    t1 = time.perf_counter()
    rk.integrate(f, y0, t_end)
    rk.t = t0 # reset time
    t2 = time.perf_counter()
    times_custom.append(t2 - t1)

    t3 = time.perf_counter()
    solve_ivp(f, (t0, t_end), y0, method='DOP853')
    t4 = time.perf_counter()
    times_scipy.append(t4 - t3)

mean_custom = np.mean(times_custom)
mean_scipy = np.mean(times_scipy)

print(f"Average runtime over {n_runs} runs:")
print(f"  AdaptiveERK       : {mean_custom:.2E} seconds")
print(f"  SciPy DOP853      : {mean_scipy:.2E} seconds")


# check for accuracy
ts = np.linspace(t0, t_end, 100)
ys = []
t = t0
y = y0.copy()
rk = odesolvers.AdaptiveERK(A_dp, b_dp, b_star_dp, c_dp, order=5)

for t in ts:
    _, y = rk.integrate(f, y, t)
    ys.append(y.copy())
ys = np.array(ys)

y_exact = np.cos(ts)
v_exact = -np.sin(ts)

err_y = np.mean(np.abs(ys[:, 0] - y_exact))
err_v = np.mean(np.abs(ys[:, 1] - v_exact))
print(f"Average error in y: {err_y:.2E}")
print(f"Average error in v: {err_v:.2E}")

assert err_y < 1e-6, "Error in y exceeds tolerance"
assert err_v < 1e-6, "Error in v exceeds tolerance"
