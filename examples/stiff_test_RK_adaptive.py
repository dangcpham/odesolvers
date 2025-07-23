import numpy as np
import matplotlib.pyplot as plt
import numba as nb
import odesolvers
import time
from scipy.integrate import solve_ivp

@nb.njit
def f_stiff(t, y):
    if t < 0.3:
        return np.array([-10.0 * y[0]])
    else:
        return np.array([-1000.0 * y[0]])

y0 = np.array([19.0])
t0 = 0.0
t_end = 1.0

# Adaptive ERK coefficients (Dormand-Prince 5(4))
A_dp = np.array([
    [0, 0, 0, 0, 0, 0, 0],
    [1/5, 0, 0, 0, 0, 0, 0],
    [3/40, 9/40, 0, 0, 0, 0, 0],
    [44/45, -56/15, 32/9, 0, 0, 0, 0],
    [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0, 0],
    [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0, 0],
    [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]
])
b_dp = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0])
b_star_dp = np.array([5179/57600, 0, 7571/16695, 393/640,
                      -92097/339200, 187/2100, 1/40])
c_dp = np.array([0, 1/5, 3/10, 4/5, 8/9, 1.0, 1.0])

# time comparison with SciPy's DOP853
rk = odesolvers.AdaptiveERK(A_dp, b_dp, b_star_dp, c_dp, order=5, t0=t0)
# pre-compile the Numba functions
rk.integrate(f_stiff, y0, t_end)
rk.t = t0

n_runs = 100
times_custom = []
times_scipy = []
for _ in range(n_runs):
    t1 = time.perf_counter()
    rk.integrate(f_stiff, y0, t_end)
    rk.t = t0
    t2 = time.perf_counter()

    t3 = time.perf_counter()
    solve_ivp(f_stiff, (t0, t_end), y0, method='DOP853')
    t4 = time.perf_counter()
    
    times_custom.append(t2 - t1)
    times_scipy.append(t4 - t3)

mean_custom = np.mean(times_custom)
mean_scipy = np.mean(times_scipy)

print(f"Average runtime over {n_runs} runs:")
print(f"  AdaptiveERK       : {mean_custom:.2E} seconds")
print(f"  SciPy DOP853      : {mean_scipy:.2E} seconds")


# check timestepping
ts, ys, dts = [], [], []
t, y = t0, y0.copy()

rk = odesolvers.AdaptiveERK(A_dp, b_dp, b_star_dp, c_dp, order=5, t0=t0)
while t < t_end:
    dts.append(rk.dt)
    ts.append(t)
    ys.append(y)
    t, y = rk.step(f_stiff, y)

ts = np.array(ts)
ys = np.array(ys)

plt.plot(ts, dts, '.-', label="Time step size", color="tab:orange")
plt.xlabel("Time $t$")
plt.ylabel("Timestep $dt$")
plt.yscale("log")
plt.title("Adaptive Timestepping")
plt.legend()
plt.show()

from scipy.integrate import solve_ivp
sol = solve_ivp(f_stiff, (t0, t_end), y0, method='DOP853')
plt.plot(sol.t, sol.y[0], '.-', label="SciPy DOP853", color="tab:blue",)
plt.plot(ts, ys, '.-', label="Adaptive ERK", color="tab:green")
plt.xlabel("Time $t$")
plt.ylabel("y(t)")
plt.legend()
plt.show()