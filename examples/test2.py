import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import odesolvers

# Dormand-Prince tableau
A_dp = np.array([
    [0,         0,         0,        0,         0,       0,     0],
    [1/5,       0,         0,        0,         0,       0,     0],
    [3/40,      9/40,      0,        0,         0,       0,     0],
    [44/45,    -56/15,     32/9,     0,         0,       0,     0],
    [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0, 0],
    [9017/3168,  -355/33, 46732/5247, 49/176, -5103/18656, 0, 0],
    [35/384,     0,       500/1113, 125/192, -2187/6784, 11/84, 0]
])

b_dp = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0])  # 5th order
b_star_dp = np.array([5179/57600, 0, 7571/16695, 393/640,
                      -92097/339200, 187/2100, 1/40])                 # 4th order

c_dp = np.array([0, 1/5, 3/10, 4/5, 8/9, 1.0, 1.0])


# SHO system
def f(t, y):
    return np.array([y[1], -y[0]])

# Initial conditions
y0 = np.array([1.0, 0.0])
t0, t_end = 0.0, 10.0
dt0 = 0.1
tol = 1e-8

rk = odesolvers.AdaptiveERK(A_dp, b_dp, b_star_dp, c_dp, order=4, len_y=len(y0))
ts, ys = rk.integrate(f, t0, y0, t_end, dt0, tol=tol)

# scipy
sol = solve_ivp(f, (t0, t_end), y0, method='DOP853', atol=tol, rtol=tol, dense_output=True)
ts_scipy = ts
ys_scipy = sol.sol(ts_scipy).T

y_diff = ys[:, 0] - ys_scipy[:, 0]

# Exact solution
def y_exact(t): return np.cos(t)
def v_exact(t): return -np.sin(t)
y_true = y_exact(ts)
v_true = v_exact(ts)

plt.plot(ts, ys[:,0] - y_true, label='AdaptiveERK - Exact', color='C3')
plt.plot(ts, ys_scipy[:,0] - y_true, '--', label='SciPy DOP853 - Exact', color='C0')
plt.xlabel('t')
plt.ylabel('Error in y(t)')
plt.legend()
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(ts, ys[:, 0], label="AdaptiveERK")
plt.plot(ts, ys_scipy[:, 0], '--', label="SciPy DOP853")
plt.xlabel("t")
plt.ylabel("y(t)")
plt.legend()
plt.title("Displacement: y(t)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 3))
plt.plot(ts, y_diff, label="y(t) - SciPy", color='C3')
plt.xlabel("t")
plt.ylabel("Error in y")
plt.title("Error Compared to SciPy DOP853")
plt.tight_layout()
plt.show()
