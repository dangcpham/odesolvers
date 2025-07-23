import numpy as np
import matplotlib.pyplot as plt
import odesolvers
import numba as nb

# butcher tableau for RK4
A_rk4 = np.array([
    [0,   0,   0, 0],
    [0.5, 0,   0, 0],
    [0,   0.5, 0, 0],
    [0,   0,   1, 0]
])
b_rk4 = np.array([1/6, 1/3, 1/3, 1/6])
c_rk4 = np.array([0, 0.5, 0.5, 1.0])

@nb.njit
def f(t, yv):
    return np.array([yv[1], -yv[0]])

y0 = np.array([1.0, 0.0])  # y(0) = 1, v(0) = 0
t0 = 0.0
t_end = 10.0
dt = 1e-2

rk4 = odesolvers.FixedERK(A_rk4, b_rk4, c_rk4, t0=t0)

ts = [t0]
ys = [y0.copy()]

t = t0
y = y0.copy()
while t < t_end:
    t, y = rk4.step(f, y, dt=dt)
    ts.append(t)
    ys.append(y.copy())

ts = np.array(ts)
ys = np.array(ys)

# exact solution
y_exact = np.cos(ts)
v_exact = -np.sin(ts)
energy = 0.5 * (ys[:, 0]**2 + ys[:, 1]**2)
y_err = np.mean(np.abs(ys[:, 0] - y_exact))
v_err = np.mean(np.abs(ys[:, 1] - v_exact))
E_err = np.abs(energy[-1] - energy[0])

print(f"Average error in y: {y_err:.2E}")
print(f"Average error in v: {v_err:.2E}")
print(f"Energy conservation error: {E_err:.2E}")

assert y_err < 1e-8, "Error in y exceeds tolerance"
assert v_err < 1e-8, "Error in v exceeds tolerance"
assert E_err < 1e-8, "Energy conservation error exceeds tolerance"