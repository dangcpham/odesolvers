import numpy as np
import matplotlib.pyplot as plt
import odesolvers
import numba as nb


# Butcher tableau for RK4
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
    y, v = yv
    return np.array([v, -y])

y0 = np.array([1.0, 0.0])  # y(0) = 1, v(0) = 0
t0 = 0.0
t_end = 10.0
dt = 0.05

rk4 = odesolvers.ERK(A_rk4, b_rk4, c_rk4, len(y0))

ts = [t0]
ys = [y0.copy()]

t = t0
y = y0.copy()
while t < t_end:
    y = rk4.step(f, t, y, dt)
    t += dt
    ts.append(t)
    ys.append(y.copy())

ts = np.array(ts)
ys = np.array(ys)

# Exact
y_exact = np.cos(ts)
v_exact = -np.sin(ts)

plt.plot(ys[:, 0], ys[:, 1], label='RK4 Solution')
plt.plot(y_exact, v_exact, label='Exact Solution', linestyle='--')
plt.xlabel('y')
plt.ylabel('v')
plt.title('Phase Space Plot')
plt.legend()
plt.show()

print(np.mean(np.abs(ys[:, 0] - y_exact)))
print(np.mean(np.abs(ys[:, 1] - v_exact)))

energy = 0.5 * (ys[:, 0]**2 + ys[:, 1]**2)
print(np.abs(energy[-1] - energy[0]))