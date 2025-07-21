import numpy as np
import matplotlib.pyplot as plt
import odesolvers
import numba as nb


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

@nb.njit
def f(t, yv):
    y, v = yv
    return np.array([v, -y])

y0 = np.array([1.0, 0.0])
t0 = 0.0
t_end = 10.0
dt = 1e-6

rkdp = odesolvers.AdaptiveERK(A_dp, b_dp, b_star_dp, c_dp, 
                              len_y=len(y0), order=5)

t = t0
y = y0.copy()
ts, ys = rkdp.integrate(f, t0=t, y0=y, t_end=t_end, dt=dt, tol=1e-8)

# Exact
y_exact = np.cos(ts)
v_exact = -np.sin(ts)

plt.plot(ts, ys[:,0] - y_exact, label='y(t) - Exact')
plt.xlabel('t')
plt.ylabel('Error in y(t)')
plt.legend()
plt.show()

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
plt.plot(ts, energy)
plt.show()