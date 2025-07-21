import numpy as np
import numba as nb

# I. Simple Explicit Runge-Kutta (ERK) method implementation with fixed
#    timestep, given a Butcher tableau.

@nb.njit()
def ERK_step(f, t, y, dt, A, b, c, s, k):
        # Reset k for the new step
        k[:] = 0.0

        for i in range(s):
            ti = t + c[i] * dt
            yi = y.copy()
            for j in range(i): yi += dt * A[i, j] * k[j]
            k[i] = f(ti, yi)

        y_new = y + dt * np.dot(b, k)
        return y_new

@nb.njit()
def ERK_integrate(f, t0, y0, t_end, dt, A, b, c, s, k):
    t = t0
    y = y0.copy()
    ys = [y.copy()]
    ts = [t]

    while t < t_end:
        y = ERK_step(f, t, y, dt, A, b, c, s, k)
        ys.append(y.copy())
        ts.append(t)
        t += dt

    return np.array(ts), np.array(ys)

class ERK:
    def __init__(self, A, b, c, len_y):
        self.A = A
        self.b = b
        self.c = c

        self.verify_b()

        self.s = len(b)
        self.k = np.zeros((len(b), len_y), dtype=np.float64)


    def verify_b(self):
        if not np.isclose(np.sum(self.b), 1.0):
            raise ValueError("The sum of the b coefficients must be 1.")

    def step(self, f, t, y, dt):
        return ERK_step(f, t, y, dt, 
                          self.A, self.b, self.c, self.s, self.k)

    def integrate(self, f, t0, y0, t_end, dt):
        return ERK_integrate(f, t0, y0, t_end, dt, 
                               self.A, self.b, self.c, self.s, self.k)


# II. Explicit ERK with adaptive timestep by having an embedded method.
#     Here, a lower order method is used to estimate the error which allows
#     for adaptive timestep.
#     c.f. https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods#Adaptive_Runge%E2%80%93Kutta_methods
#          https://fncbook.github.io/v1.0/ivp/adaptive.html

@nb.njit()
def ERK_next_dt(dt, err, tol, order):
    # adaptive step size control
    if err != 0.: dt *= (tol / err) ** (1.0 / (order + 1))
    return dt

def ERK_step_embedded(f, t, y, dt, 
                      A, b, b_star, c, s, k, tol, order):
    # Reset k for the new step
    k[:] = 0.0

    for i in range(s):
        ti = t + c[i] * dt
        yi = y.copy()
        for j in range(i): yi += dt * A[i, j] * k[j]
        k[i] = f(ti, yi)

    y_new = y + dt * np.dot(b, k)
    y_embedded = y + dt * np.dot(b_star, k)
    err = np.linalg.norm(y_new - y_embedded)
    dt_next = ERK_next_dt(dt, err, tol=tol, order=order)

    return y_new, err, dt_next

def ERK_integrate_adaptive(f, t0, y0, t_end, dt, A, b, b_star, c, s, k, order,
                           tol=1e-6):
    t = t0
    y = y0.copy()

    ts = [t]
    ys = [y.copy()]

    while t < t_end:
        if t + dt > t_end: dt = t_end - t

        y_new, err, dt_new = ERK_step_embedded(f, t, y, dt, 
                                           A, b, b_star, c, s, k, tol, order)

        # solution accepted
        if err < tol:
            t += dt
            y = y_new
            ts.append(t)
            ys.append(y.copy())
            dt = dt_new
        # solution rejected
        else:
            # if the new dt is larger than the current dt here,
            # something is very wrong
            if dt_new > dt:
                raise ValueError("Rejected because error is too high, but new dt is larger than current dt.")
            # accept the new dt if it is smaller
            dt = dt_new

    return np.array(ts), np.array(ys)

class AdaptiveERK:
    def __init__(self, A, b, b_star, c, order, len_y):
        self.A = A
        self.b = b            # weights for the main method
        self.b_star = b_star  # weights for the embedded method
        self.c = c
        self.order = order
        self.s = len(b)
        self.k = np.zeros((self.s, len_y))  # defer length if needed
    
    def step(self, f, t, y, dt):
        return ERK_step_embedded(f, t, y, dt, 
                                  self.A, self.b, self.b_star, self.c, 
                                  self.s, self.k)
    
    def integrate(self, f, t0, y0, t_end, dt, tol=1e-6):
        return ERK_integrate_adaptive(f, t0, y0, t_end, dt, 
                                      self.A, self.b, self.b_star, 
                                      self.c, self.s, self.k, 
                                      self.order, tol)

