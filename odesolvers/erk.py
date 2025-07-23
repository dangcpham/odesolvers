import numpy as np
import numba as nb

# I. Simple Explicit Runge-Kutta (ERK) method implementation with fixed
#    timestep, given a Butcher tableau.

@nb.njit()
def ERK_step(f, t, y, dt, A, b, c, s, k, _yi):
    # runge-kutta step
    for i in range(s):
        ti = t + c[i] * dt
        _yi[:] = y
        for j in range(i): _yi += dt * A[i, j] * k[j]
        k[i] = f(ti, _yi)

    # solution
    y_new = y + dt * np.dot(b, k)

    return t+dt, y_new

@nb.njit()
def ERK_integrate(f, t, y, t_end, dt, A, b, c, s, k, _yi):
    while t < t_end:
        # avoid overshooting the end time
        if t + dt > t_end: dt = t_end - t

        # returns t, y
        t, y = ERK_step(f, t, y, dt, A, b, c, s, k, _yi)

    return t, y

class FixedERK:
    def __init__(self, A, b, c, t0=0.0):
        self.A = A
        self.b = b
        self.c = c

        self.verify_b()

        self.s = len(b)
        self.k = None
        self.y = None
        self._yi = None
        self.t = t0

    def verify_b(self):
        if not np.isclose(np.sum(self.b), 1.0):
            raise ValueError("The sum of the b coefficients must be 1.")

    def step(self, f, y, dt):
        if self.k is None:
            len_y = len(y)
            self.k = np.zeros((self.s, len_y), dtype=np.float64)
            self._yi = np.empty_like(y, dtype=np.float64)

        self.t, self.y = ERK_step(f, self.t, y, dt, 
                                  self.A, self.b, self.c, 
                                  self.s, self.k, self._yi)
        self.dt = dt

        return self.t, self.y

    def integrate(self, f, y, t_end, dt):
        if self.k is None:
            len_y = len(y)
            self.k = np.zeros((self.s, len_y), dtype=np.float64)
            self._yi = np.empty_like(y, dtype=np.float64)

        self.t, self.y = ERK_integrate(f, self.t, y, t_end, dt, 
                               self.A, self.b, self.c, self.s, self.k, self._yi)
        self.dt = dt

        return self.t, self.y


# II. Explicit ERK with adaptive timestep by having an embedded method.
#     Here, a lower order method is used to estimate the error which allows
#     for adaptive timestep.
#     c.f. https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods#Adaptive_Runge%E2%80%93Kutta_methods
#          https://fncbook.github.io/v1.0/ivp/adaptive.html

@nb.njit(cache=True)
def ERK_step_embedded(f, t, y, dt,
                      A, b, b_star, c, s, k, 
                      atol, rtol, expo, Nmax, _yi):
    for _ in range(Nmax):
        # runge-kutta step
        for i in range(s):
            _yi[:] = y
            for j in range(i): _yi += dt * A[i, j] * k[j]
            k[i] = f(t + c[i]*dt, _yi)

        # solution for the main and embedded methods
        y_new      = y + dt * np.dot(b, k)
        y_embedded = y + dt * np.dot(b_star, k)

        # error estimate and new dt
        # err = np.linalg.norm(y_new - y_embedded)
        tol = atol + rtol * np.maximum(np.abs(y), np.abs(y_new))
        y_diff_scaled = (y_new - y_embedded) / tol
        err_sqr_sum = 0.0
        for d in y_diff_scaled: err_sqr_sum += d * d
        # taking the mean of the squared error
        err2 = (err_sqr_sum / y.shape[0])

        if err2 != 0.: 
            # expo = -(1/2) / order (factor of 2 from sqrt of mean error^2)
            dt_next = 0.999 * dt * (err2 ** expo)
        else:
            dt_next = dt

        # solution accepted
        if err2 < 1.: 
            return t + dt, y_new, err2, dt_next

        # solution rejected, retry with new dt
        else: 
            dt = dt_next

    print("t=", t, "y=", y, "dt=", dt)
    raise RuntimeError("Failed to converge within max iterations, Nmax=", Nmax)

@nb.njit(cache=True)
def ERK_integrate_adaptive(f, t, y, t_end, dt, A, b, b_star, c, s, k, expo,
                           atol, rtol, Nmax, _yi, 
                           pre_integration_func, post_integration_func):

    while t < t_end:
        t, y = pre_integration_func(t, y)

        # avoid overshooting the end time
        if t + dt > t_end: dt = t_end - t

        # returns t, y, error, new dt
        t, y, _, dt = ERK_step_embedded(f, t, y, dt, 
                                           A, b, b_star, c, s, k, 
                                           atol=atol, rtol=rtol,
                                           expo=expo, Nmax=Nmax,
                                           _yi=_yi)

        t, y = post_integration_func(t, y)

    return t, y, dt

@nb.njit(cache=True)
def default_nothing(t, y):
    return t, y

class AdaptiveERK:
    def __init__(self, A, b, b_star, c, order, atol=1e-6, rtol=1e-3,
                 Nmax=100, t0=0.0):
        self.A = A
        self.b = b            # weights for the main method
        self.b_star = b_star  # weights for the embedded method
        self.c = c
        self.order = order    # order of the main method

        self.s = len(b)

        self.atol = atol
        self.rtol = rtol
        self.t = t0

        self.k = None
        self.y = None
        self.dt = 1e-3
        self.Nmax = Nmax

        # temp storage for intermediate results
        self._yi = None # yi = np.empty_like(y)
        self.expo = -0.5 / order  # exponent for error control

    def step(self, f, y):
        if self.k is None:
            len_y = len(y)
            self.k = np.zeros((self.s, len_y))
            self.y = np.empty((len_y,))
            self._yi = np.empty_like(y)

        self.t, self.y, _, self.dt = ERK_step_embedded(f, 
                                    self.t, y, self.dt, 
                                    self.A, self.b, self.b_star, self.c, 
                                    self.s, self.k, self.atol, self.rtol,
                                    self.expo,
                                    self.Nmax, self._yi)

        return self.t, self.y

    def integrate(self, f, y, t_end,
                  pre_integration_func=default_nothing, 
                  post_integration_func=default_nothing):

        if self.k is None:
            len_y = len(y)
            self.k = np.zeros((self.s, len_y))
            self.y = np.empty((len_y,))
            self._yi = np.empty_like(y)

        self.t, self.y, self.dt = ERK_integrate_adaptive(f, 
                                      self.t, y, t_end, self.dt, 
                                      self.A, self.b, self.b_star, 
                                      self.c, self.s, self.k, 
                                      self.expo, self.atol, self.rtol,
                                      self.Nmax, self._yi,
                                    pre_integration_func=pre_integration_func,
                                    post_integration_func=post_integration_func)

        return self.t, self.y

