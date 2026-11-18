"""The four algorithms used in the experiments."""

import numpy as np

def _inverse_local_L(x, old_x, F_x, old_F, operator_difference):
    """Return 1/L_k for a step-size plot."""
    distance = np.linalg.norm(x - old_x)
    if operator_difference is None:
        norm_change = np.linalg.norm(F_x - old_F)
        if (distance <= 1e-14 * max(1, np.linalg.norm(x), np.linalg.norm(old_x))
                or norm_change <= 1e-14 * max(1, np.linalg.norm(F_x), np.linalg.norm(old_F))):
            return np.nan
    else:
        norm_change = np.linalg.norm(operator_difference(x, old_x))
    return distance / norm_change if distance > 0 and norm_change > 0 else np.nan


def AEGRAAL(F, prox_g, measure, x0, x1, initial_step, c, rho, sigma, max_iterations=5000, tolerance=1e-9, save_steps=False,
               operator_difference=None): 
    """AE-GRAAL: lambda_k = min(rho*lambda_(k-1), sigma/L_k), where L_k = ||F(x^k)-F(x^(k-1))|| / ||x^k-x^(k-1)||."""
    old_x, x, x_bar = x0.copy(), x1.copy(), x1.copy()
    old_F, F_x = F(old_x), F(x)
    step = initial_step
    history = [[0, *measure(x, F_x), step]] # meaure represents the residual such as natural ||x-prox(x- F(x))|| or objective gap (F(x^k)-F^*)
    steps = []

    for k in range(1, max_iterations + 1):
        if tolerance > 0 and history[-1][1] <= tolerance:
            break
        distance = np.linalg.norm(x - old_x)
        change = np.linalg.norm(F_x - old_F)
        new_step = rho * step
        if change != 0:
            new_step = min(new_step, sigma * distance / change)
        step = new_step

        if save_steps and k <= 1000:
            inverse_L = _inverse_local_L(x, old_x, F_x, old_F, operator_difference)
            steps.append([k, step, inverse_L])

        x_bar = ((c - 1) * x + x_bar) / c # convex combination of x^k and x_bar^(k-1)
        old_x, old_F = x, F_x
        x = prox_g(x_bar - step * F_x, step) # proximal map of g in the algorithm
        F_x = F(x)
        history.append([k, *measure(x, F_x), step])

    return np.asarray(history), x, steps


def aGRAAL(F, prox_g, measure, x0, x1, initial_step, max_iterations=5000, tolerance=1e-9, phi=1.5, cap=1e6,
           save_steps=False, operator_difference=None):
    old_x, x, x_bar = x0.copy(), x1.copy(), x1.copy()
    old_F, F_x = F(old_x), F(x)
    step, theta = initial_step, 1
    rho = 1 / phi + 1 / phi**2
    history = [[0, *measure(x, F_x), step]]
    steps = []

    for k in range(1, max_iterations + 1):
        if tolerance > 0 and history[-1][1] <= tolerance:
            break

        distance = np.linalg.norm(x - old_x)
        change = np.linalg.norm(F_x - old_F)
        new_step = min(rho * step, cap)
        if change != 0:
            new_step = min(new_step, phi * theta / (4*step) * (distance / change)**2)
        theta = phi * new_step / step
        step = new_step

        if save_steps and k <= 1000:
            inverse_L = _inverse_local_L(x, old_x, F_x, old_F, operator_difference)
            steps.append([k, step, inverse_L])

        x_bar = ((phi - 1) * x + x_bar) / phi
        old_x, old_F = x, F_x
        x = prox_g(x_bar - step * F_x, step)
        F_x = F(x)
        history.append([k, *measure(x, F_x), step])

    return np.asarray(history), x, steps


def fista(F, prox_g, measure, x1, step, max_iterations=5000,
          tolerance=1e-9):
    """FISTA with a constant step size."""
    x, y, t = x1.copy(), x1.copy(), 1
    history = [[0, *measure(x, F(x)), step]]

    for k in range(1, max_iterations + 1):
        if tolerance > 0 and history[-1][1] <= tolerance:
            break

        new_x = prox_g(y - step * F(y), step)
        new_t = (1 + np.sqrt(1 + 4 * t**2)) / 2
        y = new_x + (t - 1) / new_t * (new_x - x)
        x, t = new_x, new_t
        history.append([k, *measure(x, F(x)), step])

    return np.asarray(history), x


# Optional implementation for the fixed step-szie case (E-GRAAL). This is an extended version of GRAAL, 
# where one can select the convex combination parameter c up to 3.

def EGRAAL(F, prox_g, measure, x0, step, c=2,
           max_iterations=5000, tolerance=1e-9):
    """Extended GRAAL with an admissible (see the paper for details) constant step size."""
    if not (1 < c < 3 and  0 <= tolerance < np.inf):
        raise ValueError('Check c and tolerance.')

    x, x_bar = x0.copy(), x0.copy()
    F_x = F(x)
    history = [[0, *measure(x, F_x), step]]

    for k in range(1, max_iterations + 1):
        if tolerance > 0 and history[-1][1] <= tolerance:
            break
        x_bar = ((c - 1) * x + x_bar) / c
        x = prox_g(x_bar - step * F_x, step)
        F_x = F(x)
        history.append([k, *measure(x, F_x), step])

    return np.asarray(history), x
