"""AE-GRAAL, aGRAAL and FISTA for monotone variational inequalities."""

import numpy as np


def algorithm2(F, prox_g, measure, x0, x1, initial_step, c, rho,
               sigma, max_iterations=5000, tolerance=1e-9, save_steps=False,
               operator_difference=None):
    """operator_difference, if supplied, is used only for the step-size plot."""
    x_previous, x, x_bar = x0.copy(), x1.copy(), x1.copy()
    F_previous, F_x = F(x_previous), F(x)
    step = float(initial_step)
    history = [[0]+list(measure(x, F_x))+[step]]
    steps = []
    for k in range(1, max_iterations+1):
        if tolerance > 0 and history[-1][1] <= tolerance:
            break
        norm_x = np.linalg.norm(x-x_previous)
        norm_F = np.linalg.norm(F_x-F_previous)
        if norm_F != 0:
            local_step = sigma*norm_x/norm_F
            step = min(rho*step, local_step)
        else:
            step = rho*step
        
        if save_steps and k <= 1000:
            if operator_difference is not None:
                norm_difference = np.linalg.norm(operator_difference(x, x_previous))
                inverse_L = norm_x/norm_difference if norm_x > 0 and norm_difference > 0 else np.nan
            else:
                roundoff = 100*np.finfo(float).eps
                reliable = (norm_x > roundoff*max(1, np.linalg.norm(x), np.linalg.norm(x_previous))
                            and norm_F > roundoff*max(1, np.linalg.norm(F_x), np.linalg.norm(F_previous)))
                inverse_L = norm_x/norm_F if reliable else np.nan
            steps.append([k, step, inverse_L])
        x_bar = ((c-1)*x+x_bar)/c
        x_previous, F_previous = x, F_x
        x = prox_g(x_bar-step*F_x, step)
        F_x = F(x)
        history.append([k]+list(measure(x, F_x))+[step])
    return np.asarray(history), x, steps


def aGRAAL(F, prox_g, measure, x0, x1, initial_step, 
            max_iterations=5000, tolerance=1e-9, phi=1.5, cap=1e6, save_steps=False,
            operator_difference=None):
    """operator_difference, if supplied, is used only for the step-size plot."""
    x_previous, x, x_bar = x0.copy(), x1.copy(), x1.copy()
    F_previous, F_x = F(x_previous), F(x)
    step, theta, rho = float(initial_step), 1, 1/phi+1/phi**2
    history = [[0]+list(measure(x, F_x))+[step]]
    steps = []
    for k in range(1, max_iterations+1):
        if tolerance > 0 and history[-1][1] <= tolerance:
            break
        norm_x = np.linalg.norm(x-x_previous)
        norm_F = np.linalg.norm(F_x-F_previous)
        if norm_F != 0:
            local_step = phi*theta/(4*step)*(norm_x/norm_F)**2
            new_step = min(rho*step, local_step, cap)
        else:
            new_step = min(rho*step, cap)
        theta = phi*new_step/step
        step = new_step

        if save_steps and k <= 1000:
            if operator_difference is not None:
                norm_difference = np.linalg.norm(operator_difference(x, x_previous))
                inverse_L = norm_x/norm_difference if norm_x > 0 and norm_difference > 0 else np.nan
            else:
                roundoff = 100*np.finfo(float).eps
                reliable = (norm_x > roundoff*max(1, np.linalg.norm(x), np.linalg.norm(x_previous))
                            and norm_F > roundoff*max(1, np.linalg.norm(F_x), np.linalg.norm(F_previous)))
                inverse_L = norm_x/norm_F if reliable else np.nan
            steps.append([k, step, inverse_L])
        x_bar = ((phi-1)*x+x_bar)/phi
        x_previous, F_previous = x, F_x
        x = prox_g(x_bar-step*F_x, step)
        F_x = F(x)
        history.append([k]+list(measure(x, F_x))+[step])
    return np.asarray(history), x, steps


def fista(F, prox_g, measure, x1, step, max_iterations=5000, tolerance=1e-9):
    x, y, t = x1.copy(), x1.copy(), 1
    F_y = F(y)
    history = [[0]+list(measure(y, F_y))+[step]]
    for k in range(1, max_iterations+1):
        if tolerance > 0 and history[-1][1] <= tolerance:
            break
        x_new = prox_g(y-step*F_y, step)
        t_new = 0.5*(1+np.sqrt(1+4*t*t))
        y = x_new+(t-1)/t_new*(x_new-x)
        x, t = x_new, t_new
        F_y = F(y)
        history.append([k]+list(measure(y, F_y))+[step])
    return np.asarray(history), x

#Optional implementation (E-GRAAL). This is an extended version of GRAAL, where one can select the convex combination parameter c up to 3.

def E-GRAAL(F, prox_g, measure, x0, step, c=2, max_iterations=5000, tolerance=1e-9):
    """The E-GRAAL. Supply an admissible constant step."""
    if not (1 < c < 3 and 0 < step < np.inf
            and max_iterations >= 1 and 0 <= tolerance < np.inf):
        raise ValueError('Check c, step, max_iterations and tolerance.')

    x, x_bar = x0.copy(), x0.copy()
    F_x = F(x)
    history = [[0, *measure(x, F_x), step]]

    for k in range(1, max_iterations+1):
        if tolerance > 0 and history[-1][1] <= tolerance:
            break

        x_bar = ((c-1)*x+x_bar)/c # The convex combination step. Note that we have provided the theoretical analysis which restricts c to 3 as an upper bound.
        x = prox_g(x_bar-step*F_x, step) # The proximal operator
        F_x = F(x)

        history.append([k, *measure(x, F_x), step])

    return np.asarray(history), x
