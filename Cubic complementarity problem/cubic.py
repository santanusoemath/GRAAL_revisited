"""Cubic complementarity experiments."""

from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from Algorithms import algorithm2, aGRAAL
from graph_plots import *

DIMENSIONS = (5000, 10000, 50000)
VAR_EPS = 1e-2
RUNS = range(5)
C_VALUES = np.arange(11, 30)/10
MAX_ITERATIONS, TOLERANCE = 4000, 1e-12
STEP_ITERATIONS = 1000
RESULTS = Path(__file__).resolve().parent/'results'


def parameters_for_c(c, margin=VAR_EPS):
    if not (1 < c < 3 and 0 < margin < 1):
        raise ValueError('Require 1<c<3 and 0<margin<1.')
    eta = (c-1)/(2*c)
    beta = 1+1/c-eta
    if beta <= 0 or beta*(eta+2/c) <= 1:
        raise ValueError('This eta is not admissible for the selected c.')

    r_plus = beta/c+np.sqrt(beta**2/c**2+beta*eta)
    rho = (1+r_plus)/2
    denominator = 2*(eta+beta/c**2)
    sigma_max = min(beta*eta, beta*eta+2*beta*rho/c-rho**2)/denominator
    sigma = (1-margin)*sigma_max
    if not (1 < rho < r_plus and 0 < sigma < sigma_max and 2*sigma < beta):
        raise ValueError('The parameters are outside the admissible regions.')
    return float(rho), float(sigma)


def H(x):
    return x+x**3+np.roll(x, -1)-np.roll(x, 1)


def F_difference(x, y):
    """Compute F(x)-F(y) without subtracting the two operator values."""
    d = x-y
    return (1+x*x+x*y+y*y)*d+np.roll(d, -1)-np.roll(d, 1)


def main():
    scan_parameters = [parameters_for_c(c) for c in C_VALUES]
    RESULTS.mkdir(exist_ok=True)
    dimension_counts = {}
    with threadpool_limits(limits=1):
        for dimension in DIMENSIONS:
            histories = {'AE-GRAAL-A': [], 'AE-GRAAL-B': [], 'aGRAAL': []}
            counts = np.empty((len(RUNS), len(C_VALUES)))
            for run_index, seed in enumerate(RUNS):
                print(f'Cubic VI, n={dimension}, run {seed}', flush=True)
                rng = np.random.default_rng(seed)
                random_vector = rng.standard_normal(dimension)
                solution = np.maximum(random_vector, 0)
                slack = np.maximum(-random_vector, 0)
                offset = H(solution)-slack

                def F(x):
                    return H(x)-offset

                def prox_g(x, step):
                    return np.maximum(x, 0)

                def measure(x, F_x):
                    return (np.linalg.norm(x-prox_g(x-F_x, 1)),)

                x1 = np.ones(dimension)
                x0 = np.maximum(x1+1e-3*rng.standard_normal(dimension), 0)
                initial_step = np.linalg.norm(x1-x0)/np.linalg.norm(F(x1)-F(x0))

                # Compare the three methods on this realisation.
                history_A, _, steps_A = algorithm2(
                    F, prox_g, measure, x0, x1, initial_step,
                    c=2, rho=5/4, sigma=(1-VAR_EPS)*5/18,
                    max_iterations=MAX_ITERATIONS, tolerance=TOLERANCE,
                    save_steps=run_index == 0, operator_difference=F_difference)
                history_B, _, steps_B = algorithm2(
                    F, prox_g, measure, x0, x1, initial_step,
                    c=2, rho=7/6, sigma=(1-VAR_EPS)*14/45,
                    max_iterations=MAX_ITERATIONS, tolerance=TOLERANCE,
                    save_steps=run_index == 0, operator_difference=F_difference)
                history_G, _, steps_G = aGRAAL(
                    F, prox_g, measure, x0, x1, initial_step,
                    max_iterations=MAX_ITERATIONS, tolerance=TOLERANCE,
                    save_steps=run_index == 0, operator_difference=F_difference)
                for name, history in {'AE-GRAAL-A': history_A,
                                      'AE-GRAAL-B': history_B,
                                      'aGRAAL': history_G}.items():
                    histories[name].append(history)
                    print(f'  {name}: {int(history[-1, 0])} updates, '
                          f'R={history[-1, 1]:.3e}', flush=True)
                if run_index == 0:
                    plot_steps({'AE-GRAAL-A': steps_A, 'AE-GRAAL-B': steps_B,
                                'aGRAAL': steps_G},
                               RESULTS/f'cubic_n{dimension}_steps.pdf', STEP_ITERATIONS)

                # Reuse the same problem and start for every c.
                for c_index, c in enumerate(C_VALUES):
                    rho, sigma = scan_parameters[c_index]
                    history, _, _ = algorithm2(
                        F, prox_g, measure, x0, x1, initial_step,
                        c=c, rho=rho, sigma=sigma,
                        max_iterations=MAX_ITERATIONS, tolerance=TOLERANCE)
                    counts[run_index, c_index] = history[-1, 0]
                    if history[-1, 1] > TOLERANCE:
                        print(f'  c={c:g}: iteration limit reached, '
                              f'R={history[-1, 1]:.3e}', flush=True)
            plot_residual(histories, RESULTS/f'cubic_n{dimension}_residual.pdf')
            dimension_counts[dimension] = counts
    plot_c_scan(dimension_counts, C_VALUES, RESULTS/'cubic_c_scan.pdf',
                label_prefix='n=')


if __name__ == '__main__':
    main()
