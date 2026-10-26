r"""Nash--Cournot experiments on the positive-output feasible set.

Firm $i$ chooses its output $q_i$, while the market price depends on the
total output $Q=\sum_i q_i$. We solve the resulting variational inequality
on $C=[10^{-3},\infty)^n$. The positive lower bound ensures that the
inverse demand is defined and $F$ is locally Lipschitz on $C$.

For three cost scenarios, we compare AE-GRAAL-A, AE-GRAAL-B and aGRAAL
using the natural residual $R_k$ and the step sizes $\lambda_k$. We also
plot the iteration counts of AE-GRAAL for $c=1.1,1.2,\ldots,2.9$.
"""

from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from Algorithms import algorithm2, aGRAAL
from graph_plots import *

SCENARIOS, RUNS = (1, 2, 3), range(10)
N_FIRMS, MAX_ITERATIONS, TOLERANCE = 1000, 50000, 1e-9
MIN_OUTPUT = 1e-3
C_VALUES = np.arange(11, 30)/10
VAR_EPS = 1e-2
STEP_ITERATIONS = 1000
RESULTS = Path(__file__).resolve().parent/'results'


def parameters_for_c(c, margin=VAR_EPS):
    if not (1 < c < 3 and 0 < margin < 1):
        raise ValueError('Require 1<c<3 and 0<margin<1.')
    eta = (c-1)/(4*c)
    beta = 1+1/c-eta
    if beta <= 0 or beta*(eta+2/c) <= 1: # Checking the admissibility of eta for the selected c
        raise ValueError('This eta is not admissible for the selected c.')

    r_plus = beta/c+np.sqrt(beta**2/c**2+beta*eta)
    rho = (1+r_plus)/2
    denominator = 2*(eta+beta/c**2)
    sigma_max = min(beta*eta, beta*eta+2*beta*rho/c-rho**2)/denominator
    sigma = (1-margin)*sigma_max
    if not (1 < rho < r_plus and 0 < sigma < sigma_max and 2*sigma < beta): # Checking the admissibility of rho and sigma for the selected c
        raise ValueError('The parameters are outside the admissible regions.')
    return float(rho), float(sigma)


def main():
    RESULTS.mkdir(exist_ok=True)
    scenario_counts = {}
    with threadpool_limits(limits=1):
        for scenario in SCENARIOS:
            histories = {'AE-GRAAL-A': [], 'AE-GRAAL-B': [], 'aGRAAL': []}
            counts = np.empty((len(RUNS), len(C_VALUES)))
            for run_index, seed in enumerate(RUNS):
                print(f'Nash scenario {scenario}, realisation {run_index+1}', flush=True)
                rng = np.random.RandomState(seed)
                a = rng.uniform(1, 100, N_FIRMS)
                ell = rng.uniform(0.5, 5, N_FIRMS)
                gamma = 1.1 if scenario == 1 else 1.5
                if scenario == 1:
                    beta = rng.uniform(0.5, 2, N_FIRMS)
                elif scenario == 2:
                    beta = rng.uniform(0.3, 4, N_FIRMS)
                else:
                    beta = np.full(N_FIRMS, 0.5)
                inverse_beta = 1/beta
                price_scale = 5000**(1/gamma)

                def F(q):
                    total_output = q.sum()
                    price = price_scale*total_output**(-1/gamma)
                    return a+(ell*q)**inverse_beta-price+q*price/(gamma*total_output)

                def prox_g(v, step):
                    return np.maximum(v, MIN_OUTPUT)

                def measure(q, F_q):
                    return (np.linalg.norm(q-prox_g(q-F_q, 1)),)

                x1 = np.ones(N_FIRMS)

                # Keep this draw to reproduce the original realisations.
                rng.randn(N_FIRMS)
                noise = rng.randn(N_FIRMS)
                x0 = prox_g(x1+1e-9*noise, 1)
                norm_x = np.linalg.norm(x1-x0)
                norm_F = np.linalg.norm(F(x1)-F(x0))
                if norm_F == 0:
                    raise ValueError('F(x0) and F(x1) are equal; choose a different x0.')
                initial_step = norm_x/norm_F
                
                # Run all algorithms.
                history_A, _, steps_A = algorithm2(F, prox_g, measure, x0, x1, initial_step, c=2, rho=5/4, sigma=(1-VAR_EPS)*5/18,
                                                   max_iterations=MAX_ITERATIONS, tolerance=TOLERANCE, save_steps=run_index == 0)
                history_B, _, steps_B = algorithm2(F, prox_g, measure, x0, x1, initial_step, c=2, rho=7/6, sigma=(1-VAR_EPS)*14/45,
                                                   max_iterations=MAX_ITERATIONS, tolerance=TOLERANCE, save_steps=run_index == 0)
                history_G, _, steps_G = aGRAAL(F, prox_g, measure, x0, x1, initial_step,
                    max_iterations=MAX_ITERATIONS, tolerance=TOLERANCE, save_steps=run_index == 0)
                histories['AE-GRAAL-A'].append(history_A)
                histories['AE-GRAAL-B'].append(history_B)
                histories['aGRAAL'].append(history_G)
                if run_index == 0:
                    plot_steps({'AE-GRAAL-A': steps_A, 'AE-GRAAL-B': steps_B,
                                'aGRAAL': steps_G}, RESULTS/f'nash_s{scenario}_steps.pdf',
                               STEP_ITERATIONS)
                for c_index, c in enumerate(C_VALUES):
                    rho, sigma = parameters_for_c(c)
                    history, _, _ = algorithm2(
                        F, prox_g, measure, x0, x1, initial_step,
                        max_iterations=MAX_ITERATIONS, tolerance=TOLERANCE, c=c, rho=rho, sigma=sigma)
                    counts[run_index, c_index] = history[-1, 0]
                    if history[-1, 1] > TOLERANCE:
                        print(f'  c={c:g}: iteration limit reached, R={history[-1, 1]:.3e}', flush=True)
            plot_residual(histories, RESULTS/f'nash_s{scenario}.pdf')
            scenario_counts[scenario] = counts
    plot_c_scan(scenario_counts, C_VALUES, RESULTS/'nash_c_scan.pdf')


if __name__ == '__main__':
    main()
