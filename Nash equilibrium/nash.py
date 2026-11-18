r"""Nash--Cournot experiments.

Firm $i$ chooses its output $q_i$, while the market price depends on the
total output $Q=\sum_i q_i$. We solve the resulting variational inequality
on $C=[eps,\infty)^n$, eps>0. The positive lower bound ensures that the
inverse demand is defined and $F$ is locally Lipschitz on $C$.

For three cost scenarios, we compare AE-GRAAL-A, AE-GRAAL-B (the two instances of AE-GRAAL) 
and aGRAAL using the natural residual $R_k$ and the step sizes $\lambda_k$. We also plot 
the iteration counts of AE-GRAAL for $c=1.1,1.2,\ldots,2.9$.
"""

from pathlib import Path
import numpy as np
from Algorithms import AEGRAAL, aGRAAL
from graph_plots import *


SCENARIOS, RUNS = (1, 2, 3), range(10)
N_FIRMS, MAX_ITERATIONS, TOLERANCE = 1000, 50000, 1e-9
MIN_OUTPUT = 1e-3
C_VALUES = np.arange(11, 30) / 10
VAR_EPS = 1e-2
STEP_ITERATIONS = 1000
RESULTS = Path(__file__).resolve().parent / 'results'


def parameters_for_c(c, margin=VAR_EPS):
    """Choose the admissible rho and sigma for the Nash c scan."""
    if not (1 < c < 3 and 0 < margin < 1):
        raise ValueError('Require 1 < c < 3 and 0 < margin < 1.')
    eta = (c - 1)/ (4*c) # eta the auxiliary parameter
    beta = 1 + 1/c - eta  # beta depends on c and eta
    r_plus = beta/c + np.sqrt((beta/c)**2 + beta * eta) # This parameter is used to choose rho in (1,r_plus)
    rho = (1 + r_plus) / 2 # Choose rho in the middle of the admissible range
    sigma_max = min(beta * eta, beta * eta + 2*beta*rho/c - rho**2) # Coefficients of the inverse of the local Lipschitz constant
    sigma_max /= 2 * (eta + beta/c**2) # The admissible range of sigma is (0,sigma_max)
    sigma = (1 - margin) * sigma_max 
    if not (1 < rho < r_plus and 0 < sigma < sigma_max and 2*sigma < beta):
        raise ValueError('The chosen rho or sigma is not admissible.')
    return rho, sigma


def main():
    RESULTS.mkdir(exist_ok=True)
    scenario_counts = {}

    for scenario in SCENARIOS:
        histories = {'AE-GRAAL-A': [], 'AE-GRAAL-B': [], 'aGRAAL': []}
        counts = np.empty((len(RUNS), len(C_VALUES)))

        for run_index, seed in enumerate(RUNS):
            print(f'Nash scenario {scenario}, realisation {run_index + 1}', flush=True)
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
            price_scale = 5000**(1 / gamma)

            def F(q):
                total = q.sum()
                price = price_scale * total**(-1 / gamma)
                return a + (ell * q)**(1/beta) - price + q * price / (gamma * total)

            def prox_g(v, step):
                return np.maximum(v, MIN_OUTPUT)

            def measure(q, F_q):
                return (np.linalg.norm(q - prox_g(q - F_q, 1)),)

            x1 = np.ones(N_FIRMS)
            rng.randn(N_FIRMS)  
            noise = rng.randn(N_FIRMS)
            x0 = prox_g(x1 + 1e-9 * noise, 1)
            norm_change = np.linalg.norm(F(x1) - F(x0))
            if norm_change == 0:
                raise ValueError('F(x0) and F(x1) are equal.')
            initial_step = np.linalg.norm(x1 - x0) / norm_change

            # Run all algorithms
            history_A, _, steps_A = AEGRAAL(F, prox_g, measure, x0, x1, initial_step, c=2, rho=5/4, sigma=(1 - VAR_EPS) * 5/18,
                          max_iterations=MAX_ITERATIONS, tolerance=TOLERANCE, save_steps=run_index == 0)
            history_B, _, steps_B = AEGRAAL(F, prox_g, measure, x0, x1, initial_step, c=2, rho=7/6, sigma=(1 - VAR_EPS) * 14/45,
                          max_iterations=MAX_ITERATIONS, tolerance=TOLERANCE, save_steps=run_index == 0)
            history_G, _, steps_G = aGRAAL(F, prox_g, measure, x0, x1, initial_step, max_iterations=MAX_ITERATIONS, tolerance=TOLERANCE, 
                          save_steps=run_index == 0)
            
            histories['AE-GRAAL-A'].append(history_A)
            histories['AE-GRAAL-B'].append(history_B)
            histories['aGRAAL'].append(history_G)

            if run_index == 0:
                plot_steps({'AE-GRAAL-A': steps_A, 'AE-GRAAL-B': steps_B,
                            'aGRAAL': steps_G},
                           RESULTS / f'nash_s{scenario}_steps.pdf', STEP_ITERATIONS)

            for c_index, c in enumerate(C_VALUES):
                rho, sigma = parameters_for_c(c)
                history, _, _ = AEGRAAL(
                    F, prox_g, measure, x0, x1, initial_step,
                    c=c, rho=rho, sigma=sigma,
                    max_iterations=MAX_ITERATIONS, tolerance=TOLERANCE)
                counts[run_index, c_index] = history[-1, 0]
                if history[-1, 1] > TOLERANCE:
                    print(f'  c={c:g}: iteration limit reached, R={history[-1, 1]:.3e}',
                          flush=True)

        plot_residual(histories, RESULTS / f'nash_s{scenario}.pdf')
        scenario_counts[scenario] = counts

    plot_c_scan(scenario_counts, C_VALUES, RESULTS / 'nash_c_scan.pdf')


if __name__ == '__main__':
    main()
