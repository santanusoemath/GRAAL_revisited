"""Sparse logistic regression experiments from Section 6.2 of the paper.

Put the LIBSVM files in the data folder.
"""

from pathlib import Path
import numpy as np
from scipy.special import expit
from scipy.sparse.linalg import svds
from sklearn.datasets import load_svmlight_file
from Algorithms import AEGRAAL, aGRAAL, fista
from graph_plots import *


DATASETS = ['mushrooms', 'real-sim.bz2'] # Change or add datasets here.
MAX_ITERATIONS, TOLERANCE, SEED = 5000, 1e-9, 0 # TOLERANCE is used only to select the objective reference (i.e., J_*).
STEP_ITERATIONS = 500 # Number of iterations to plot step sizes.
VAR_EPS = 1e-2
ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
RESULTS = ROOT / 'results'


def reference_value(histories, tolerance):
    accurate = any(np.any(history[:, 1] <= tolerance)
                   for history in histories.values())
    if not accurate:
        return None
    return min(history[:, 2].min() for history in histories.values())


def main():
    RESULTS.mkdir(exist_ok=True)
    found_dataset = False

    for filename in DATASETS:
        path = DATA / filename
        if not path.is_file():
            path = ROOT / filename
        if not path.is_file():
            print(f'Skipping {filename}: put it in {DATA} or {ROOT}.', flush=True)
            continue

        found_dataset = True
        print(f'Running {filename}', flush=True)
        A, b = load_svmlight_file(str(path), zero_based=False) # Load dataset
        classes = np.unique(b)
        if len(classes) != 2 or not np.isfinite(classes).all():
            raise ValueError(f'{filename} must have two finite label values.')
        b = np.where(b == classes[-1], 1, -1)  #convert to -1/+1 labels
        K = A.multiply(-b[:, None]).tocsr()
        gamma = 0.005 * np.linalg.norm(A.T @ b, np.inf) # Regularisation parameter

        def F(x):     # Compute the gradient of the smooth part of the objective
            return K.T @ expit(K @ x)

        def prox_g(x, step):  # Proximal operator of the algorithm's non-smooth part (i.e., the L1 regularisation)
            return np.sign(x) * np.maximum(np.abs(x) - step * gamma, 0)

        def measure(x, F_x): # Compute the residual and objective value
            residual = np.linalg.norm(x - prox_g(x - F_x, 1))
            objective = np.logaddexp(0, K @ x).sum() + gamma * np.abs(x).sum()
            return residual, objective

        x1 = np.zeros(A.shape[1])  # initial points
        x0 = np.random.RandomState(SEED).randn(len(x1))
        change = np.linalg.norm(F(x1) - F(x0))
        if change == 0:
            raise ValueError('F(x0) and F(x1) are equal; choose another start.')
        initial_step = 0.005 * np.linalg.norm(x1 - x0) / change

        # Run all algorithms
        history_A, _, _ = AEGRAAL(F, prox_g, measure, x0, x1, initial_step, c=2, rho=5/4, sigma=(1 - VAR_EPS) * 5/18,
              max_iterations=MAX_ITERATIONS, tolerance=0)
        history_B, _, _ = AEGRAAL(F, prox_g, measure, x0, x1, initial_step, c=2, rho=7/6, sigma=(1 - VAR_EPS) * 14/45,
              max_iterations=MAX_ITERATIONS, tolerance=0)
        history_G, _, _ = aGRAAL(F, prox_g, measure, x0, x1, initial_step, max_iterations=MAX_ITERATIONS, tolerance=0)

        # The Lipschitz constant of the logistic gradient is ||A||_2^2 / 4.
        if min(K.shape) == 1:
            norm_K = np.sqrt(K.multiply(K).sum())
        else:
            norm_K = svds(K, k=1, return_singular_vectors=False,
                          tol=1e-10, random_state=SEED)[0]
        history_F, _ = fista(F, prox_g, measure, x1, 4 / norm_K**2,
                             max_iterations=MAX_ITERATIONS, tolerance=0)

        histories = {'AE-GRAAL-A': history_A, 'AE-GRAAL-B': history_B,
                     'aGRAAL': history_G, 'FISTA': history_F}
        plots = {name: [history] for name, history in histories.items()}
        dataset_name = filename.removesuffix('.bz2').removesuffix('.gz')
        reference = reference_value(histories, TOLERANCE)

        if reference is not None:
            plot_history(plots, RESULTS / f'logistic_{dataset_name}.pdf',
                         column=2, reference=reference, ylabel=r'$J(x^k)-J^*$')
        else:
            print(f'{filename}: no run reached residual {TOLERANCE:g}; '
                  'an objective gap cannot be certified.', flush=True)

        plot_history(plots, RESULTS / f'logistic_{dataset_name}_steps.pdf',
                     column=-1, ylabel=r'Stepsize ($\lambda_k$)',
                     max_iterations=STEP_ITERATIONS)

    if not found_dataset:
        raise FileNotFoundError('No LIBSVM dataset found.')


if __name__ == '__main__':
    main()
