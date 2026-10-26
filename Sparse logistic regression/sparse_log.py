"""Sparse logistic regression experiments."""

from pathlib import Path
import numpy as np
from scipy.special import expit
from scipy.sparse.linalg import svds
from sklearn.datasets import load_svmlight_file
from threadpoolctl import threadpool_limits
from Algorithms import *
from graph_plots import *

DATASETS = ['mushrooms', 'real-sim.bz2'] # Change or add datasets here.
MAX_ITERATIONS, TOLERANCE, SEED = 5000, 1e-9, 0  # Tolerance is used only to select the objective reference.
STEP_ITERATIONS = 500 # Number of iterations to plot step sizes.
VAR_EPS = 1e-2 # Margin for AE-GRAAL parameters.
ROOT = Path(__file__).resolve().parent
DATA = ROOT/'data'
RESULTS = ROOT/'results'


def reference_value(histories, tolerance):
    eligible = [history[history[:, 1] <= tolerance, 2]
                for history in histories.values()]
    eligible = [values.min() for values in eligible if len(values)]
    if not eligible:
        return None
    reference = min(eligible)
    best_objective = min(history[:, 2].min() for history in histories.values())
    roundoff = 100*np.finfo(float).eps*max(1., abs(reference))
    if best_objective < reference-roundoff:
        print('No reliable objective reference, so increase MAX_ITERATIONS.', flush=True)
        return None
    return reference


def main():
    RESULTS.mkdir(exist_ok=True)
    with threadpool_limits(limits=1):
        for filename in DATASETS:
            print(filename, flush=True)
            A, b = load_svmlight_file(str(DATA/filename), zero_based=False) # Load dataset
            classes = np.unique(b) # Get unique class labels
            if len(classes) != 2 or not np.isfinite(classes).all():
                raise ValueError('Exactly two finite class labels are required.')
            b = np.where(b == classes[-1], 1, -1) #convert to -1/+1 labels
            K = A.multiply(-b[:, None]).tocsr()
            KT = K.T
            gamma = 0.005*np.linalg.norm(A.T@b, np.inf) # Regularisation parameter

            def F(x): # Compute the gradient of the smooth part of the objective
                return np.asarray(KT@expit(K@x)).ravel()

            def prox_g(x, step): # Proximal operator of the objective
                return x+np.clip(-x, -step*gamma, step*gamma)

            def measure(x, F_x): # Compute the residual and objective value
                residual = np.linalg.norm(x-prox_g(x-F_x, 1))
                objective = np.logaddexp(0, K@x).sum()+gamma*np.abs(x).sum()
                return residual, objective

            x1 = np.zeros(A.shape[1]) # initial points
            x0 = np.random.RandomState(SEED).randn(len(x1))
            norm_F = np.linalg.norm(F(x1)-F(x0))
            if norm_F == 0:
                raise ValueError('F(x0) and F(x1) are equal; choose a different x0.')
            initial_step = 0.005*np.linalg.norm(x1-x0)/norm_F

            # Run all algorithms
            history_A, _, _ = algorithm2(F, prox_g, measure, x0, x1, initial_step, c=2, rho=5/4, sigma=(1-VAR_EPS)*5/18,
                                            max_iterations=MAX_ITERATIONS, tolerance=0)
            history_B, _, _ = algorithm2(F, prox_g, measure, x0, x1, initial_step, c=2, rho=7/6, sigma=(1-VAR_EPS)*14/45,
                                            max_iterations=MAX_ITERATIONS, tolerance=0)
            history_G, _, _ = aGRAAL(F, prox_g, measure, x0, x1, initial_step, max_iterations=MAX_ITERATIONS, tolerance=0)
            if min(K.shape) == 1:
                norm_K = float(np.sqrt(K.multiply(K).sum()))
            else:
                norm_K = float(svds(K, k=1, return_singular_vectors=False,
                                   tol=1e-10, random_state=SEED)[0])
            history_F, _ = fista(F, prox_g, measure, x1, 4/norm_K**2, max_iterations=MAX_ITERATIONS, tolerance=0.)
            histories = {'AE-GRAAL-A': history_A, 'AE-GRAAL-B': history_B,
                         'aGRAAL': history_G, 'FISTA': history_F}
            plots = {name: [history] for name, history in histories.items()}
            dataset_name = filename.removesuffix('.bz2').removesuffix('.gz')
            gap_file = RESULTS/f'logistic_{dataset_name}.pdf'
            reference = reference_value(histories, TOLERANCE)
            if reference is not None:
                plot_history(plots, gap_file, column=2, reference=reference,
                             ylabel=r'$J(x^k)-J^*$')
            else:
                gap_file.unlink(missing_ok=True)
                print('No suitable objective reference; increase MAX_ITERATIONS.', flush=True)
            plot_history(plots, RESULTS/f'logistic_{dataset_name}_steps.pdf', column=-1,
                         ylabel=r'Stepsize ($\lambda_k$)', max_iterations=STEP_ITERATIONS)


if __name__ == '__main__':
    main()
