NUMERICAL EXPERIMENTS

This repository contains the Python code to compare AE-GRAAL (Adaptive Extended Golden Ratio Algorithm) against aGRAAL (Adaptive Golden Ratio Algorithm) on three benchmark problems.

Install the dependencies:

    python -m pip install -r requirements.txt

Run each experiment separately:

    python nash.py
    python cubic_vi.py
    python sparse_log.py

Nash–Cournot equilibrium:
The experiments use 1,000 firms, three scenarios and ten realisations per scenario. Each method stops when the natural residual (see the paper) reaches 1e-9 or after 50,000 updates. The script saves seven PDFs: three residual plots, three stepsize plots and one combined comparison across c values. Stepsize plots show the first realisation (seed 0), up to 1,000 updates.

Cubic complementarity:
The experiments use dimensions 5,000, 10,000 and 50,000, with five realisations per dimension. Each method stops when the natural residual reaches 1e-12 or after 4,000 updates.

Sparse logistic regression:
The selected datasets are mushrooms and real-sim.bz2. Place them in data/ following data/README.txt. All four methods perform 5,000 iterations on mushrooms and real-sim.bz2, and 1,800 iterations on ijcnn1. The residual tolerance is used only to select a numerical reference for the optimal objective value. If no suitable reference is found, try changing the number of iterations. A separate plot shows the first 500 stepsizes.

All figures are saved as separate PDFs in results/.

Malitsky’s original aGRAAL implementation and experiments:
https://github.com/ymalitsky/graal
