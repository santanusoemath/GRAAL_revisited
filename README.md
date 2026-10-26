NUMERICAL EXPERIMENTS

This repository contains the Python code to implement the AE-GRAAL and aGRAAL algorithms for three benchmark problems.

Install the dependencies:

    python -m pip install -r requirements.txt

Run one experiment at a time:

    python nash.py
    python cubic_vi.py
    python sparse_log.py


Nash--Cournot: 1,000 firms, three scenarios, ten realisations each.
Each method stops at a residual of 1e-9 or after 50,000 updates. Seven PDFs are saved:
three residual plots, three step-size plots and one combined c-scan.
Step-size plots use up to 1,000 updates of the first realisation.

Cubic complementarity: dimensions 5,000, 10,000 and 50,000, with five
realisations each. Stop at a residual of 1e-12 or after 4,000 updates.

Sparse logistic regression: the selected files are mushrooms and
real-sim.bz2. Put them in data/ as described in data/README.txt.
All four methods run for 5,000 updates. The tolerance determines a
numerical objective reference; it does not stop the methods. If there
is no suitable reference, the objective-gap PDF is omitted. The other
PDF shows step sizes for the first 500 updates.


Malitsky's original aGRAAL code and experiments:
https://github.com/ymalitsky/graal

