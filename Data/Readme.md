##LOGISTIC REGRESSION DATA

Place your local LIBSVM files in this folder. The current datasets include:

    mushrooms
    real-sim.bz2
    ijcnn1.bz2
    a9a
    kdda.t.bz2


Keep .bz2 files compressed as the loader reads them directly. For an
already decompressed file, use its actual filename in DATASETS.

The loader expects binary labels and one-based LIBSVM feature indices.
It maps the larger label to +1 and the smaller label to -1. Features
are not rescaled, and no intercept is added. 

Note that the Nash and cubic experiments generate their own inputs and do not
use this folder.
