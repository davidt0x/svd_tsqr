# %
# import numpy as np
from __future__ import annotations

import functools

import cupy as np

m = 300_000
n = int(m / 3000)

A = np.random.random((m, n))

# Get the ground truth QR factorization

Q_gt, R_gt = np.linalg.qr(A)


def unique_R(R):
    """
    QR factorization is only unique up to the signs of the rows of R. Lets enforce positive diagonals of R to get
    a unique factorization. See:

        https://www.mathworks.com/matlabcentral/answers/83798-sign-differences-in-qr-decomposition

    """
    D = np.diag(np.sign(np.diag(R)))
    return np.dot(D, R)


R_gt = unique_R(R_gt)

# %%

# Split A into chunks
num_chunks = 100
A_chunks = np.vsplit(A, num_chunks)

# Get the R from the QR of each chunk
R_chunks = [np.linalg.qr(A_local)[1] for A_local in A_chunks]

# All reduce the list of chunks by stacking Rs and taking QR
R = functools.reduce(lambda R_a, R_b: np.linalg.qr(np.vstack((R_a, R_b)))[1], R_chunks)

# Check that we get the same as the original
np.allclose(unique_R(R), R_gt)
