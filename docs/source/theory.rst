Methodology
===========

Diffusion Denoising Objective
-----------------------------
Standard continuous optimization methods minimize reconstruction error $||X - XW||^2$. DDCD instead perturbs data with Gaussian noise (forward diffusion) and learns to predict that noise (reverse denoising)[cite: 81].

The objective function for the linear case is:

.. math::

    \min_{W} \frac{1}{2n} ||(X_t - X_tW) - \text{diag}(\sqrt{1-\bar{\alpha}_t})Z(I-W)||_F^2 + \mathcal{R}(W)

Adaptive k-hop Constraint
-------------------------
To ensure the graph is a DAG, DDCD uses a dynamic constraint schedule[cite: 117]:

1.  **Local Alignment:** Checks only small cycles ($k=3$) early in training.
2.  **Structure Refinement:** Increases $k$ to 10 as the matrix sparsifies.
3.  **Global Guarantee:** Enforces full DAG constraint ($k=d$) only at the end.

This avoids the $\mathcal{O}(d^3)$ cost of matrix exponentials for the majority of training[cite: 105].