Denoising Diffusion Causal Discovery (DDCD)
===========================================

**DDCD** is a Python package for learning causal structures (Directed Acyclic Graphs) from observational data. 

Unlike traditional constraint-based methods (like PC) or score-based methods (like GES), DDCD leverages the **denoising score matching objective** of diffusion models to smooth the optimization landscape. This allows for faster, more stable convergence, especially in high-dimensional settings[cite: 10, 12].

Key Features
------------

* **Gradient Smoothing:** Uses a denoising diffusion objective to avoid sharp local minima often found in continuous optimization methods like NOTEARS[cite: 36].
* **Adaptive k-hop Constraint:** Replaces expensive matrix exponential calculations with a dynamic acyclicity constraint that transitions from local to global checks, reducing runtime significantly[cite: 11, 37].
* **Scalability:** Implements permutation-invariant batch sampling to handle large datasets efficiently[cite: 38].
* **Versatility:** Includes three specialized models:
    * **DDCD-Linear:** For standard linear Structural Equation Models (SEMs)[cite: 141].
    * **DDCD-Nonlinear:** Uses an autoencoder-style architecture to disentangle nonlinear relationships[cite: 202].
    * **DDCD-Smooth:** A robust variant for real-world data with heterogeneous feature scaling (e.g., biological or clinical data)[cite: 226].

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   theory
   examples
   api