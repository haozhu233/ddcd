Examples and Tutorials
======================

This section demonstrates how to use DDCD for Linear, Nonlinear, and Real-world scenarios.

.. note::
   These examples use `gcastle` for synthetic data generation, as suggested in the project README.

1. Linear Causal Discovery
--------------------------
Use ``DDCD_Linear_Trainer`` for standard linear Structural Equation Models. This model matches or exceeds the performance of state-of-the-art algorithms like GOLEM and DAGMA while being significantly faster[cite: 418].

.. code-block:: python

    import ddcd
    import torch
    import numpy as np
    from castle.datasets import IIDSimulation, DAG

    # 1. Generate Synthetic Linear Data (Scale-Free Graph)
    # 100 nodes, 1000 edges, Gaussian noise
    dag_adj = DAG.scale_free(n_nodes=100, n_edges=1000, weight_range=(0.5, 1.5), seed=42)
    X = IIDSimulation(W=dag_adj, n=2000, method='linear', sem_type='gauss', noise_scale=1).X

    # 2. Initialize the Trainer
    # Uses the adaptive k-hop schedule automatically during training
    model = ddcd.DDCD_Linear_Trainer(
        X, 
        device='cuda' if torch.cuda.is_available() else 'cpu',
        batch_size=128,
        lr=0.001
    )

    # 3. Train the model
    # The paper suggests 5000 steps for convergence on linear benchmarks
    model.train(n_steps=5000)

    # 4. Extract the Adjacency Matrix
    # Thresholding is standard practice to remove noise
    w_est = model.get_adj()
    adjacency_matrix = (np.abs(w_est) > 0.3).astype(int)

    print(f"Recovered {adjacency_matrix.sum()} edges.")

2. Nonlinear Causal Discovery
-----------------------------
For data with nonlinear dependencies, use ``DDCD_NonLinear_Trainer``. This model learns a latent representation to recover the structural equations[cite: 202].

.. code-block:: python

    import ddcd
    from castle.datasets import IIDSimulation, DAG

    # 1. Generate Nonlinear Data
    # Using a nonlinear mechanism (e.g., MLP or Cosine)
    dag_adj = DAG.erdos_renyi(n_nodes=20, n_edges=40, seed=42)
    X = IIDSimulation(W=dag_adj, n=2000, method='nonlinear', sem_type='mlp').X

    # 2. Initialize Nonlinear Trainer
    # Adjust hidden_dims based on complexity of the data
    model = ddcd.DDCD_NonLinear_Trainer(
        X, 
        hidden_dims=[32, 32], 
        device='cuda',
        dag_scaling_factor=3.0
    )

    # 3. Train
    # Nonlinear models typically converge faster (e.g., 1000 steps)
    model.train(n_steps=1000)

    w_est = model.get_adj()

3. Robust Discovery for Real-World Data (DDCD-Smooth)
-----------------------------------------------------
Real-world data often violates the scale-invariance assumptions of standard SEMs. ``DDCD_Smooth`` learns a normalized adjacency matrix, making it ideal for datasets like gene expression or clinical records[cite: 226, 230].

.. code-block:: python

    import ddcd
    import pandas as pd
    
    # 1. Load Real-World Data
    # Example: Clinical data with varying feature scales
    df = pd.read_csv('myocardial_infarction_data.csv') 
    X = df.values

    # 2. Initialize Smooth Trainer
    # This model normalizes features internally via Tanh activations
    model = ddcd.DDCD_Smooth_Trainer(
        X, 
        device='cuda',
        time_dim=16,
        batch_size=64
    )

    # 3. Train
    model.train(n_steps=2000)

    # 4. Analysis
    # For Smooth models, we analyze the normalized adjacency strength
    weighted_adj = model.get_adj()
    
    # Extract top causal links
    threshold = 0.2
    strong_links = np.argwhere(np.abs(weighted_adj) > threshold)
    print(f"Found {len(strong_links)} strong causal connections.")