# Causal Structure Learning via Diffusion Denoising Objectives

Understanding causal dependencies in observational data is critical for informing decision-making. These relationships are often modeled as Bayesian Networks (BNs) and Directed Acyclic Graphs (DAGs). Existing methods, such as NOTEARS and DAG-GNN, often face issues with scalability and stability in high-dimensional data, especially when there is a feature-sample imbalance. Here, we show that the denoising score matching objective of diffusion models could smooth the gradients for faster, more stable convergence. We also propose an adaptive k-hop acyclicity constraint that improves runtime over existing solutions that require matrix inversion. We name this framework Denoising Diffusion Causal Discovery (DDCD). Unlike generative diffusion models, DDCD utilizes the reverse denoising process to infer a parameterized causal structure rather than to generate data. We demonstrate the competitive performance of DDCDs on synthetic benchmarking data. We also show that our methods are practically useful by conducting qualitative analyses on two real-world examples.

# Get started

## Installation
```
pip install ddcd
```

## Example
```
import ddcd
from castle.datasets import IIDSimulation, DAG

# an unwanted behavior from castle
torch.set_default_dtype(torch.float)

# Generating synthetic data 
dag_adj = DAG.scale_free(
    n_nodes = 100, n_edges = 1000,
    weight_range = (0.5, 1.5), seed=42
)

X = IIDSimulation(
    W=dag_adj, 
    n=2000, method='linear', 
    sem_type='gauss', noise_scale=1
).X 

# Training
model = ddcd.DDCD_Linear_Trainer(X, device='cuda')
model.train(5000)

w = model.get_adj()
A = (np.abs(w) > 0.3) 
```