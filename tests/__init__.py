import torch

# The oracles compare against closed-form evidences, bounds and gradients to tight tolerances,
# so every test runs in double precision.
torch.set_default_dtype(torch.float64)
