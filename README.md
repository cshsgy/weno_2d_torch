# weno_2d_torch

Boundary condition flags:
0 = ZERO_GRADIENT
1 = DOUBLE_PERIDIOC
2 = ZERO_FLUX

There are still several things to be done in this build:
1. Some of the tensors are somehow put onto CPU instead of GPU. Need to check how to optimize the performance by running codes on GPU.
2. Use ATen to generate parallized functions.
3. Add the functionality of grid specification.
