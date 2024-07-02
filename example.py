import torch
import weno5_advection

from matplotlib import pyplot as plt

# Initialize your tensors
nx, ny = 20, 20
Lx, Ly = 1.0, 1.0
dx = Lx / nx
dy = Ly / ny
dt = 0.001
num_steps = 500

x = torch.linspace(0, Lx, nx)
y = torch.linspace(0, Ly, ny)
X, Y = torch.meshgrid(x, y)
u = torch.exp(-100.0 * ((X - 0.5)**2 + (Y - 0.5)**2))
ax = torch.ones_like(u)
ay = torch.ones_like(u)
dV = torch.ones_like(u) * dx * dy
dA_x = torch.ones(u.size(0), u.size(1) + 1) * dy
dA_y = torch.ones(u.size(0) + 1, u.size(1)) * dx

# Define boundary condition (0 for ZERO_GRADIENT and 1 for DOUBLE_PERIODIC and 2 for ZERO_FLUX)
bc = 2

# Perform time-stepping
for _ in range(num_steps):
    u = weno5_advection.time_step_2d_general(u, ax, ay, dt, dA_x, dA_y, dV, bc)

plt.contourf(X.cpu().numpy(), Y.cpu().numpy(), u.cpu().numpy())
plt.show()