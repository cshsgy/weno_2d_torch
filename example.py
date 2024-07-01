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

# Define boundary condition (0 for ZERO_GRADIENT and 1 for DOUBLE_PERIODIC?)
bc = 1

# Perform time-stepping
for _ in range(num_steps):
    u = weno5_advection.time_step_2d_return(u, ax, ay, dt, dx, dy, bc)

plt.contourf(X.cpu().numpy(), Y.cpu().numpy(), u.cpu().numpy())
plt.show()