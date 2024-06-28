// weno5_advection.h
#ifndef WENO5_ADVECTION_H
#define WENO5_ADVECTION_H

#include <torch/torch.h>
#include <tuple>

// Boundary condition types
enum BoundaryCondition {
    ZERO_GRADIENT,
    DOUBLE_PERIODIC
};

// WENO5 reconstruction function
torch::Tensor weno5_reconstruction(const torch::Tensor& u, const torch::Tensor& a, BoundaryCondition bc);

// Residual calculation function
torch::Tensor residual(const torch::Tensor& u, const torch::Tensor& a, double dx, BoundaryCondition bc);

// Time stepping function using RK3
void time_step(torch::Tensor& u, const torch::Tensor& a, double dt, double dx, BoundaryCondition bc);

// 2D-related functions
torch::Tensor apply_boundary_conditions_x(const torch::Tensor& u, BoundaryCondition bc);
torch::Tensor apply_boundary_conditions_y(const torch::Tensor& u, BoundaryCondition bc);
torch::Tensor weno5_reconstruction_x(const torch::Tensor& u, const torch::Tensor& a, BoundaryCondition bc);
torch::Tensor weno5_reconstruction_y(const torch::Tensor& u, const torch::Tensor& a, BoundaryCondition bc);
torch::Tensor residual_2d(const torch::Tensor& u, const torch::Tensor& ax, const torch::Tensor& ay, double dx, double dy, BoundaryCondition bc);
void time_step_2d(torch::Tensor& u, const torch::Tensor& ax, const torch::Tensor& ay, double dt, double dx, double dy, BoundaryCondition bc);

#endif // WENO5_ADVECTION_H
