// weno5_advection.h
#ifndef WENO5_ADVECTION_H
#define WENO5_ADVECTION_H

#include <torch/torch.h>
#include <tuple>

// WENO5 reconstruction function
torch::Tensor weno5_reconstruction(const torch::Tensor& u, const torch::Tensor& a, int bc);

// Residual calculation function
torch::Tensor residual(const torch::Tensor& u, const torch::Tensor& a, double dx, int bc);

// Time stepping function using RK3
void time_step(torch::Tensor& u, const torch::Tensor& a, double dt, double dx, int bc);

// 2D-related functions
torch::Tensor apply_boundary_conditions_x(const torch::Tensor& u, int bc);
torch::Tensor apply_boundary_conditions_y(const torch::Tensor& u, int bc);
torch::Tensor weno5_reconstruction_x(const torch::Tensor& u, const torch::Tensor& a, int bc);
torch::Tensor weno5_reconstruction_y(const torch::Tensor& u, const torch::Tensor& a, int bc);
torch::Tensor residual_2d(const torch::Tensor& u, const torch::Tensor& ax, const torch::Tensor& ay, double dx, double dy, int bc);
void time_step_2d(torch::Tensor& u, const torch::Tensor& ax, const torch::Tensor& ay, double dt, double dx, double dy, int bc);

torch::Tensor residual_2d_general(const torch::Tensor& u, const torch::Tensor& ax, const torch::Tensor& ay, 
        const torch::Tensor& dA_x, const torch::Tensor& dA_y, const torch::Tensor& dV, int bc);
torch::Tensor time_step_2d_general(const torch::Tensor& u, const torch::Tensor& ax, const torch::Tensor& ay, 
        double dt, const torch::Tensor& dA_x, const torch::Tensor& dA_y, const torch::Tensor& dV, int bc);

// Python-related functions
torch::Tensor time_step_2d_return(const torch::Tensor& u, const torch::Tensor& ax, const torch::Tensor& ay, double dt, double dx, double dy, int bc);


#endif // WENO5_ADVECTION_H
