// weno5_advection.cpp
#include <torch/torch.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>

#include "weno5_advection.h"

// Helper function to compute the WENO5 weights
torch::Tensor weno5_weights(const torch::Tensor& beta) {
    auto epsilon = 1e-6;
    torch::Tensor gamma = torch::tensor({1.0/10.0, 3.0/5.0, 3.0/10.0}, beta.device());
    // If beta is 3*n, then we need to have gamma be 3*1, to enable broadcasting
    int sum_axis = 1;
    if (beta.size(0) == 3) {
        gamma = gamma.unsqueeze(1);
        sum_axis = 0;
    }
    auto alpha = gamma / ((beta + epsilon).pow(2));
    auto sum_alpha = alpha.sum(sum_axis, true);
    auto weights = alpha / sum_alpha;
    return weights;
}

// Helper functions to apply boundary conditions
torch::Tensor apply_boundary_conditions(const torch::Tensor& u, int bc) {
    int n = u.size(0);
    auto extended_u = torch::zeros({n + 6});
    if (bc == 0) {
        extended_u.slice(0, 3, -3) = u;
        extended_u[0] = u[0];
        extended_u[1] = u[0];
        extended_u[2] = u[0];
        extended_u[-2] = u[-1];
        extended_u[-1] = u[-1];
        extended_u[-3] = u[-1];
    } else if (bc == 1) {
        extended_u.slice(0, 3, -3) = u;
        extended_u[0] = u[-3];
        extended_u[1] = u[-2];
        extended_u[2] = u[-1];
        extended_u[-3] = u[0];
        extended_u[-2] = u[1];
        extended_u[-1] = u[2];
    } else if (bc == 2) {
        extended_u.slice(0, 3, -3) = u;
        extended_u[0] = 2*u[0]-u[2];
        extended_u[1] = 2*u[0]-u[1];
        extended_u[2] = u[0];
        extended_u[-3] = u[-1];
        extended_u[-2] = 2*u[-1] - u[-2];
        extended_u[-1] = 2*u[-1] - u[-3];
    }

    return extended_u;
}

torch::Tensor apply_boundary_conditions_x(const torch::Tensor& u, int bc) {
    int n = u.size(1);
    auto extended_u = torch::zeros({u.size(0), n + 6}, u.device());
    if (bc == 0) {
        extended_u.slice(1, 3, -3) = u;
        extended_u.index({torch::indexing::Slice(), 0}) = u.index({torch::indexing::Slice(), 0});
        extended_u.index({torch::indexing::Slice(), 1}) = u.index({torch::indexing::Slice(), 0});
        extended_u.index({torch::indexing::Slice(), 2}) = u.index({torch::indexing::Slice(), 0});
        extended_u.index({torch::indexing::Slice(), -3}) = u.index({torch::indexing::Slice(), -1});
        extended_u.index({torch::indexing::Slice(), -2}) = u.index({torch::indexing::Slice(), -1});
        extended_u.index({torch::indexing::Slice(), -1}) = u.index({torch::indexing::Slice(), -1});
    } else if (bc == 1) {
        extended_u.slice(1, 3, -3) = u;
        extended_u.index({torch::indexing::Slice(), 0}) = u.index({torch::indexing::Slice(), -3});
        extended_u.index({torch::indexing::Slice(), 1}) = u.index({torch::indexing::Slice(), -2});
        extended_u.index({torch::indexing::Slice(), 2}) = u.index({torch::indexing::Slice(), -1});
        extended_u.index({torch::indexing::Slice(), -3}) = u.index({torch::indexing::Slice(), 0});
        extended_u.index({torch::indexing::Slice(), -2}) = u.index({torch::indexing::Slice(), 1});
        extended_u.index({torch::indexing::Slice(), -1}) = u.index({torch::indexing::Slice(), 2});
    } else if (bc == 2) {
        extended_u.slice(1, 3, -3) = u;
        extended_u.index({torch::indexing::Slice(), 0}) = u.index({torch::indexing::Slice(), 2});
        extended_u.index({torch::indexing::Slice(), 1}) = u.index({torch::indexing::Slice(), 1});
        extended_u.index({torch::indexing::Slice(), 2}) = u.index({torch::indexing::Slice(), 0});
        extended_u.index({torch::indexing::Slice(), -3}) = u.index({torch::indexing::Slice(), -1});
        extended_u.index({torch::indexing::Slice(), -2}) = u.index({torch::indexing::Slice(), -2});
        extended_u.index({torch::indexing::Slice(), -1}) = u.index({torch::indexing::Slice(), -3});
    }

    return extended_u;
}

torch::Tensor apply_boundary_conditions_y(const torch::Tensor& u, int bc) {
    int n = u.size(0);
    auto extended_u = torch::zeros({n + 6, u.size(1)}, u.device());
    if (bc == 0) {
        extended_u.slice(0, 3, -3) = u;
        extended_u.index({0, torch::indexing::Slice()}) = u.index({0, torch::indexing::Slice()});
        extended_u.index({1, torch::indexing::Slice()}) = u.index({0, torch::indexing::Slice()});
        extended_u.index({2, torch::indexing::Slice()}) = u.index({0, torch::indexing::Slice()});
        extended_u.index({-3, torch::indexing::Slice()}) = u.index({-1, torch::indexing::Slice()});
        extended_u.index({-2, torch::indexing::Slice()}) = u.index({-1, torch::indexing::Slice()});
        extended_u.index({-1, torch::indexing::Slice()}) = u.index({-1, torch::indexing::Slice()});
    } else if (bc == 1) {
        extended_u.slice(0, 3, -3) = u;
        extended_u.index({0, torch::indexing::Slice()}) = u.index({-3, torch::indexing::Slice()});
        extended_u.index({1, torch::indexing::Slice()}) = u.index({-2, torch::indexing::Slice()});
        extended_u.index({2, torch::indexing::Slice()}) = u.index({-1, torch::indexing::Slice()});
        extended_u.index({-3, torch::indexing::Slice()}) = u.index({0, torch::indexing::Slice()});
        extended_u.index({-2, torch::indexing::Slice()}) = u.index({1, torch::indexing::Slice()});
        extended_u.index({-1, torch::indexing::Slice()}) = u.index({2, torch::indexing::Slice()});
    } else if (bc == 2) {
        extended_u.slice(0, 3, -3) = u;
        extended_u.index({0, torch::indexing::Slice()}) = u.index({2, torch::indexing::Slice()});
        extended_u.index({1, torch::indexing::Slice()}) = u.index({1, torch::indexing::Slice()});
        extended_u.index({2, torch::indexing::Slice()}) = u.index({0, torch::indexing::Slice()});
        extended_u.index({-3, torch::indexing::Slice()}) = u.index({-1, torch::indexing::Slice()});
        extended_u.index({-2, torch::indexing::Slice()}) = u.index({-2, torch::indexing::Slice()});
        extended_u.index({-1, torch::indexing::Slice()}) = u.index({-3, torch::indexing::Slice()});
    }

    return extended_u;
}

// WENO5 reconstruction function
torch::Tensor weno5_reconstruction(const torch::Tensor& u, const torch::Tensor& a, int bc) {
    int n = u.size(0);
    torch::Tensor flux = torch::zeros({n+2});
    // Apply boundary conditions, extend the domain to n+6
    auto extended_u = apply_boundary_conditions(u, bc);
    auto extended_a = apply_boundary_conditions(a, bc);
    // Convert u to flux
    extended_u = extended_u * extended_a;
    // Left flux
    for (int i = 2; i < n + 4; ++i) {
        auto f = extended_u.slice(0, i-2, i+3);
        auto beta = torch::zeros({3});
        beta[0] = 13.0/12.0 * (f[0] - 2*f[1] + f[2]).pow(2) + 1.0/4.0 * (f[0] - 4*f[1] + 3*f[2]).pow(2);
        beta[1] = 13.0/12.0 * (f[1] - 2*f[2] + f[3]).pow(2) + 1.0/4.0 * (f[1] - f[3]).pow(2);
        beta[2] = 13.0/12.0 * (f[2] - 2*f[3] + f[4]).pow(2) + 1.0/4.0 * (3*f[2] - 4*f[3] + f[4]).pow(2);
        
        auto weights = weno5_weights(beta);
        flux[i-2] = weights[0] * (2*f[0] - 7*f[1] + 11*f[2]) / 6.0 +
                  weights[1] * (-f[1] + 5*f[2] + 2*f[3]) / 6.0 +
                  weights[2] * (2*f[2] + 5*f[3] - f[4]) / 6.0;
    }

    return flux; // Output dimension n+2
}

// WENO5 reconstruction function on x-axis, along the rows
torch::Tensor weno5_reconstruction_x(const torch::Tensor& u, const torch::Tensor& a, int bc) {
    int n = u.size(1);
    torch::Tensor flux = torch::zeros({u.size(0), n+2}, u.device());
    // Apply boundary conditions, extend the domain to n+6
    auto extended_u = apply_boundary_conditions_x(u, bc);
    auto extended_a = apply_boundary_conditions_x(a, bc);
    // Convert u to flux
    extended_u = extended_u * extended_a;
    // Left flux
    for (int i = 2; i < n + 4; ++i) {
        auto f1 = extended_u.index({torch::indexing::Slice(), i-2});
        auto f2 = extended_u.index({torch::indexing::Slice(), i-1});
        auto f3 = extended_u.index({torch::indexing::Slice(), i});
        auto f4 = extended_u.index({torch::indexing::Slice(), i+1});
        auto f5 = extended_u.index({torch::indexing::Slice(), i+2});
        auto beta = torch::zeros({u.size(0), 3});
        beta.index({torch::indexing::Slice(), 0}) = 13.0/12.0 * (f1 - 2*f2 + f3).pow(2) + 1.0/4.0 * (f1 - 4*f2 + 3*f3).pow(2);
        beta.index({torch::indexing::Slice(), 1}) = 13.0/12.0 * (f2 - 2*f3 + f4).pow(2) + 1.0/4.0 * (f2 - f4).pow(2);
        beta.index({torch::indexing::Slice(), 2}) = 13.0/12.0 * (f3 - 2*f4 + f5).pow(2) + 1.0/4.0 * (3*f3 - 4*f4 + f5).pow(2);
        
        auto weights = weno5_weights(beta); 

        flux.index({torch::indexing::Slice(), i-2}) = weights.index({torch::indexing::Slice(), 0}) * (2*f1 - 7*f2 + 11*f3) / 6.0 +
                  weights.index({torch::indexing::Slice(), 1}) * (-f2 + 5*f3 + 2*f4) / 6.0 +
                  weights.index({torch::indexing::Slice(), 2}) * (2*f3 + 5*f4 - f5) / 6.0;
    }

    return flux; // Output dimension n+2
}

// WENO5 reconstruction function on y-axis, along the columns
torch::Tensor weno5_reconstruction_y(const torch::Tensor& u, const torch::Tensor& a, int bc) {
    int n = u.size(0);
    torch::Tensor flux = torch::zeros({n+2, u.size(1)}, u.device());
    // Apply boundary conditions, extend the domain to n+6
    auto extended_u = apply_boundary_conditions_y(u, bc);
    auto extended_a = apply_boundary_conditions_y(a, bc);
    // Convert u to flux
    extended_u = extended_u * extended_a;
    // Left flux
    for (int i = 2; i < n + 4; ++i) {
        auto f1 = extended_u.index({i-2, torch::indexing::Slice()});
        auto f2 = extended_u.index({i-1, torch::indexing::Slice()});
        auto f3 = extended_u.index({i, torch::indexing::Slice()});
        auto f4 = extended_u.index({i+1, torch::indexing::Slice()});
        auto f5 = extended_u.index({i+2, torch::indexing::Slice()});
        auto beta = torch::zeros({3, u.size(1)});
        beta.index({0, torch::indexing::Slice()}) = 13.0/12.0 * (f1 - 2*f2 + f3).pow(2) + 1.0/4.0 * (f1 - 4*f2 + 3*f3).pow(2);
        beta.index({1, torch::indexing::Slice()}) = 13.0/12.0 * (f2 - 2*f3 + f4).pow(2) + 1.0/4.0 * (f2 - f4).pow(2);
        beta.index({2, torch::indexing::Slice()}) = 13.0/12.0 * (f3 - 2*f4 + f5).pow(2) + 1.0/4.0 * (3*f3 - 4*f4 + f5).pow(2);
        
        auto weights = weno5_weights(beta); 

        flux.index({i-2, torch::indexing::Slice()}) = weights.index({0, torch::indexing::Slice()}) * (2*f1 - 7*f2 + 11*f3) / 6.0 +
                  weights.index({1, torch::indexing::Slice()}) * (-f2 + 5*f3 + 2*f4) / 6.0 +
                  weights.index({2, torch::indexing::Slice()}) * (2*f3 + 5*f4 - f5) / 6.0;
    }

    return flux; // Output dimension n+2
}

// Residual calculation function
torch::Tensor residual(const torch::Tensor& u, const torch::Tensor& a, double dx, int bc) {
    auto flux = weno5_reconstruction(u, a, bc);
    torch::Tensor res = torch::zeros_like(u);
    res = (flux.slice(0, 1, -1) - flux.slice(0, 0, -2)) / dx;
    return res;
}

// 2D residual calculation function
torch::Tensor residual_2d(const torch::Tensor& u, const torch::Tensor& ax, const torch::Tensor& ay, double dx, double dy, int bc) {
    auto flux_x = weno5_reconstruction_x(u, ax, bc);
    auto flux_y = weno5_reconstruction_y(u, ay, bc);
    torch::Tensor res = torch::zeros_like(u, u.device());
    res = (flux_x.slice(1, 1, -1) - flux_x.slice(1, 0, -2)) / dx;
    res += (flux_y.slice(0, 1, -1) - flux_y.slice(0, 0, -2)) / dy;
    return res;
}

// 2D residual calculation function
torch::Tensor residual_2d_general(const torch::Tensor& u, const torch::Tensor& ax, const torch::Tensor& ay, 
        const torch::Tensor& dA_x, const torch::Tensor& dA_y, const torch::Tensor& dV, int bc) {
    auto flux_x = weno5_reconstruction_x(u, ax, bc);
    auto flux_y = weno5_reconstruction_y(u, ay, bc);
    torch::Tensor res = torch::zeros_like(u, u.device());
    res = (flux_x.slice(1, 1, -1) * dA_x.slice(1, 1, torch::nullopt) - flux_x.slice(1, 0, -2) * dA_x.slice(1, 0, -1)) / dV;
    res += (flux_y.slice(0, 1, -1) * dA_y.slice(0, 1, torch::nullopt) - flux_y.slice(0, 0, -2) * dA_y.slice(0, 0, -1)) / dV;
    return res;
}

// Time stepping function using RK3
void time_step(torch::Tensor& u, const torch::Tensor& a, double dt, double dx, int bc) {
    auto k1 = residual(u, a, dx, bc);
    auto u1 = u - dt * k1;
    auto k2 = residual(u1, a, dx, bc);
    auto u2 = 3.0/4.0 * u + 1.0/4.0 * (u1 - dt * k2);
    auto k3 = residual(u2, a, dx, bc);
    u = 1.0/3.0 * u + 2.0/3.0 * (u2 - dt * k3);
}

// 2D time stepping function using RK3
void time_step_2d(torch::Tensor& u, const torch::Tensor& ax, const torch::Tensor& ay, double dt, double dx, double dy, int bc) {
    auto k1 = residual_2d(u, ax, ay, dx, dy, bc);
    auto u1 = u - dt * k1;
    // std::cout << u1.device() << std::endl;
    auto k2 = residual_2d(u1, ax, ay, dx, dy, bc);
    auto u2 = 3.0/4.0 * u + 1.0/4.0 * (u1 - dt * k2);
    auto k3 = residual_2d(u2, ax, ay, dx, dy, bc);
    u = 1.0/3.0 * u + 2.0/3.0 * (u2 - dt * k3);
}

// 2D time stepping function using RK3
torch::Tensor time_step_2d_return(const torch::Tensor& u, const torch::Tensor& ax, const torch::Tensor& ay, double dt, double dx, double dy, int bc) {
    auto k1 = residual_2d(u, ax, ay, dx, dy, bc);
    auto u1 = u - dt * k1;
    // std::cout << u1.device() << std::endl;
    auto k2 = residual_2d(u1, ax, ay, dx, dy, bc);
    auto u2 = 3.0/4.0 * u + 1.0/4.0 * (u1 - dt * k2);
    auto k3 = residual_2d(u2, ax, ay, dx, dy, bc);
    auto new_u = 1.0/3.0 * u + 2.0/3.0 * (u2 - dt * k3);
    return new_u;
}

// 2D time stepping function using RK3, general grid specification with dA_x, dA_y and dV
torch::Tensor time_step_2d_general(const torch::Tensor& u, const torch::Tensor& ax, const torch::Tensor& ay, 
        double dt, const torch::Tensor& dA_x, const torch::Tensor& dA_y, const torch::Tensor& dV, int bc) {
    auto k1 = residual_2d_general(u, ax, ay, dA_x, dA_y, dV, bc);
    auto u1 = u - dt * k1;
    // std::cout << u1.device() << std::endl;
    auto k2 = residual_2d_general(u1, ax, ay, dA_x, dA_y, dV, bc);
    auto u2 = 3.0/4.0 * u + 1.0/4.0 * (u1 - dt * k2);
    auto k3 = residual_2d_general(u2, ax, ay, dA_x, dA_y, dV, bc);
    auto new_u = 1.0/3.0 * u + 2.0/3.0 * (u2 - dt * k3);
    return new_u;
}

namespace py = pybind11;

// Bind the time_step_2d function to Python
PYBIND11_MODULE(weno5_advection, m) {
    m.def("time_step_2d_general", &time_step_2d_general, "Time Stepping for 2D Advection",
          py::arg("u"), py::arg("ax"), py::arg("ay"), py::arg("dt"), 
          py::arg("dA_x"), py::arg("dA_y"), py::arg("dV"), py::arg("bc"));
}
