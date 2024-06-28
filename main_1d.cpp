// main.cpp
#include <torch/torch.h>
#include <iostream>
#include "weno5_advection.h"

int main() {
    // Domain and initial condition
    int n = 20;
    double L = 1.0;
    double dx = L / n;
    auto x = torch::linspace(0, L, n);
    auto u = torch::exp(-100.0 * (x - 0.5).pow(2));
    
    // Velocity field
    auto a = torch::ones_like(u) * 1.0; // Uniform velocity field

    // Time-stepping parameters
    double dt = 0.001; // Ensure CFL condition: dt < dx / max(a)
    int num_steps = 500;

    // Choose boundary condition
    BoundaryCondition bc = DOUBLE_PERIODIC; // two options: ZERO_GRADIENT, DOUBLE_PERIODIC


    for (int i = 0; i < n; ++i) {
        std::printf("%.2f ", abs(u[i].item<double>()));
    }    
    std::cout << std::endl;
    std::cout << std::endl;

    // Time-stepping loop
    for (int i = 0; i < num_steps; ++i) {
        time_step(u, a, dt, dx, bc);
    }

    for (int i = 0; i < n; ++i) {
        std::printf("%.2f ", abs(u[i].item<double>()));
    }    
    
    return 0;
}
