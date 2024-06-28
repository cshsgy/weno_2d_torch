#include <torch/torch.h>
#include <iostream>
#include "weno5_advection.h"
#include "netcdf_output.h"

int main() {
    // Check if Metal is available
    torch::Device device = torch::kCPU;
    if (torch::hasMPS()) {
        std::cout << "MPS is available! Running on GPU." << std::endl;
        device = torch::kMPS;
    } else {
        std::cout << "MPS is not available. Running on CPU." << std::endl;
    }


    // Domain and initial condition
    int Cn = 10;
    int nx = Cn;
    int ny = Cn;
    double Lx = 1.0;
    double Ly = 1.0;
    double dx = Lx / nx;
    double dy = Ly / ny;
    auto x = torch::linspace(0, Lx, nx);
    auto y = torch::linspace(0, Ly, ny);
    auto X = torch::meshgrid({x, y});
    auto u = torch::exp(-100.0 * ((X[0] - 0.5).pow(2) + (X[1] - 0.5).pow(2)));
    u.to(device);
    output_to_netcdf(u, "initial_" + std::to_string(Cn) + ".nc");
    // Velocity field
    auto ax = torch::ones_like(u) * 1.0; 
    auto ay = torch::ones_like(u) * 1.0; 
    ax.to(device);
    ay.to(device);

    // Time-stepping parameters
    double dt = 0.001; // Ensure CFL condition: dt < min(dx, dy) / max(a)
    int num_steps = 1000; // Adjust the number of steps to control the simulation time

    // Choose boundary condition
    BoundaryCondition bc = DOUBLE_PERIODIC; // or ZERO_GRADIENT
    
    // Time-stepping loop
    for (int i = 0; i < num_steps; ++i) {
        time_step_2d(u, ax, ay, dt, dx, dy, bc);
    }
    
    // Output the result
    output_to_netcdf(u, "final_" + std::to_string(Cn) + ".nc");    
    return 0;
}
