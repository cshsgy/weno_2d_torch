#ifndef NETCDF_OUTPUT_H
#define NETCDF_OUTPUT_H

#include <torch/torch.h>
#include <string>

// Function to output a tensor to a NetCDF file
void output_to_netcdf(const torch::Tensor& tensor, const std::string& filename);

#endif // NETCDF_OUTPUT_H
