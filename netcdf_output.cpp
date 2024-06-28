#include "netcdf_output.h"
#include <netcdf.h>
#include <iostream>

// Function to handle NetCDF errors
void handle_nc_error(int retval) {
    if (retval != NC_NOERR) {
        std::cerr << "NetCDF error: " << nc_strerror(retval) << std::endl;
        exit(EXIT_FAILURE);
    }
}

// Function to output a tensor to a NetCDF file
void output_to_netcdf(const torch::Tensor& tensor, const std::string& filename) {
    // Ensure tensor is on CPU
    torch::Tensor cpu_tensor = tensor.to(torch::kCPU);
    
    // Get tensor dimensions
    auto sizes = cpu_tensor.sizes();
    int ndim = sizes.size();
    std::vector<int> dim_sizes(ndim);
    for (int i = 0; i < ndim; ++i) {
        dim_sizes[i] = sizes[i];
    }
    
    // Create NetCDF file
    int ncid;
    handle_nc_error(nc_create(filename.c_str(), NC_CLOBBER, &ncid));
    
    // Define dimensions
    std::vector<int> dimids(ndim);
    for (int i = 0; i < ndim; ++i) {
        handle_nc_error(nc_def_dim(ncid, ("dim_" + std::to_string(i)).c_str(), dim_sizes[i], &dimids[i]));
    }
    
    // Define variable
    int varid;
    handle_nc_error(nc_def_var(ncid, "data", NC_FLOAT, ndim, dimids.data(), &varid));
    
    // End define mode
    handle_nc_error(nc_enddef(ncid));
    
    // Write data to variable
    handle_nc_error(nc_put_var_float(ncid, varid, cpu_tensor.data_ptr<float>()));
    
    // Close NetCDF file
    handle_nc_error(nc_close(ncid));
}
