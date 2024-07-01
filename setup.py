from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

torch_include = '/opt/homebrew/lib/python3.11/site-packages/torch/include/'
torch_api_include = '/opt/homebrew/lib/python3.11/site-packages/torch/include/torch/csrc/api/include/'
netcdf_include = '/opt/homebrew/opt/netcdf/include/'

setup(
    name='weno5_advection',
    ext_modules=[
        CppExtension(
            name='weno5_advection',
            sources=['weno5_advection.cpp'],
            extra_compile_args=['-std=c++17'],
            include_dirs=[torch_include, torch_api_include]
            # torch.utils.cpp_extension.include_paths(), but somehow does not work
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)

# Sample usage: python setup.py build_ext --inplace
