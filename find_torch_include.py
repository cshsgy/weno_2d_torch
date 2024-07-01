import torch
import os

print(torch.__version__)
print(torch.__file__)
# Get the include paths for PyTorch
torch_include = os.path.join(torch.utils.cpp_extension.include_paths()[0])
torch_include_api = os.path.join(torch.utils.cpp_extension.include_paths()[1])

print(f'torch_include: {torch_include}')
print(f'torch_include_api: {torch_include_api}')
