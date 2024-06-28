import numpy as np
from netCDF4 import Dataset
from matplotlib import pyplot as plt

data_dir = '../build/'
final_files = ['final_10.nc', 'final_20.nc', 'final_40.nc', 'final_80.nc']
init_files = ['initial_10.nc', 'initial_20.nc', 'initial_40.nc', 'initial_80.nc']

fig, axs = plt.subplots(2, 2, figsize=(10, 10))
error = []
error_x = [10, 20, 40, 80]
for i in range(len(final_files)):
    final_file = data_dir + final_files[i]
    init_file = data_dir + init_files[i]
    final_data = Dataset(final_file, 'r')
    init_data = Dataset(init_file, 'r')

    final = final_data.variables['data']
    init = init_data.variables['data']

    final = np.array(final)
    init = np.array(init)
    if i==1:
        axs[0, 0].imshow(final)
        axs[0, 0].set_title('Final data-C20')
    if i==2:
        axs[0, 1].imshow(final)
        axs[0, 1].set_title('Final data-C40')
    if i==3:
        axs[1, 0].imshow(final)
        axs[1, 0].set_title('Final data-C80')

    error.append(np.sum(np.abs(final - init) * np.abs(final - init)))

    final_data.close()
    init_data.close()

axs[1, 1].scatter(np.log(error_x), np.log(error))
axs[1, 1].set_title('Error: 3rd to 4th order convergence')
axs[1, 1].set_xlabel('log N')
axs[1, 1].set_ylabel('log error')
# plot fitting line
fit = np.polyfit(np.log(error_x), np.log(error), 1)
fit_fn = np.poly1d(fit)
axs[1, 1].plot(np.log(error_x), fit_fn(np.log(error_x)), '--k', label='y = {:.2f}x + {:.2f}'.format(fit[0], fit[1]))
axs[1, 1].legend()
plt.savefig('example_and_error.png')