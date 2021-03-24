
from utils import data_IO
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':

    binvox_file = '/home/zmy/Desktop/model.binvox'
    output_array = data_IO.read_voxel_data(binvox_file)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    voxel_array = np.swapaxes(output_array, 1, 2)
    ax.voxels(voxel_array.astype(np.bool), edgecolors='k')
    plt.axis('off')
    plt.savefig('./model.png')
    plt.close()


