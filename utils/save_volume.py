
import numpy as np
import matplotlib.pyplot as plt
from utils import data_IO

# using in test_VAE.py
def save_binvox_output(output_arr, output_hash, output_dir, outname, save_bin = False, save_img = True):

    # save objedt as .binvox
    if save_bin:
        s1 = output_dir + '/' + output_hash + outname + '.binvox'
        print('The s1 is', s1)
        s1 = bytes(s1, 'utf-8')
        data_IO.write_binvox_file(output_arr, s1)

    # save the model image
    if save_img:
        fig = plt.figure()
        ax =fig.gca(projection = '3d')
        ax.voxels(output_arr.astype(np.bool), edgecolors='k')
        plt.savefig(output_dir + '/' + output_hash + outname + '.png')
        plt.close()

# using in test_MMI.py
def save_binvox_output_2(output_arr, output_hash, output_dir, outname, save_bin = False, save_img = True):

    # save objedt as .binvox
    if save_bin:
        s1 = output_dir + '/' + output_hash + outname + '.binvox'
        print('The s1 is', s1)
        s1 = bytes(s1, 'utf-8')
        data_IO.write_binvox_file(output_arr, s1)

    # save the model image
    if save_img:
        fig = plt.figure()
        ax =fig.gca(projection = '3d')
        voxel_array = np.swapaxes(output_arr, 1, 2)
        ax.voxels(voxel_array.astype(np.bool), edgecolors='k')
        plt.savefig(output_dir + '/' + output_hash + outname + '.png')
        plt.close()

def binvox2image(voxel_file, hash_id, output_dir, outname=''):

    voxel_array = data_IO.read_voxel_data(voxel_file)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(voxel_array.astype(np.bool), edgecolors='k')
    plt.savefig(output_dir + '/' + hash_id + outname + '.png')
    plt.close()

def binvox2image_2(voxel_file, hash_id, output_dir, outname=''):

    voxel_array = data_IO.read_voxel_data(voxel_file)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    voxel_array=np.swapaxes(voxel_array, 1,2)

    ax.voxels(voxel_array.astype(np.bool), edgecolors='k')
    plt.savefig(output_dir + '/' + hash_id + outname + '.png')
    plt.close()


