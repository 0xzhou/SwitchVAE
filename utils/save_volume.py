from __future__ import division, print_function, absolute_import

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from skimage import measure
from utils import data_IO

def save_output(output_arr, output_size, output_dir, file_idx):
    plot_out_arr = np.array([])
    with_border_arr = np.zeros([34, 34, 34])
    for x_i in range(0, output_size):
        for y_j in range(0, output_size):
            for z_k in range(0, output_size):
                plot_out_arr = np.append(plot_out_arr, output_arr[x_i, y_j, z_k])
                
    text_save = np.reshape(plot_out_arr, (output_size * output_size * output_size))
    np.savetxt(output_dir + '/volume' + str(file_idx) + '.txt', text_save)

    output_image = np.reshape(plot_out_arr, (output_size, output_size, output_size)).astype(np.float32)

    for x_i in range(0, output_size):
        for y_j in range(0, output_size):
            for z_k in range(0, output_size):
                with_border_arr[x_i + 1, y_j + 1, z_k + 1] = output_image[x_i, y_j, z_k]

    if not np.any(with_border_arr):
        verts, faces, normals, values = [], [], [], []
    else:
        verts, faces, normals, values = measure.marching_cubes_lewiner(with_border_arr, level = 0.0, gradient_direction = 'descent')
        faces = faces + 1

    obj_save = open(output_dir + '/volume' + str(file_idx) + '.obj', 'w')
    for item in verts:
        obj_save.write('v {0} {1} {2}\n'.format(item[0], item[1], item[2]))
    for item in normals:
        obj_save.write('vn {0} {1} {2}\n'.format(-item[0], -item[1], -item[2]))
    for item in faces:
        obj_save.write('f {0}//{0} {1}//{1} {2}//{2}\n'.format(item[0], item[2], item[1]))
    obj_save.close()

    output_image = np.rot90(output_image)
    x, y, z = output_image.nonzero()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    ax.scatter(x, y, z, zdir = 'z', c = 'red')
    plt.savefig(output_dir + '/volume' + str(file_idx) + '.png')
    plt.close()


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
        output_arr = np.swapaxes(output_arr, 1, 2)
        ax.voxels(output_arr.astype(np.bool), edgecolors='k')
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
    voxel_array = np.swapaxes(voxel_array,1,2)
    ax.voxels(voxel_array.astype(np.bool), edgecolors='k')
    plt.savefig(output_dir + '/' + hash_id + outname + '.png')
    plt.close()


