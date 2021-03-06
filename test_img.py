
import matplotlib.pyplot as plt
from utils import data_IO
import glob
from PIL import Image
import numpy as np


def create_img_from_modelnet(modelnet_path, save_path):
    binvox_files = glob.glob(modelnet_path+"/*binvox")
    names = [names.split('.') for names in os.listdir(modelnet_path) if names.endswith('.binvox')]

    for i,file in enumerate(binvox_files):
        array=data_IO.read_voxel_data(file)
        array=np.int32(array)
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.voxels(array.astype(np.bool), edgecolors='k')
        plt.savefig(save_path + '/' + names[i][0] + '.png')
        plt.close()


if __name__ == '__main__':


    # image=data_IO.preprocess_modelnet_img(img='/home/zmy/Datasets/ModelNet40_images/modelnet40_images_new_12x/airplane/train/airplane_0001.obj.shaded_v001.png',
    #                                              BG_rgb=[255, 255, 255],
    #                                              aim_size=(137, 137))
    #
    # # image = Image.open('/home/zmy/Datasets/ModelNet40_images/modelnet40_images_new_12x/airplane/train/airplane_0001.obj.shaded_v001.png')
    # # image = np.asarray(image)
    # print("test pixel", image[5][5][:])
    # print("test pixel", image[65][65][:])
    # print("test pixel", image[60][60][:])
    # print("test pixel", image[55][55][:])
    # print("test pixel", image[50][50][:])
    # print("The shape of processed img", image.shape)
    # plt.imshow(image)
    # plt.show()

    print(1111111)
    shapenet_img = data_IO.imagePath2matrix('/home/zmy/Datasets/3d-r2n2-datasat/ShapeNetRendering/03001627/1a6f615e8b1b5ae4dbbc9440457e303e')
    img = shapenet_img[2]
    print("test pixel", img[5][5][:])
    print("test pixel", img[65][65][:])
    print("test pixel", img[60][60][:])
    print("test pixel", img[55][55][:])
    print("test pixel", img[50][50][:])
    print("The shape of processed img", img.shape)
    plt.imshow(img)
    plt.show()


