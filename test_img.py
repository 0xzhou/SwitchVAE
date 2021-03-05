
import matplotlib.pyplot as plt
from utils import data_IO
import glob


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


    _img=data_IO.preprocess_modelnet_img(img='/home/zmy/Datasets/ModelNet40_images/modelnet40_images_new_12x/airplane/train/airplane_0001.obj.shaded_v001.png',
                                    BG_rgb=[255, 255, 255],
                                    aim_size=(137, 137))

    print("The shape of processed img", _img.shape)
    plt.imshow(_img)
    plt.show()