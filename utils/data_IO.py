import numpy as np
import glob, os
import scipy.ndimage as nd
from utils import binvox_rw
from utils import globals as g
# import globals as g
from PIL import Image


def read_voxel_data(model_path):
    with open(model_path, 'rb') as f:
        model = binvox_rw.read_as_3d_array(f)
        return model.data


def write_binvox_file(pred, filename):
    with open(filename, 'w') as f:
        voxel = binvox_rw.Voxels(pred, [32, 32, 32], [0, 0, 0], 1, 'xzy')
        # voxel = binvox_rw.Voxels(pred, [32, 32, 32], [0, 0, 0], 1, 'xyz')
        binvox_rw.write(voxel, f)
        f.close()


def imagepath2matrix(image_dataset_path, single_image_shape=(137, 137, 3)):
    image_files = glob.glob(image_dataset_path + "/*/*" + "png")
    object_hash = os.listdir(image_dataset_path)
    images = np.zeros((24,) + single_image_shape, dtype=np.float32)
    for i, image_file in enumerate(image_files):
        image = Image.open(image_file)
        image = np.asarray(image)
        image = preprocess_img(image)
        images[i] = image

    return images

def image_folder_list2matrix(image_folder_list):
    """
    Transform image path named by hash to numpy array
    Args:
        image_folder_list:  ['~/Datasets/3d-r2n2-datasat/ShapeNetRendering/03001627/1a8bbf2994788e2743e99e0cae970928', ...]

    Returns: 5 dimensional numpy array
    """
    size = len(image_folder_list)
    images_one_batch = np.zeros((size, 24) + g.IMAGE_SHAPE, dtype=np.float32)
    for i, element in enumerate(image_folder_list):
        images_one_batch[i] = imagepath2matrix(element)
    return images_one_batch


def voxel_folder_list2matrix(voxel_file_list, padding=False):
    """
    Transform a list of voxel folders to numpy array, this function is used in train_MMI.py
    Args:
        voxel_file_list: ['~/Datasets/3d-r2n2-datasat/ShapeNetVox32/03001627/1a8bbf2994788e2743e99e0cae970928', ...]
        padding:

    Returns: 4 dimensional numpy array
    """
    size = len(voxel_file_list)
    voxel_one_batch = np.zeros((size,) + g.VOXEL_INPUT_SHAPE, dtype=np.float32)
    for i, element in enumerate(voxel_file_list):
        model = glob.glob(element + '/*')
        model = read_voxel_data(model[0])
        voxel_one_batch[i] = model.astype(np.float32)

    #voxel_one_batch = 3.0 * voxel_one_batch - 1.0
    return voxel_one_batch


def add_random_color_background(im, color_range):
    r, g, b = [np.random.randint(color_range[i][0], color_range[i][1] + 1) for i in range(3)]

    if isinstance(im, Image.Image):
        im = np.array(im)

    if im.shape[2] > 3:
        # If the image has the alpha channel, add the background
        alpha = (np.expand_dims(im[:, :, 3], axis=2) == 0).astype(np.float)
        im = im[:, :, :3]
        bg_color = np.array([[[r, g, b]]])
        im = alpha * bg_color + (1 - alpha) * im

    return im


def preprocess_img(im, train=True):
    # add random background
    im = add_random_color_background(im, g.TRAIN_NO_BG_COLOR_RANGE if train else g.TEST_NO_BG_COLOR_RANGE)

    # # If the image has alpha channel, remove it.
    # im_rgb = np.array(im)[:, :, :3].astype(np.float32)
    # if train:
    #     t_im = image_transform(im_rgb, cfg.TRAIN.PAD_X, cfg.TRAIN.PAD_Y)
    # else:
    #     t_im = crop_center(im_rgb, cfg.CONST.IMG_H, cfg.CONST.IMG_W)

    # Scale image
    im = im / 255.
    return im


def test():
    import matplotlib.pyplot as plt
    im = Image.open(
        '/home/zmy/Datasets/3d-r2n2-datasat/04379243_processed/image/test/1a9bb6a5cf448d75e6fd255f2d77a585/rendering/00.png')
    im = np.asarray(im)
    print("The shape of image array", im.shape)
    imt = preprocess_img(im)
    print("The shape of processed image array", imt.shape)
    plt.imshow(imt)
    plt.show()


if __name__ == '__main__':
    test()
