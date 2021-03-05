import numpy as np
import glob, os
from utils import binvox_rw
from utils import globals as g
from PIL import Image


def read_voxel_data(model_path):
    with open(model_path, 'rb') as f:
        model = binvox_rw.read_as_3d_array(f)
        return model.data


def write_binvox_file(pred, filename):
    with open(filename, 'w') as f:
        voxel = binvox_rw.Voxels(pred, [32, 32, 32], [0, 0, 0], 1, 'xzy')
        #voxel = binvox_rw.Voxels(pred, [32, 32, 32], [0, 0, 0], 1, 'xyz')
        binvox_rw.write(voxel, f)
        f.close()


def imagePath2matrix(imagePath, train=True):
    image_files = glob.glob(imagePath + "/*/*" + "png")
    images = np.zeros(g.VIEWS_IMAGE_SHAPE, dtype=np.float32)
    for i, image_file in enumerate(image_files):
        image = Image.open(image_file)
        image = np.asarray(image)
        image = preprocess_img(image, train)
        images[i] = image

    return images

def voxelPath2matrix(voxelPath):
    voxel_file = glob.glob(voxelPath+"/*binvox")
    voxel = read_voxel_data(voxel_file[0])
    return voxel.astype(np.float32)

def imagePathList2matrix(imagePathList, train = True):
    """
    imagePathList: ['~/Datasets/3d-r2n2-datasat/ShapeNetRendering/03001627/1a8bbf2994788e2743e99e0cae970928', ...]
    Returns: List_size x Views x Width x Height x Channels (numpy array)
    """
    size = len(imagePathList)
    images = np.zeros((size, 24) + g.IMAGE_SHAPE, dtype=np.float32)
    for i, path in enumerate(imagePathList):
        images[i] = imagePath2matrix(path, train)
    return images


def voxelPathList2matrix(voxelPathList):
    """
    voxelPathList: ['~/Datasets/3d-r2n2-datasat/ShapeNetVox32/03001627/1a8bbf2994788e2743e99e0cae970928', ...]
    Returns: List_size x 1 x 32 x 32 x 32 (numpy array)
    """
    size = len(voxelPathList)
    voxels = np.zeros((size,) + g.VOXEL_INPUT_SHAPE, dtype=np.float32)
    for i, path in enumerate(voxelPathList):
        voxels[i] = voxelPath2matrix(path)
    # voxels = 3.0 * voxels - 1.0
    return voxels


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

def preprocess_modelnet_img(img, BG_rgb=[255,255,255], aim_size=(137,137)):
    image = Image.open(img)
    image = image.resize(aim_size, Image.BICUBIC)
    image = np.asarray(image)
    scr_img = [0, 0, 0]
    r_img = image[:, :, 0].copy()
    g_img = image[:, :, 1].copy()
    b_img = image[:, :, 2].copy()
    img = r_img + g_img + b_img
    value_const = scr_img[0] + scr_img[1] + scr_img[2]
    r_img[img == value_const] = BG_rgb[0]
    g_img[img == value_const] = BG_rgb[1]
    b_img[img == value_const] = BG_rgb[2]
    img = np.dstack([r_img, g_img, b_img])
    img = img/255
    return img

def multicat_path_list(processed_dataset_path, category_list, use_mode='train'):
    voxel_path_list=[]
    image_path_list=[]
    multicat_hash_id = []
    for category in category_list:
        category_voxel_train_path = os.path.join(processed_dataset_path, category, 'voxel', use_mode )
        category_image_train_path = os.path.join(processed_dataset_path, category, 'image', use_mode)
        hash_id = os.listdir(category_voxel_train_path)
        category_hash_id = [category+'_'+id for id in hash_id]
        multicat_hash_id += category_hash_id

        voxel_path_list += [os.path.join(category_voxel_train_path,id) for id in hash_id]
        image_path_list += [os.path.join(category_image_train_path,id) for id in hash_id]

    return voxel_path_list,image_path_list,multicat_hash_id


def objectIdList2matrix(objectIdlist, dataset, train_or_test):
    size = len(objectIdlist)
    batch_image_array = np.zeros((size, 12) + g.IMAGE_SHAPE, dtype=np.float32)
    for i, id in enumerate(objectIdlist):
        category = id[:-5]  # return the category
        view_image_array = np.zeros((12,) + g.IMAGE_SHAPE, dtype=np.float32)
        for view in range(12):
            image_file = os.path.join(dataset, category, train_or_test, id + '.obj.shaded_v' + str(view + 1).zfill(3) + '.png')
            img_array=preprocess_modelnet_img(image_file, BG_rgb=[240,240,240], aim_size=(137,137))

            view_image_array[view] = img_array
        batch_image_array[i]=view_image_array

    return batch_image_array

