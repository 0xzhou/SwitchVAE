import os
import numpy as np
from utils import binvox_rw
import glob

if __name__ == '__main__':

    ModelNet10_ROOT = '/home/zmy/Datasets/ModelNet10/ModelNet10'
    ModelNet40_ROOT = '/home/zmy/Datasets/ModelNet40'
    image_ROOT = '/home/zmy/mmi_dataset/ModelNet40_images/modelnet40_images_new_12x'
    ModelNet10_CLASSES = ['bathtub', 'bed', 'chair', 'desk', 'dresser',
                          'monitor', 'night_stand', 'sofa', 'table', 'toilet']
    ModelNet40_CLASSES = [ 'airplane', 'bowl', 'table', 'chair', 'vase', 'glass_box', 'bathtub', 'toilet', 'range_hood',
                          'flower_pot', 'laptop', 'plant', 'cup', 'person', 'tent', 'sofa', 'monitor', 'keyboard',
                          'desk', 'mantel', 'curtain', 'bed', 'lamp', 'bench', 'dresser','car', 'sink',
                          'night_stand', 'stool', 'door', 'guitar', 'stairs', 'radio', 'tv_stand', 'cone', 'xbox',
                          'wardrobe', 'bookshelf', 'bottle', 'piano']


    # ---------------Block1---------------------------


    # X = {'train': [], 'test': []}
    # y = {'train': [], 'test': []}
    #
    # for label, cl in enumerate(ModelNet10_CLASSES):
    #     for split in ['train', 'test']:
    #         examples_dir = os.path.join(ModelNet10_ROOT, cl, split)
    #         for example in os.listdir(examples_dir):
    #             if 'binvox' in example:  # Ignore OFF files
    #                 with open(os.path.join(examples_dir, example), 'rb') as file:
    #                     data = np.int32(binvox_rw.read_as_3d_array(file).data)
    #                     X[split].append(data)
    #                     y[split].append(label)
    # X['train']=np.expand_dims(X['train'], axis=1)
    # X['test'] = np.expand_dims(X['test'], axis=1)
    #
    # np.savez_compressed('/home/zmy/Datasets/modelnet10.npz',
    #                     X_train=X['train'],
    #                     X_test=X['test'],
    #                     y_train=y['train'],
    #                      y_test=y['test'])
    #----------------------------------------------------


    # -----------------------Block2--------------------------
    # X = {'train': [], 'test': []}
    # y = {'train': [], 'test': []}
    # for label, cl in enumerate(ModelNet40_CLASSES):
    #     for split in ['train', 'test']:
    #         examples_dir = os.path.join(ModelNet40_ROOT, cl, split)
    #         for example in os.listdir(examples_dir):
    #             if 'binvox' in example:  # Ignore OFF files
    #                 with open(os.path.join(examples_dir, example), 'rb') as file:
    #                     data = np.int32(binvox_rw.read_as_3d_array(file).data)
    #                     X[split].append(data)
    #                     y[split].append(label)
    #
    # X['train'] = np.expand_dims(X['train'], axis=1)
    # X['test'] = np.expand_dims(X['test'], axis=1)
    # np.savez_compressed('/home/zmy/Datasets/modelnet40.npz',
    #                     X_train=X['train'],
    #                     X_test=X['test'],
    #                     y_train=y['train'],
    #                     y_test=y['test'])
    #-------------------------------------------------------


    #-------------------------------------------------------------

    # X = {'train': [], 'test': []}
    # y = {'train': [], 'test': []}
    #
    # for label, cl in enumerate(ModelNet10_CLASSES):
    #     for split in ['train', 'test']:
    #         examples_dir = os.path.join(image_ROOT, cl, split)
    #         file_list = os.listdir(examples_dir)
    #         id_list = [name.split('.')[0] for name in file_list if not name.startswith('.')]
    #         unique_id_list = list(set(id_list))
    #         X[split]+= unique_id_list
    #         y[split]+= [label] * len(unique_id_list)
    #
    # np.savez_compressed('/home/zmy/mmi_dataset/modelnet10_image.npz',
    #                     X_train=X['train'],
    #                     X_test=X['test'],
    #                     y_train=y['train'],
    #                     y_test=y['test'])
    #-------------------------------------------------------------------------------------

    #-------------------------------------------------------------

    X = {'train': [], 'test': []}
    y = {'train': [], 'test': []}

    for label, cl in enumerate(ModelNet40_CLASSES):
        for split in ['train', 'test']:
            examples_dir = os.path.join(image_ROOT, cl, split)
            file_list = os.listdir(examples_dir)
            id_list = [name.split('.')[0] for name in file_list if not name.startswith('.')]
            unique_id_list = list(set(id_list))
            X[split]+= unique_id_list
            y[split]+= [label] * len(unique_id_list)

    np.savez_compressed('/home/zmy/mmi_dataset/modelnet40_image.npz',
                        X_train=X['train'],
                        X_test=X['test'],
                        y_train=y['train'],
                        y_test=y['test'])
    #-------------------------------------------------------------------------------------
