import os
import numpy as np
from utils import binvox_rw

if __name__ == '__main__':
    ModelNet10_ROOT = '/home/zmy/Datasets/ModelNet10/ModelNet10'
    ModelNet40_ROOT = '/home/zmy/Datasets/ModelNet40'
    ModelNet10_CLASSES = ['bathtub', 'bed', 'chair', 'desk', 'dresser',
                          'monitor', 'night_stand', 'sofa', 'table', 'toilet']

    ModelNet40_CLASSES = ['bowl', 'table', 'chair', 'vase', 'glass_box', 'bathtub', 'toilet', 'range_hood',
                          'flower_pot', 'laptop', 'plant', 'cup', 'person', 'tent', 'sofa', 'monitor', 'keyboard',
                          'desk', 'mantel', 'curtain', 'bed', 'lamp', 'bench', 'dresser', 'airplane', 'car', 'sink',
                          'night_stand', 'stool', 'door', 'guitar', 'stairs', 'radio', 'tv_stand', 'cone', 'xbox',
                          'wardrobe', 'bookshelf', 'bottle', 'piano']

    #We'll put the data into these arrays
    X = {'train': [], 'test': []}
    y = {'train': [], 'test': []}

    #Iterate over the classes and train/test directories
    for label, cl in enumerate(ModelNet10_CLASSES):
        for split in ['train', 'test']:
            examples_dir = os.path.join(ModelNet10_ROOT, cl, split)
            for example in os.listdir(examples_dir):
                if 'binvox' in example:  # Ignore OFF files
                    with open(os.path.join(examples_dir, example), 'rb') as file:
                        data = np.int32(binvox_rw.read_as_3d_array(file).data)
                        padded_data = np.pad(data, 3, 'constant')
                        X[split].append(padded_data)
                        y[split].append(label)


    #Save to a NumPy archive called "modelnet10.npz"
    np.savez_compressed('/home/zmy/Datasets/modelnet10.npz',
                        X_train=X['train'],
                        X_test=X['test'],
                        y_train=y['train'],
                        y_test=y['test'])

    # Iterate over the classes and train/test directories
    for label, cl in enumerate(ModelNet40_CLASSES):
        for split in ['train', 'test']:
            examples_dir = os.path.join(ModelNet40_ROOT, cl, split)
            for example in os.listdir(examples_dir):
                if 'binvox' in example:  # Ignore OFF files
                    with open(os.path.join(examples_dir, example), 'rb') as file:
                        data = np.int32(binvox_rw.read_as_3d_array(file).data)
                        padded_data = np.pad(data, 3, 'constant')
                        X[split].append(padded_data)
                        y[split].append(label)

    # Save to a NumPy archive called "modelnet40.npz"
    np.savez_compressed('/home/zmy/Datasets/modelnet40.npz',
                        X_train=X['train'],
                        X_test=X['test'],
                        y_train=y['train'],
                        y_test=y['test'])
