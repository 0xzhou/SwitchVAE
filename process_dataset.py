

import os, shutil, sys
from utils import arg_parser, save_volume

def create_dataset(category, voxel_dataset_path, image_dataset_path, split_scale, save_path, sub_num):
    '''
    The script will create subdate set for training and testing from volumentric data and image data.

    Args:
        category: the category in the dataset, normally is a 8-digits number
        split_scale: (number of train samples, number of test samples), like (0.8:0.2)
        save_path: function will generate the datasets under the save_path
        sub_num: the number of objects in test_sub, for the visulization when test model
    Returns: 0
    '''

    save_path = os.path.join(save_path, category+'_processed')
    os.makedirs(save_path)

    object_number = len(os.listdir(voxel_dataset_path))
    train_dataset_size = int(object_number * split_scale[0])
    voxel_file_names = os.listdir(voxel_dataset_path)

    #random.shuffle(voxel_file_names)

    voxel_train_names = voxel_file_names[:train_dataset_size]
    voxel_test_names = voxel_file_names[train_dataset_size:]
    voxel_test_sub_names = voxel_file_names[train_dataset_size: train_dataset_size+sub_num]
    vol_train_dataset_save_path = os.path.join(save_path, 'voxel', 'train')
    vol_test_dataset_save_path = os.path.join(save_path, 'voxel', 'test')
    vol_test_sub_dataset_save_path = os.path.join(save_path, 'voxel', 'test_sub')
    vol_test_sub_dataset_visualized_save_path = os.path.join(save_path, 'voxel', 'test_sub_visulization')
    os.makedirs(vol_test_sub_dataset_visualized_save_path)

    img_train_dataset_save_path = os.path.join(save_path, 'image','train')
    img_test_dataset_save_path = os.path.join(save_path, 'image','test')
    img_test_sub_dataset_save_path = os.path.join(save_path, 'image','test_sub')

    copy_object_data(voxel_train_names, voxel_dataset_path, image_dataset_path,
                     vol_train_dataset_save_path, img_train_dataset_save_path)

    copy_object_data(voxel_test_names, voxel_dataset_path, image_dataset_path,
                     vol_test_dataset_save_path, img_test_dataset_save_path)

    copy_object_data(voxel_test_sub_names, voxel_dataset_path, image_dataset_path,
                     vol_test_sub_dataset_save_path, img_test_sub_dataset_save_path)

    # Visulize the test_sub dataset
    for id in voxel_test_sub_names:
        voxel_file = os.path.join(vol_test_sub_dataset_save_path,id,'model.binvox')
        shutil.copy2(voxel_file, vol_test_sub_dataset_visualized_save_path)
        voxel_file = os.path.join(vol_test_sub_dataset_visualized_save_path,'model.binvox')
        new_name = os.path.join(vol_test_sub_dataset_visualized_save_path,id+'.binvox')
        os.rename(voxel_file, new_name)

        save_volume.binvox2image_2(new_name, id, vol_test_sub_dataset_visualized_save_path)


def copy_object_data(id_list, vol_path, img_path, vol_save_path, img_save_path):

    for id in id_list:
        vol_original_path = os.path.join(vol_path, id)
        img_original_path = os.path.join(img_path, id)
        vol_save_to = os.path.join(vol_save_path, id)
        img_save_to = os.path.join(img_save_path, id)
        shutil.copytree(vol_original_path, vol_save_to)
        shutil.copytree(img_original_path, img_save_to)

def main(args):
    split_scale = tuple(args.split_scale)
    create_dataset(args.category, args.voxel_dataset_path, args.image_dataset_path,
                   split_scale, args.save_path, args.sub_num)

if __name__ == '__main__':
    main(arg_parser.parse_dataset_arguments(sys.argv[1:]))




