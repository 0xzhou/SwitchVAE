import os, shutil
from utils import save_train

if __name__ == '__main__':

    # voxel_path = '/home/zmy/Datasets/3d-r2n2-datasat/03001627_processed/image/test'
    # to_copy_hash_list = os.listdir('/home/zmy/Datasets/3d-r2n2-datasat/03001627_processed/voxel/test_select_visualization')
    # save_path = '/home/zmy/Datasets/3d-r2n2-datasat/03001627_processed/image/test_select'
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    #
    # for hash in to_copy_hash_list:
    #     hash = hash[:-4]
    #     print("Copy folder:", hash)
    #     ori_file_path = os.path.join(voxel_path,hash)
    #     file_save_to = os.path.join(save_path,hash)
    #     shutil.copytree(ori_file_path,file_save_to)
    save_train.save_config_pro('./')


