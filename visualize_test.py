import os
from utils import save_volume

def main():
    voxel_dataset_path = '/home/zmy/Datasets/3d-r2n2-datasat/03001627_processed/voxel/test'
    save_path ='/home/zmy/Datasets/3d-r2n2-datasat/03001627_processed/voxel/test_visualization'
    os.makedirs(save_path)

    voxel_file_names = os.listdir(voxel_dataset_path)

    for i, hash in enumerate(voxel_file_names):
        voxel_file = os.path.join(voxel_dataset_path,hash,'model.binvox')
        save_volume.binvox2image_2(voxel_file,hash, save_path)


if __name__ == '__main__':
    main()
