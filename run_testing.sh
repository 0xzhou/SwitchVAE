
python test_MMI.py \
          --weights_file /home/zmy/GitHub/MMI-VAE/training/2021_02_14_00_17_06/weights_180_-6.7309.h5 \
          --input_form 'voxel' \
          --image_data_dir /home/zmy/Datasets/3d-r2n2-datasat/02828884_processed/image/test_sub \
          --voxel_data_dir /home/zmy/Datasets/3d-r2n2-datasat/02828884_processed/voxel/test_sub \
          --save_dir /home/zmy/GitHub/MMI-VAE/training/2021_02_14_00_17_06 \
          --save_ori True   \
          --generate_img True \
          --latent_vector_size 128

