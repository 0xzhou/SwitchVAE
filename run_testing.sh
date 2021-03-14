
python test_VAE.py  \
          --weights_file /home/zmy/TrainingData/Experiments-BCE/2021_02_28_22_26_43/end_weights.h5 \
          --weights_dir /home/zmy/Downloads/OneDrive-2021-03-04_2 \
          --input_form 'image' \
          --image_data_dir /home/zmy/Datasets/3d-r2n2-datasat/03001627_processed/image/test_sub \
          --voxel_data_dir /home/zmy/Datasets/3d-r2n2-datasat/03001627_processed/voxel/test_sub \
          --modelnet_voxel_dataset /home/zmy/Datasets/ModelNet40 \
          --modelnet_image_dataset /home/zmy/Datasets/ModelNet40_images/modelnet40_images_new_12x \
          --save_dir /home/zmy/Downloads/OneDrive-2021-03-04_2 \
          --save_ori 1   \
          --generate_img 1 \
          --save_bin 1 \
          --latent_vector_size 128

