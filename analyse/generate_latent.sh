python generate_latent.py \
          --weights_file /home/zmy/TrainingData/training3/2021_02_14_12_03_52/weights_120_-4.5295.h5 \
          --input_form 'image' \
          --image_data_dir /home/zmy/Datasets/3d-r2n2-datasat/03001627_processed/image/test_sub \
          --voxel_data_dir /home/zmy/Datasets/3d-r2n2-datasat/03001627_processed/voxel/test_sub \
          --save_dir /home/zmy/TrainingData/training3/2021_02_14_12_03_52 \
          --save_ori True   \
          --generate_img True \
          --latent_vector_size 128