
python test_VAE.py  \
          --weights_file /home/zmy/TrainingData/Experiments-BCE/2021_02_28_22_26_43/end_weights.h5 \
          --weights_dir /home/zmy/TrainingData/Experiments-BCE/2021_02_28_22_26_43/ \
          --input_form 'both' \
          --image_data_dir /home/zmy/Datasets/3d-r2n2-datasat/03001627_processed/image/test_sub \
          --voxel_data_dir /home/zmy/Datasets/3d-r2n2-datasat/03001627_processed/voxel/test/ \
          --save_dir /home/zmy/TrainingData/Experiments-BCE/2021_02_28_22_26_43 \
          --save_ori 0   \
          --generate_img 0 \
          --save_bin 0 \
          --latent_vector_size 128

