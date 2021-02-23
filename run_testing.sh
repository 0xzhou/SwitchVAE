
python test_MMI.py \
          --weights_file /home/zmy/TrainingData/2021.2.17/2021_02_23_19_26_44/end_weights.h5 \
          --input_form 'voxel' \
          --image_data_dir /home/zmy/Datasets/3d-r2n2-datasat/03001627_processed/image/test_sub \
          --voxel_data_dir /home/zmy/Datasets/3d-r2n2-datasat/03001627_processed/voxel/test_sub \
          --save_dir /home/zmy/TrainingData/2021.2.17/2021_02_23_19_26_44/weights_end \
          --save_ori True   \
          --generate_img True \
          --latent_vector_size 128

