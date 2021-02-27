
python test_decoder.py \
          --weights_file /home/zmy/Desktop/test/end_weights_wu.h5 \
          --input_form 'image' \
          --image_data_dir /home/zmy/Datasets/3d-r2n2-datasat/03001627_processed/image/test_sub \
          --voxel_data_dir /home/zmy/Datasets/3d-r2n2-datasat/03001627_processed/voxel/test_sub \
          --save_dir /home/zmy/Desktop/test \
          --save_ori True   \
          --generate_img True \
          --latent_vector_size 128

