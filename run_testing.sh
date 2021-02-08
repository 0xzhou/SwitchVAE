
python test_MMI.py \
          --weights_file /home/zmy/GitHub/MMI-VAE/training/2021_02_06_21_43_56/weights_006-37.7518.h5 \
          --input_form 'voxel' \
          --image_data_dir /home/zmy/Datasets/3d-r2n2-datasat/ShapeNetRendering/03001627_test \
          --voxel_data_dir ./dataset/03001627_test_sub \
          --save_dir /home/zmy/GitHub/MMI-VAE/training/2021_02_06_21_43_56 \
          --save_ori True   \
          --generate_img True \
          --latent_vector_size 200

