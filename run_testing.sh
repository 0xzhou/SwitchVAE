
python test_MMI.py \
          --weights_file /home/zmy/TrainingData/2021.2.17/2021_02_19_00_01_31/weights_090_365.4811.h5 \
          --input_form 'image' \
          --image_data_dir /home/zmy/Datasets/3d-r2n2-datasat/03001627_processed/image/test_sub \
          --voxel_data_dir /home/zmy/Datasets/3d-r2n2-datasat/03001627_processed/voxel/test_sub \
          --save_dir /home/zmy/TrainingData/2021.2.17/2021_02_19_00_01_31/weights_90 \
          --save_ori True   \
          --generate_img True \
          --latent_vector_size 128

