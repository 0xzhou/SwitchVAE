python generate_latent.py \
          --latent_vector_size 128 \
          --weights_dir /home/zmy/Downloads/bothTrain_onModelNet40_lr2e-5_epoch1000_annealing5cyc_lrDecayNo \
          --input_form 'image' \
          --save_dir /home/zmy/Downloads/bothTrain_onModelNet40_lr2e-5_epoch1000_annealing5cyc_lrDecayNo \
          --dataset 'modelnet' \
          --voxel_npz /home/zmy/mmi_dataset/modelnet10.npz \
          --image_npz /home/zmy/mmi_dataset/modelnet10_image.npz \
          --image_data_dir /home/zmy/Datasets/3d-r2n2-datasat/03001627_processed/image/test_sub \
          --voxel_data_dir /home/zmy/Datasets/3d-r2n2-datasat/ShapeNetVox32/04379243 \
          --save_ori 0   \
          --generate_img 0 \

