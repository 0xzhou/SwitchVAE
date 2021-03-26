python generate_latent.py \
          --latent_vector_size 128 \
          --weights_dir /home/zmy/Downloads/bothTrain_onChair_uniLoss1_switchImg08_epoch1000 \
          --input_form 'image' \
          --save_dir /home/zmy/Downloads/bothTrain_onChair_uniLoss1_switchImg08_epoch1000 \
          --dataset 'modelnet' \
          --voxel_npz /home/zmy/Datasets/modelnet10.npz \
          --image_npz /home/zmy/Datasets/modelnet10_image.npz \
          --image_data_dir /home/zmy/Datasets/3d-r2n2-datasat/03001627_processed/image/test_sub \
          --voxel_data_dir /home/zmy/Datasets/3d-r2n2-datasat/ShapeNetVox32/04379243 \
          --save_ori 0   \
          --generate_img 0 \

