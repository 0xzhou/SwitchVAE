
python train_SwitchVAE.py --loss vae  \
                --binvox_dir /home/zmy/mmi_dataset/3d-r2n2-datasat_processed/03001627/voxel/train \
                --image_dir /home/zmy/mmi_dataset/3d-r2n2-datasat/03001627_processed/image/train \
                --processed_dataset /home/zmy/mmi_dataset/3d-r2n2-datasat_processed \
                --modelnet_voxel_dataset /home/zmy/mmi_dataset/ModelNet10 \
                --modelnet_image_dataset /home/zmy/mmi_dataset/ModelNet40_images/modelnet40_images_new_12x \
                --save_dir /home/zmy/TrainingData \
                --num_epochs 400 \
                --batch_size 8 \
                --initial_learning_rate 1e-4 \
                --beta 3 \
                --latent_vector_size 128

