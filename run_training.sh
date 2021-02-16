
python train_MMI.py --loss btcvae  \
                --binvox_dir /home/zmy/Datasets/3d-r2n2-datasat/03001627_processed/voxel/train \
                --image_dir /home/zmy/Datasets/3d-r2n2-datasat/03001627_processed/image/train \
                --save_dir ./training \
                --num_epochs 150 \
                --batch_size 6 \
                --base_learning_rate 0.002 \
                --beta 2 \
                --latent_vector_size 128
