
python train_MMI.py --loss vae  \
                --binvox_dir ./dataset/03001627_train \
                --image_dir /home/zmy/Datasets/3d-r2n2-datasat/ShapeNetRendering/03001627 \
                --save_dir ./training \
                --num_epochs 6 \
                --batch_size 2 \
                --beta 1 \
                --latent_vector_size 128
