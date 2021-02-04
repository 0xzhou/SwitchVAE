
python train.py --loss bce  \
                --binvox_dir ./dataset/03001627_train \
                --image_dir /home/zmy/Datasets/3d-r2n2-datasat/ShapeNetRendering/03001627 \
                --save_dir ./training \
                --num_epochs 150 \
                --batch_size 10 \
                --beta 1 \
                --latent_vector_size 128
