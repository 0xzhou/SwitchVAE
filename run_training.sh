
python train.py --model bvae  \
                --data_dir ./dataset/03001627_train \
                --save_dir ./training \
                --num_epochs 150 \
                --batch_size 10 \
                --beta 1 \
                --latent_vector_size 128
