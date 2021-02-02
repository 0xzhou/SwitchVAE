
python train.py --model btcvae  \
                --data_dir ./dataset/03001627_train \
                --save_dir ./training \
                --num_epochs 150 \
                --batch_size 10 \
                --beta 2 \
                --latent_vector_size 128
