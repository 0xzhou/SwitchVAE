import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os, glob, shutil


def crop_center(im, new_height, new_width):
    height = im.shape[0]  # Get dimensions
    width = im.shape[1]
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = (width + new_width) // 2
    bottom = (height + new_height) // 2
    return im[top:bottom, left:right]


def process_folder():
    img_path1 = '/home/zmy/Downloads/betaDisentangle/374e_beta5_dim55_30'
    aim_path = '/home/zmy/Downloads/betaDisentangle/374e'
    img_files = [img for img in os.listdir(img_path1) if img.endswith('.png')]
    print(img_files)
    for img_file in img_files:
        old_file = os.path.join(img_path1, img_file)
        prefix, rest = img_file.split('_')[0], img_file.split('_')[1]
        new_name = '1' + prefix.zfill(2) + '_' + rest
        new_file = os.path.join(aim_path, new_name)
        shutil.copy(old_file, new_file)
        print(prefix, rest, new_name, new_file)


if __name__ == '__main__':

    fig = plt.figure(figsize=(60, 24), dpi=100)  # width, height

    rows = 3
    columns = 6

    img_path = '/home/zmy/Downloads/betaDisentangle/1a6f'

    # Method1: plot all cols
    row_start = 0
    col_start = 0

    # Method2: plot selected cols
    # aim_col_list = [1,3,5,7,9,10]
    # start_inx = 0
    # col_start = aim_col_list[start_inx]

    for i in range(1, columns * rows + 1):
        fig.add_subplot(rows, columns, i)
        plt.axis('off')
        name = str(row_start) + str(col_start).zfill(2) + '_gen.png'
        print("Ploting Image", name)
        I = mpimg.imread(os.path.join(img_path, name))
        I = crop_center(I, 300, 300)
        plt.imshow(I)

        # Method1:
        col_start += 1

        # Method2:
        # start_inx+=1
        # if start_inx < len(aim_col_list):
        #     col_start = aim_col_list[start_inx]

        # Method1:
        if col_start > 10:
            col_start = 0
            row_start += 1

        # Method2:
        # if start_inx >= len(aim_col_list):
        #     start_inx = 0
        #     col_start = aim_col_list[start_inx]
        #     row_start += 1

    fig.tight_layout()
    # plt.subplots_adjust(left=0.05, right=0.95,bottom=0.05,top=0.95,wspace=0.2, hspace=0.2)
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    plt.savefig('/home/zmy/Downloads/betaDisentangle/1a6f_2.png')
