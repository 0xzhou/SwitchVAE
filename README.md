# Keras Implementation of 3D-VAE

### Packages

```markdown
python 3.6
tensorflow-gpu 1.13.1
matplotlib 3.3.2
scikit-image 0.17.2 
```

### Dataset

We use the [ShapeNetCore](https://www.shapenet.org/download/shapenetcore) dataset, in this Github repository we only provide the **volumetric data** in .binvox files of **chair class(03001627)** for training and testing, which is under `./dataset`. If you want to train with many other different type of objects, the complete dataset is available in their website.

- Volumetric data

There are 6778 elements in the chair class, the full object is available under `/dataset/03001627`. 

We also divide it into a train set and a test set, the `/dataset/03001627_train` folder consists of 5778 elements and the`/dataset/03001627_test` folder consists of 1000 elements.

In testing, if you want to generate images of the reconstructed objects at the same time, **it is recommended** to use this `/dataset/03001627_test_sub`, which consists of 100 objects from the 1000 test objects, because we use CPU to generate images, it takes about 10 minutes to test and generate images on 100 objects.

In `03001627_test_sub_pics`, you could see the ground truth image of the 100 test objects.

- Image data

If you want to train the MMI-VAE model, both volumetric data and image data are required. For the space consideration, the image dataset is available here:

**ShapeNet** rendered images http://cvgl.stanford.edu/data2/ShapeNetRendering.tgz



### Training

Set your configuration in the `run_training.py` includes hyper parameters, train dataset, **save path** (it's recommended to set it out of the repository)etc, there are the options you could choose in [arg_parser.py](https://github.com/Mingy2018/MMI-VAE/blob/main/utils/arg_parser.py)

- Volumetric Data VAE

Set training configurations in `run_training.sh` file and use `train_VAE.py` to train.

Start training:

```shell
sh run_training.sh
```

- MMI-VAE (Multi-Modal Input VAE)

The training of MMI-VAE model is still under debugging. Set training configurations in `run_training.sh` file and use `train_MMI.py` to train.

**Attention:**

For volumetric dataset, use the dataset in this repository:  `/dataset/03001627_train` .

For Image dataset: get the local path of **chair class(03001627)** after downloading and set it in `run_training.sh` like following: (The number of objects in two folders are different, don't worry, training script will match them according to their hash_id).

```sh
--binvox_dir ./dataset/03001627_train
--image_dir yourlocalpath/ShapeNetRendering/03001627
```

Start training:

```sh
sh run_training.sh
```



### Test

- Test in Volumetric Data VAE

After training the weights of models is saved as `.h5` file and used for testing, set the test configuration in `run_testing,sh`, such as **set the weights file path** and test dataset path. You could also set the path where save the reconstructed objects in `save_dir`, and choose if generate the images or not for **visualization**. Furthermore, you could also choose if save the original data for comparison by setting `save_ori`.

Start testing

```shell
sh run_testing.sh
```

- Test in MMI-VAE

We use `test_MMI.py` to test MMI model, the test model take only one input, either voxel or image, you could define the `input_from` in `run_testing.sh`. Once you select the `input_form`, also define the corresponding test dataset in `run_testing.sh`.

Start testing

```sh
sh run_testing.sh
```





