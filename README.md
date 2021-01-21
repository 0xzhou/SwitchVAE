# Keras Implementation of 3D-VAE

### Packages

```
python 3.6
tensorflow-gpu 1.13.1
matplotlib 3.3.2
scikit-image 0.17.2 
```

### Dataset

We use the [ShapeNetCore v2 (Fall 2016)](https://www.shapenet.org/download/shapenetcore) dataset, in the Github project we only provide the **chair class(03001627)** in the dataset for training and testing. If you want to train with many other different type of objects, the complete dataset is available in link.

There is 6778 elements in the chair class, the full object is available under `/dataset/03001627`. 

We also divide it into a train set and a test set, the `/dataset/03001627_train` folder consists of 5778 elements and the`/dataset/03001627_test` folder consists of 1000 elements.

 If you want to generate object 3d images at the same time, **it is recommended** to use this `/dataset/03001627_test_img`, which consists of 100 objects from the 1000 test objects, because we use CPU to generate 3d images, it takes about 10 minutes to test and generate images on 100 objects.

### Training

Set your configuration in the `run_training.py` includes hyper parameters, train dataset, **save path** (it's recommended to set it out of the repository)etc, there are the options you could choose in [arg_parser.py]()

Start training:

```shell
sh run_training.sh
```

### Test

After training the weights of models is saved as `.h5` file and used for testing, set the test configuration in `run_testing,sh`, such as **set the weights file path** and test dataset path. You could also set the path where save the reconstructed objects in `save_dir`, and choose if generate the images or not for **visualization**. Furthermore, you could also choose if save the original data for comparison by setting `save_ori`.

Start testing

```shell
sh run_testing.sh
```







