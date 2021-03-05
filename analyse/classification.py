
from sklearn import svm
import numpy as np
from sklearn.utils import shuffle


if __name__ == '__main__':

    latent_key = ['z', 'z_mean']
    svm_kernel = ['linear', 'rbf']
    latent_file_dirs = ['/home/zmy/Downloads/OneDrive-2021-03-04/modelnet10_image_latent_240BG.npz',
                    '/home/zmy/Downloads/OneDrive-2021-03-04/modelnet40_image_latent_240BG.npz']
    save_dir = '/home/zmy/Downloads/OneDrive-2021-03-04/'

    for latent_file in latent_file_dirs:
        for key in latent_key:
            for kernel in svm_kernel:

                latent_features_data = np.load(latent_file)
                train_latent_features, train_labels = shuffle(latent_features_data['train'+'_'+key], latent_features_data['train_labels'])
                test_latent_features, test_labels = shuffle(latent_features_data['test'+'_'+key], latent_features_data['test_labels'])
                train_latent_features[:,:] = np.nan_to_num(train_latent_features[:,:])

                classifier = svm.SVC(kernel=kernel)
                classifier.fit(train_latent_features, train_labels)

                train_accuracy = np.sum(train_labels[:]==classifier.predict(train_latent_features[:]))/len(train_labels)
                test_accuracy = np.sum(test_labels[:]==classifier.predict(test_latent_features[:]))/len(test_labels)

                dataset = latent_file.split('/')[-1].split('_')[0]
                print('-----------------------------------')
                print("Latents generated from:", dataset)
                print("Using latent information:", key)
                print("SVM kernel:", kernel)
                print("Train Accuracy:", train_accuracy)
                print("Test Accuracy:", test_accuracy)
