
from sklearn import svm, manifold
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


if __name__ == '__main__':

    latent_key = ['z_mean']
    #latent_key = ['z', 'z_mean']
    #latent_key = ['z_cat']
    #svm_kernel = ['rbf', 'linear']
    svm_kernel = ['rbf']
    latent_file_dirs = ['/home/zmy/Downloads/bothTrain_onModelNet40_lr2e-5_epoch1000_annealing5cyc_lrDecayNo/modelnet10_image_BG0_latent.npz',
                    '/home/zmy/Downloads/bothTrain_onModelNet40_lr2e-5_epoch1000_annealing5cyc_lrDecayNo/modelnet40_image_BG0_latent.npz']


    for i, latent_file in enumerate(latent_file_dirs):
        for key in latent_key:
            for kernel in svm_kernel:

                latent_features_data = np.load(latent_file)
                train_latent_features, train_labels = shuffle(latent_features_data['train'+'_'+key], latent_features_data['train_labels'])
                test_latent_features, test_labels = shuffle(latent_features_data['test'+'_'+key], latent_features_data['test_labels'])
                #train_latent_features[:,:] = np.nan_to_num(train_latent_features[:,:])

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
                print('-----------------------------------')

                tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
                Y = tsne.fit_transform(train_latent_features)
                ax = plt.figure(figsize=(8, 8), facecolor='white')
                plt.scatter(Y[:, 0], Y[:, 1], c=train_labels[:], edgecolors='none',
                            cmap='terrain')
                plt.xticks([])
                plt.yticks([])
                plt.axis('tight')
                #plt.show()
                plt.savefig('/home/zmy/Downloads/bothTrain_onModelNet40_lr2e-5_epoch1000_annealing5cyc_lrDecayNo/0BG_image_latent_tsne'+str(i)+'_'+key+'_'+kernel+''+'.png')
