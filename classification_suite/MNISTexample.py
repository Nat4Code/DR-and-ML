import math
#import faiss
import struct
import random
import numpy as np
import matplotlib.pyplot as plt
from array import array
from os.path  import join
from collections import Counter
from multiprocessing import Pool
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from concurrent.futures import ThreadPoolExecutor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

############################
# MNIST Data Loader Class: #
############################
class MnistDataloader(object):
    def __init__(self, training_images_filepath,training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath

    def read_images_labels(self, images_filepath, labels_filepath):        
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img            
        
        return images, labels
    
    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train),(x_test, y_test)

################

########################################
# Set file paths to correct locations: #
########################################
input_path = 'data'
training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

#######################
# Load MINST dataset: #
#######################
mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, 
                                   test_images_filepath, test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

##############################################
# Show some random training and test images: #
##############################################
images_2_show = []
titles_2_show = []
for i in range(0, 10):
    r = random.randint(1, 60000)
    images_2_show.append(x_train[r])
    titles_2_show.append('training image [' + str(r) + '] = ' + str(y_train[r]))    

for i in range(0, 5):
    r = random.randint(1, 10000)
    images_2_show.append(x_test[r])        
    titles_2_show.append('test image [' + str(r) + '] = ' + str(y_test[r]))    

# Show Images: #
################
def show_images(images, title_texts):
    '''Shows a list of images with their related titles'''
    cols = 5
    rows = int(len(images)/cols) + 1
    plt.figure(figsize=(30,20))
    index = 1    
    for x in zip(images, title_texts):        
        image = x[0]        
        title_text = x[1]
        plt.subplot(rows, cols, index)        
        plt.imshow(image, cmap=plt.cm.gray)
        if (title_text != ''):
            plt.title(title_text, fontsize = 15);        
        index += 1

#####################
# Standardize Data: #
#####################
def standardize(data):
    '''Standardizes our data and returns standardize dataset'''
    std_imgs = []

    for image in data:
        mean = np.mean(image, axis=0)
        std = np.std(image, axis=0)
        std[std == 0] = 1

        std_data = (image-mean)/std
        std_imgs.append(std_data)

    return std_imgs

Xn_train = standardize(x_train)
Xn_test = standardize(x_test)

################
# Perform FLD: #
#################################################
Xn_tr = [np.array(m).flatten() for m in Xn_train]
Xn_te = [np.array(n).flatten() for n in Xn_test]

# We use FLD to reduce a dimension:
FDA = LDA(n_components=2)
Xn_fda = FDA.fit_transform(Xn_tr, y_train)
#Xn_fda_te = FDA.fit_transform(Xn_te, y_test)

# We use PCA to reduce the rest:
#pca = PCA(n_components=2)
#X_fda = pca.fit_transform(Xn_tr_red)
#X_fda_te = pca.fit_transform(Xn_te_red)

# Construct the Scatter plot for FLD:
scatter = plt.scatter(X_fda[:, 0], X_fda[:, 1], 
                      c=y_train, alpha=0.6, marker="+")
#for x, y, y_tr, mk in zip(X_pca[:,0], X_pca[:,1], y_train, markerList):
    #plt.plot(x, y, alpha=0.6, marker=mk)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('FLD: Dimensional Reduction')
plt.colorbar(scatter, label='Digit Prediction')
plt.savefig('FLD.png')