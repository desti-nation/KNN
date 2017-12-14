# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 15:06:39 2017

@author: bob

envi: win10 python3.6

knn of cs231n http://cs231n.github.io/classification/

data_set:CFAIR  http://www.cs.toronto.edu/~kriz/cifar.html
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, 
with 6000 images per class. 
There are 50000 training images and 10000 test images. 

"""
import numpy as np


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding = 'bytes')
    return dict
    
def load_CIFAR10(file, train = True):
    data = None
    labels = []
    if train == True:
        for i in range(1, 6):
            batch_data = unpickle(file + str(i))
            labels += batch_data[b'labels']
            if data is None:
                data = batch_data[b'data']
            else:
                data = np.vstack((data, batch_data[b'data']))
    else:
        batch_data = unpickle(file)
        labels = batch_data[b'labels']
        data = batch_data[b'data']
    return data, labels
    
    
class NearestNeighbor(object):
    def __init__(self):
        pass
    
    def train(self, train_data, train_labels):
        self.train_data = train_data
        self.train_labels = train_labels
    
    def predict(self, test_data):
        num_test = test_data.shape[0]
        test_pres = np.zeros(num_test)
        
        for i in range(num_test):
            #L1 distance
            #distances = np.sum(np.abs(self.train_data - test_data[i, :]), axis = 1)
            #L2 distance
            distances = np.square(np.sum(np.square(self.train_data - test_data[i, :]), axis = 1))
            min_index = np.argmin(distances)
            test_pres[i] = self.train_labels[min_index]
        return test_pres
    
if __name__ == '__main__': 
    train_file = r'.\cifar-10-batches-py\data_batch_'
    test_file = r'.\cifar-10-batches-py\test_batch'
    train_data, train_labels = load_CIFAR10(train_file, True)
    test_data, test_labels = load_CIFAR10(test_file, False)
    nn = NearestNeighbor()
    nn.train(train_data, train_labels)
    test_pres = nn.predict(test_data)
    print( 'accurancy = %f' % (np.mean(test_pres == test_labels)) )
    
    


