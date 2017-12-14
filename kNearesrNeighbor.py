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
        for i in range(1, 2):
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
    
    def predict(self, k, test_data, test_labels):
        num_test = test_data.shape[0]
        test_pres = np.zeros(num_test)
        
        for i in range(num_test):
            #L1 distance
            #distances = np.sum(np.abs(self.train_data - test_data[i, :]), axis = 1)
            #L2 distance
            distances = np.sqrt(np.sum(np.square(self.train_data - test_data[i, :]), axis = 1))
            #distances = distances.reshape((1, -1))
            sorted_indexs = np.argsort(distances)
            class_count = np.zeros(10)
            for j in range(k):
                this_label = train_labels[sorted_indexs[j]]
                class_count[this_label] += 1
            test_pres[i] = np.argmax(class_count)
            #print('test item = %d, label = %d, pre = %d'%(i, test_labels[i], test_pres[i]))
        return test_pres
    
if __name__ == '__main__': 
    input_file = r'.\cifar-10-batches-py\data_batch_'
    test_file = r'.\cifar-10-batches-py\test_batch'
    
    input_data, input_labels = load_CIFAR10(input_file, True)
    train_data = input_data[:-1000, :] #take the 49000 data as train data
    val_data = input_data[-1000:, :] # take the last 1000 data as the validation data
    train_labels = input_labels[:-1000]
    val_labels = input_labels[-1000:]

    print(train_data.shape, len(train_labels))
    
    test_data, test_labels = load_CIFAR10(test_file, False)
    
    '''
    validation_accuracies = []
    for k in [1, 3, 5, 10, 20, 50, 100]:
        nn = NearestNeighbor()
        nn.train(train_data, train_labels)
        val_pres = nn.predict(k, val_data, val_labels)
        acc = np.mean(val_pres == val_labels)
        validation_accuracies.append((k, acc))
        print( 'k = %d, accurancy = %f' % (k, acc) )
    '''
    nn = NearestNeighbor()
    nn.train(train_data, train_labels)
    test_pres = nn.predict(100, test_data, test_labels)
    acc = np.mean(test_pres == test_labels)
    print( 'accurancy = %f' % (acc) )
    
    
'''
runfile('D:/cs231n/KNN/kNearesrNeighbor.py', wdir='D:/cs231n/KNN')
(9000, 3072) 9000
k = 1, accurancy = 0.218000
k = 3, accurancy = 0.200000
k = 5, accurancy = 0.208000
k = 10, accurancy = 0.212000
k = 20, accurancy = 0.209000
k = 50, accurancy = 0.227000
k = 100, accurancy = 0.228000
'''


