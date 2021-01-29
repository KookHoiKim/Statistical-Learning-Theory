import cv2
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
from tqdm import tqdm

#create lists to save the labels (the name of the shape)
train_labels, train_images = [],[]
#train_dir_linux = './data'
current_dir = os.path.dirname(os.path.abspath(__file__))
train_dir = 'data'
shape_list = ['dog', 'cat']

# Preprocessing
# This is option. you may change the given preprocessing
def preprocess(images, labels, is_train=True):
    """You can make your preprocessing code in this function.
    Input: images and labels
    return: preprocessed images and labels
    """
    dataDim = np.prod(images[0].shape)
    images = np.array(images)
    images = images.astype('float32')
    images /=255
    images = torch.from_numpy(images)

    return images, labels

# problem 1: Neural Network 
# Implement HERE!
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        in_dim = 300*300
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(in_dim, in_dim),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(in_dim, 2)
        )
    def forward(self, image, labels=None):
        image = torch.flatten(image)
        ret = self.classifier(image)
        return ret

if __name__ == '__main__':
    #iterate through each shape
    
    for shape in shape_list:
        print('Getting data for: ', shape)
        for file_name in os.listdir(os.path.join(current_dir, train_dir, shape)):
            train_images.append(cv2.imread(os.path.join(current_dir, train_dir, shape, file_name), 0))
            #add an integer to the labels list
            train_labels.append(shape_list.index(shape))

    print('Number of training images: ', len(train_images))
    # Preprocess (your own function)
    train_images, train_labels = preprocess(train_images, train_labels)
    
    # Make a classifier (your own function)
    model = Network()
    
    lr = 0.001
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    epochs = 10
    # Problem 2: Train your model Here!
    # Implement HERE!
    assert len(train_images) == len(train_labels), 'mismatch of length between images and labels'

    for epoch in epochs:
        
        for i in range(len(train_label)):

            image = train_images[i]
            label = train_labels[i]
            
            output = model(image)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print('Epoch {}/{}, Step {}/{}, Loss: {:.4f}'
                      .format(epoch+1, epochs, i+1, len(train_label), loss.item()))

    # Calculate accuracy
    pred_acc = np.sum(pred_labels==train_labels)/len(train_labels)*100
    print("Accuracy = {}".format(pred_acc))


    # Problem 3: Test your trained model on test dataset
    # You may modify the test code for your own
    # Check the results on test dataset.
    test_dir = './test'
    test_labels, test_images = [], []
    for shape in shape_list:
        print('Getting data for: ', shape)
        for file_name in os.listdir(os.path.join(test_dir,shape)):
            test_images.append(cv2.imread(os.path.join(test_dir,shape,file_name), 0))
            #add an integer to the labels list
            test_labels.append(shape_list.index(shape))

    print('Number of test images: ', len(test_images))

    test_images, test_labels = preprocess(test_images, test_labels, False)
    _,pred_labels = torch.max(model(test_images),dim=1)
    pred_labels = pred_labels.numpy()
    pred_acc = np.sum(pred_labels==test_labels)/len(test_labels)*100
    print("Test Accuracy = {}".format(pred_acc))
