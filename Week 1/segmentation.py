# this is an optional file for segmenting the data into a single dataframe.
# modify the code according to the need (i.e. if there is a different directory structure)

# IMPORTING MAIN LIBRARIES

import os
import glob
import numpy as np
import pandas as pd
import seaborn as sn
from PIL import Image
from collections import Counter
import matplotlib.pyplot as plt

#%% IMPORTING DATA

def loadImages(path):
    '''
        parameters
        ----------
        path : input path of the images
        
        returns
        -------
        loadedImages : list of loaded images 
    '''
    sample = []
    
    for filename in glob.glob(path):
        img = Image.open(filename,'r')
        IMG = np.array(img)
        sample.append(IMG)
        
    return sample

train_path1 = 'train/NonDemented/*.jpg' 
train_path2 = 'train/VeryMildDemented/*.jpg'
train_path3 = 'train/MildDemented/*.jpg'
train_path4 = 'train/ModerateDemented/*.jpg'

test_path1 = 'test/NonDemented/*.jpg' 
test_path2 = 'test/VeryMildDemented/*.jpg'
test_path3 = 'test/MildDemented/*.jpg'
test_path4 = 'test/ModerateDemented/*.jpg'


train_ND = loadImages(train_path1)
train_VMD = loadImages(train_path2)
train_MID = loadImages(train_path3)
train_MOD = loadImages(train_path4)

test_ND = loadImages(test_path1)
test_VMD = loadImages(test_path2)
test_MID = loadImages(test_path3)
test_MOD = loadImages(test_path4)

# CREATION OF DATASETS

df_train_ND = pd.DataFrame({'image':train_ND, 'label': 'ND'})
df_train_VMD = pd.DataFrame({'image':train_VMD, 'label': 'VMD'})
df_train_MID = pd.DataFrame({'image':train_MID, 'label': 'MID'})
df_train_MOD = pd.DataFrame({'image':train_MOD, 'label': 'MOD'})

df_test_ND = pd.DataFrame({'image':test_ND, 'label': 'ND'})
df_test_VMD = pd.DataFrame({'image':test_VMD, 'label': 'VMD'})
df_test_MID = pd.DataFrame({'image':test_MID, 'label': 'MID'})
df_test_MOD = pd.DataFrame({'image':test_MOD, 'label': 'MOD'})


train_data = [df_train_ND, df_train_VMD, df_train_MID, df_train_MOD]
train_data = pd.concat(train_data)

test_data=[df_test_ND, df_test_VMD, df_test_MID, df_test_MOD]
test_data=pd.concat(test_data)

print("train data size:",train_data.shape)
print("test data size:",test_data.shape)

#%% TRAIN LABEL SEPARATION

train_labels = train_data['label']
train_data = train_data['image']

test_labels = test_data['label']
test_data = test_data['image']


#%% LOOKING AT THE AMOUNT OF ITEMS PER CLASS 

print("Train Label:",Counter(np.array(train_labels)))
print("Test Label:",Counter(np.array(test_labels)))

