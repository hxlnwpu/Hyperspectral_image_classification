from PIL import Image
import spectral as sp
from scipy import misc
import scipy.io as sio
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report
import keras
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPooling2D, Flatten
from keras.models import Sequential
from keras.optimizers import SGD
from keras.models import load_model
from imblearn.over_sampling import SMOTE, ADASYN

# all data path
Indian_pines_dataset = r".\Data\indianpines_ds_raw.hdr"
Indian_pines_corrected_dataset = r".\Data\matData\Indian_pines_corrected"
Indian_pines_gt_dataset = r".\Data\indianpines_ts.tif"
Indian_pines__corrected_gt_dataset = r".\Data\matData\Indian_pines_gt"
Pavia_dataset = r".\Data\pavia_ds.hdr"
Pavia_gt_dataset = r".\Data\pavia_ts.tif"

# read all data to ndarry
Indian_pines_gt = misc.imread(Indian_pines_gt_dataset)
Indian_pines = sp.open_image(Indian_pines_dataset).load()
Indian_pines_corrected = sio.loadmat(Indian_pines_corrected_dataset)['indian_pines_corrected']
Indian_pines__corrected_gt = sio.loadmat(Indian_pines__corrected_gt_dataset)['indian_pines_gt']
PaviaU_gt = misc.imread(Pavia_gt_dataset)
PaviaU = sp.open_image(Pavia_dataset).load()
Indian_modify_gt = sio.loadmat('.\Data\indian_modify_gt.mat')['indian_pines_gt']
Pavia_modify_gt = sio.loadmat('.\Data\pavia_modify_gt.mat')['paviaU']

def get_allzero_data(dataset, labelset):
    a=np.reshape(labelset, (-1, 1))
    zero_num=np.sum(a==0)
    new_data_set = np.zeros((zero_num, dataset.shape[2]))
    new_label_set = np.zeros((zero_num, 1))
    zero_index=[]
    k=0
    for i in range(labelset.shape[0]):
        for j in range(labelset.shape[1]):
            if(labelset[i][j] == 0):
                new_data_set[k, :] = dataset[i,j,:]
                new_label_set[k, -1] = labelset[i][j]
                zero_index.append(str(i)+','+str(j))
                k = k+1
    return new_data_set,new_label_set,zero_index

new_data_set,new_label_set,zero_index=get_allzero_data(Indian_pines, Indian_pines_gt)
print(zero_index)