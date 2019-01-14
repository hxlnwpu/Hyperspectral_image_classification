from PIL import Image
import spectral as sp
import scipy.io as sio
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from keras.layers import Dense, Dropout, Activation,Conv2D,MaxPooling2D,Flatten
from keras.models import Sequential
from keras.optimizers import SGD
from keras.models import load_model



#all data path
Indian_pines_dataset=r"C:\Workspace\ML\Hyperspectral_image_classification\Data\Indian_pines"   #145*145*220
Indian_pines_corrected_dataset=r"C:\Workspace\ML\Hyperspectral_image_classification\Data\Indian_pines_corrected"  #145*145*200  
Indian_pines_gt_dataset=r"C:\Workspace\ML\Hyperspectral_image_classification\Data\Indian_pines_gt" #145*145
Pavia_dataset=r"C:\Workspace\ML\Hyperspectral_image_classification\Data\Pavia"   #1096*715*102
Pavia_gt_dataset=r"C:\Workspace\ML\Hyperspectral_image_classification\Data\Pavia_gt"   #1096*715
PaviaU_dataset=r"C:\Workspace\ML\Hyperspectral_image_classification\Data\PaviaU"      #610*340*103
PaviaU_gt_dataset=r"C:\Workspace\ML\Hyperspectral_image_classification\Data\PaviaU_gt"   #610*340

#read all data to ndarry
Indian_pines=sio.loadmat(Indian_pines_dataset)['indian_pines']
Indian_pines_corrected=sio.loadmat(Indian_pines_corrected_dataset)['indian_pines_corrected']
Indian_pines_gt=sio.loadmat(Indian_pines_gt_dataset)['indian_pines_gt']
Pavia=sio.loadmat(Pavia_dataset)['pavia']
Pavia_gt=sio.loadmat(Pavia_gt_dataset)['pavia_gt']
PaviaU=sio.loadmat(PaviaU_dataset)['paviaU']
PaviaU_gt=sio.loadmat(PaviaU_gt_dataset)['paviaU_gt']

def get_class_num(dataset):
    dict_k = {}
    for i in range(dataset.shape[0]):
        for j in range(dataset.shape[1]):
            #if output_image[i][j] in [m for m in range(1,17)]:
            if dataset[i][j] in [1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16]:
                if dataset[i][j] not in dict_k:
                    dict_k[dataset[i][j]]=0
                dict_k[dataset[i][j]] +=1
    return dict_k,len(np.unique(dataset))

#get all nonzero sample and label
def get_nozero_data(dataset,labelset):
    nozero_num=len(labelset.nonzero()[0])
    k=0
    new_data_set=np.zeros((nozero_num,dataset.shape[2]))
    new_label_set=np.zeros((nozero_num,1))
    for i in range(labelset.shape[0]):
        for j in range(labelset.shape[1]):
            if(labelset[i][j]!=0):
                new_data_set[k,:]=dataset[i][j]
                new_label_set[k,-1]=labelset[i][j]
                k=k+1
    return new_data_set,new_label_set

#扩展数据，便于划分patch
def extend_dataset(dataset, windowSize=5):
    new_dataset = np.zeros((dataset.shape[0] + windowSize-1, dataset.shape[1] + windowSize-1, dataset.shape[2]))
    offset=int((windowSize - 1) / 2)
    new_dataset[offset:dataset.shape[0] + offset, offset:dataset.shape[1] + offset, :] = dataset
    return new_dataset

#创建patch：每一个patch都是一个windowSize*windowSize*特征向量长度的数组。原数据每一个元素分别成为patch中心,
def create_patche(dataset, windowSize=5):
    offset = int((windowSize - 1) / 2)
    new_dataset=extend_dataset(dataset,windowSize)
    patches_data = np.zeros((dataset.shape[0] * dataset.shape[1], windowSize, windowSize, dataset.shape[2]))  
    patchIndex = 0
    for row in range(offset, new_dataset.shape[0] - offset):
        for col in range(offset, new_dataset.shape[1] - offset):
            patch = new_dataset[row - offset:row + offset + 1, col- offset:col + offset + 1]   
            patches_data[patchIndex, :, :, :] = patch
            patchIndex = patchIndex + 1
    return patches_data   #patches_data为四维数组

# data process
def data_process(dataset,datalabel,feature_rate=0.5,windowSize=5,test_rate=0.4,method=1):
    num_class=get_class_num(datalabel)[1]
    if method==0:    #去掉所有0后降到二维再划分训练集和测试集
        new_data_set,new_label_set=get_nozero_data(dataset,datalabel) 
        #pca
        pca=PCA(n_components=int(new_data_set.shape[1]*0.5))  #保留一半的特征
        pca_data=pca.fit_transform(new_data_set)
        #标准化
        scaled_data=preprocessing.scale(pca_data)
        #划分
        training_data,test_data,training_label,test_label=train_test_split(scaled_data, new_label_set, test_size=test_rate, random_state=42)
    elif method==1:   #降到二维后直接划分训练集和测试集
        new_data_set=np.reshape(dataset,(-1,dataset.shape[2]))
        new_label_set=np.reshape(datalabel,(-1,1))
        #pca
        #pca=PCA(n_components='mle')
        pca=PCA(n_components=int(new_data_set.shape[1]*feature_rate))
        pca_data=pca.fit_transform(new_data_set)
        #标准化
        scaled_data=preprocessing.scale(pca_data)
        #划分
        training_data,test_data,training_label,test_label=train_test_split(scaled_data, new_label_set, test_size=test_rate, random_state=42)
    elif method==2:   #划分数据为patch，输入CNN网络
        new_data_set=np.reshape(dataset,(-1,dataset.shape[2]))
        datalabel=np.reshape(datalabel,(-1,1))
        #pca
        feature_num=int(new_data_set.shape[1]*feature_rate)
        pca=PCA(n_components=feature_num)
        pca_data=pca.fit_transform(new_data_set)
        #标准化
        scaled_data=preprocessing.scale(pca_data)
        new_data_set=np.reshape(scaled_data, (dataset.shape[0],dataset.shape[1],feature_num))
        #创建patch
        patches_data=create_patche(new_data_set, windowSize=5)
        new_label_set=np.reshape(datalabel,(-1,1))
        #划分数据
        training_data,test_data,training_label,test_label=train_test_split(patches_data, new_label_set, test_size=test_rate, random_state=345)
    else:
        print("请选择正确的数据预处理方法，程序即将退出.....")
        return 

    print( new_data_set.shape,
           new_label_set.shape,
           training_data.shape,
           test_data.shape,
           training_label.shape,
           test_label.shape,
           num_class
        )
    return  scaled_data,new_data_set,new_label_set,training_data,test_data,training_label,test_label,num_class

num_class=get_class_num(Indian_pines_gt)
print(num_class[0])
print(num_class[1])