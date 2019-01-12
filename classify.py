from PIL import Image
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
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.models import load_model


import spectral as sp

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

#get classes num of label
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

# data process
def data_process(dataset,datalabel,test_rate=0.4,method=1):
    num_class=get_class_num(datalabel)[1]
    if method==0:    #去掉所有0后降到二维再划分训练集和测试集
        new_data_set,new_label_set=get_nozero_data(dataset,datalabel) 
        #pca
        pca=PCA(n_components='mle')
        pca_data=pca.fit_transform(new_data_set)
        #标准化
        new_data_set_scaled=preprocessing.scale(pca_data)
        #划分
        training_data,test_data,training_label,test_label=train_test_split(new_data_set_scaled, new_label_set, test_size=test_rate, random_state=42)
    elif method==1:   #降到二维后直接划分训练集和测试集
        new_data_set=np.reshape(dataset,(-1,dataset.shape[2]))
        new_label_set=np.reshape(datalabel,(-1,1))
        #pca
        pca=PCA(n_components='mle')
        pca_data=pca.fit_transform(new_data_set)
        #标准化
        new_data_set_scaled=preprocessing.scale(pca_data)
        #划分
        training_data,test_data,training_label,test_label=train_test_split(new_data_set_scaled, new_label_set, test_size=test_rate, random_state=42)
    elif method==2:   #用于CNN，数据不作处理，直接三维
        new_data_set=dataset
        new_label_set=datalabel
        new_data_set_scaled=np.zeros((1,1))
        training_data=np.zeros((1,1))  
        test_data=np.zeros((1,1))  
        training_label=np.zeros((1,1))  
        test_label=np.zeros((1,1))  
    else:
        print("请选择正确的数据预处理方法，程序即将退出.....")
        return 

    print( new_data_set_scaled.shape,
           new_data_set.shape,
           new_label_set.shape,
           training_data.shape,
           test_data.shape,
           training_label.shape,
           test_label.shape,
           num_class
        )
    return  new_data_set_scaled,new_data_set,new_label_set,training_data,test_data,training_label,test_label,num_class


    
# set model
def set_model(datasetname,new_data_set_scaled,new_data_set,
                new_label_set,training_data,test_data,
                training_label,test_label,
                num_class,method=0):
    if method==0:   
        #设置参数
        y_train = keras.utils.to_categorical(training_label, num_classes=num_class)
        #y_test = keras.utils.to_categorical(test_label, num_classes=num_class)
        model = Sequential()
        model.add(Dense(64, activation='relu', input_dim=training_data.shape[1]))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_class, activation='softmax'))
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        #训练模型
        model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
        model.fit(training_data, y_train, epochs=20,batch_size=32)
        #保存模型
        model.save(datasetname+"_model.h5")
        #测试模型 
        model = load_model(datasetname+"_model.h5")          
        # score = model.evaluate(test_data, y_test, batch_size=32)
        # print("正确率为："+str(score[1]))
        #预测所有数据
        classes = model.predict(new_data_set_scaled, batch_size=32)
        #print(classes.shape)
        predict_label=np.zeros((classes.shape[0],1))
        for i in range(classes.shape[0]):
            predict_label[i,0]=classes[i,:].argmax()
        #print(predict_label.shape  
    elif method==1:
        #y_train = keras.utils.to_categorical(training_label, num_classes=num_class)
        #y_test = keras.utils.to_categorical(test_label, num_classes=num_class)
        clf = svm.SVC(kernel='linear')
        clf.fit(training_data, training_label)
        joblib.dump(clf, datasetname+"_model.m")
        predict_label=joblib.load(datasetname+"_model.m").predict(new_data_set_scaled)
    else:
        print("请选择正确的模型，程序即将退出.....")
        return
    
    return predict_label


#分类
def classify(datasetname,data_process_method=1,model_method=1):
    if datasetname=="Indian_pines_corrected":
        dataset=Indian_pines_corrected
        datalabel=Indian_pines_gt
    elif datasetname=="PaviaU":
        dataset=PaviaU
        datalabel=PaviaU_gt
    elif datasetname=="Pavia":
        dataset=Pavia
        datalabel=Pavia_gt
    else:
        print("输入参数错误，程序即将退出")
        return 
    row=datalabel.shape[0]
    col=datalabel.shape[1]
    #数据预处理
    new_data_set_scaled,new_data_set,new_label_set,training_data,test_data,training_label,test_label,num_class=data_process(dataset,datalabel,method=data_process_method)
    #模型预测
    predict_label=set_model(datasetname,new_data_set_scaled,new_data_set,new_label_set,training_data,test_data,training_label,test_label,num_class,method=model_method)
    ##混淆矩阵
    cm=confusion_matrix(new_label_set, predict_label)
    print("混淆矩阵如下：")
    print(cm)
    ##kappa
    ck=cohen_kappa_score(new_label_set,predict_label)
    print("kappa为："+str(ck))
    ##正确率
    hit=accuracy_score(new_label_set,predict_label)
    print("正确率为："+str(hit))
    #绘图
    result=np.reshape(predict_label,(row,col))
    image = Image.fromarray(result)
    image.save(datasetname+"_predict.tif")
    print("预测结果已保存为："+datasetname+"_predict.tif")
    sp.save_rgb(datasetname+"_predict.jpg",result,colors=sp.spy_colors)
    print("预测效果可查看图片："+datasetname+"_predict.jpg")
    print("预测结果展示如下：")
    print(result)
    return result

if __name__ == '__main__':
    classify("Indian_pines_corrected",data_process_method=1,model_method=0)
    #classify("Pavia",data_process_method=1,model_method=0)
    #classify("PaviaU",data_process_method=1,model_method=0)








