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
import keras
from keras.layers import Dense, Dropout, Activation,Conv2D,MaxPooling2D,Flatten
from keras.models import Sequential
from keras.optimizers import SGD
from keras.models import load_model



#all data path
Indian_pines_dataset=r".\Data\Indian_pines"   #145*145*220
Indian_pines_corrected_dataset=r".\Data\Indian_pines_corrected"  #145*145*200  
Indian_pines_gt_dataset=r".\Data\Indian_pines_gt" #145*145
Pavia_dataset=r".\Data\Pavia"   #1096*715*102
Pavia_gt_dataset=r".\Data\Pavia_gt"   #1096*715
PaviaU_dataset=r".\Data\PaviaU"      #610*340*103
PaviaU_gt_dataset=r".\Data\PaviaU_gt"   #610*340

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
def data_process(dataset,datalabel,feature_rate=0.15,windowSize=5,test_rate=0.3,method=1):
    num_class=get_class_num(datalabel)[1]
    if method==0:    #去掉所有0后降到二维再划分训练集和测试集
        new_data_set,new_label_set=get_nozero_data(dataset,datalabel) 
        #pca
        feature_num=int(new_data_set.shape[1]*feature_rate)
        pca=PCA(n_components=feature_num)
        pca_data=pca.fit_transform(new_data_set)
        #标准化
        scaled_data=preprocessing.scale(pca_data)
        #划分
        training_data,test_data,training_label,test_label=train_test_split(scaled_data, new_label_set, test_size=test_rate, random_state=42)
    elif method==1:   #降到二维后直接划分训练集和测试集
        new_data_set=np.reshape(dataset,(-1,dataset.shape[2]))
        new_label_set=np.reshape(datalabel,(-1,1))
        #pca
        feature_num=int(new_data_set.shape[1]*feature_rate)
        pca=PCA(n_components=feature_num)
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
        scaled_data=create_patche(new_data_set, windowSize=5)
        new_label_set=np.reshape(datalabel,(-1,1))
        #划分数据
        training_data,test_data,training_label,test_label=train_test_split(scaled_data, new_label_set, test_size=test_rate, random_state=345)
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
    return  scaled_data,new_data_set,new_label_set,training_data,test_data,training_label,test_label,num_class,feature_num


    
# set model
def set_model(datasetname,scaled_data,new_data_set,
                new_label_set,training_data,test_data,
                training_label,test_label,
                num_class,feature_num,method=0):
    if method==0:    
        #设置参数
        y_train = keras.utils.to_categorical(training_label, num_classes=num_class)
        model = Sequential()
        model.add(Dense(64, activation='relu', input_dim=feature_num))
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
        classes = model.predict(scaled_data, batch_size=32)
        predict_label=np.zeros((classes.shape[0],1))
        for i in range(classes.shape[0]):
            predict_label[i,0]=classes[i,:].argmax()
    elif method==1:   #SVM
        clf = svm.SVC(kernel='linear')
        clf.fit(training_data, training_label)
        #保存模型
        joblib.dump(clf, datasetname+"_model.m")
        #预测所有数据
        predict_label=joblib.load(datasetname+"_model.m").predict(scaled_data)
    elif method==2:  #CNN
        #整理数据
        training_data = np.reshape(training_data, (training_data.shape[0],training_data.shape[3], training_data.shape[1], training_data.shape[2]))
        y_train = keras.utils.to_categorical(training_label, num_classes=num_class)
        # #设置参数
        # input_shape= training_data[0].shape
        # C1 = 3*feature_num
        # model = Sequential()
        # model.add(Conv2D(C1, (3, 3), activation='relu', input_shape=input_shape))
        # model.add(Conv2D(3*C1, (3, 3), activation='relu'))
        # model.add(Dropout(0.25))
        # model.add(Flatten())
        # model.add(Dense(6*feature_num, activation='relu'))
        # model.add(Dropout(0.5))
        # model.add(Dense(num_class, activation='softmax'))
        # sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
        # #训练模型
        # model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        # model.fit(training_data, y_train, batch_size=32, epochs=50)
        # #保存模型
        # model.save(datasetname+"_model.h5")
        #预测所有数据
        model = load_model(datasetname+"_model.h5") 
        scaled_data = np.reshape(scaled_data, (scaled_data.shape[0],scaled_data.shape[3], scaled_data.shape[1], scaled_data.shape[2]))
        classes = model.predict(scaled_data, batch_size=32)
        predict_label=np.zeros((classes.shape[0],1))
        for i in range(classes.shape[0]):
            predict_label[i,0]=classes[i,:].argmax()
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
    scaled_data,new_data_set,new_label_set,training_data,test_data,training_label,test_label,num_class,feature_num=data_process(dataset,datalabel,method=data_process_method)
    #模型预测
    predict_label=set_model(datasetname,scaled_data,new_data_set,new_label_set,training_data,test_data,training_label,test_label,num_class,feature_num,method=model_method)
    with open(datasetname+"_result.txt",'w',encoding='utf-8') as f:
        ##混淆矩阵
        cm=confusion_matrix(new_label_set, predict_label)
        print("混淆矩阵如下：")
        print(cm)
        ##kappa
        ck=cohen_kappa_score(new_label_set,predict_label)
        print("kappa为："+str(ck))
        f.write("kappa为："+str(ck)+'\n')
        ##正确率
        hit=accuracy_score(new_label_set,predict_label)
        print("正确率为："+str(hit))
        f.write("正确率为："+str(hit)+'\n')
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
    classify("Indian_pines_corrected",data_process_method=2,model_method=2)
    #classify("Pavia",data_process_method=1,model_method=0)
    #classify("PaviaU",data_process_method=2,model_method=2)








