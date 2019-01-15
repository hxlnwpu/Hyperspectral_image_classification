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
from keras.layers import Dense, Dropout, Activation,Conv2D,MaxPooling2D,Flatten
from keras.models import Sequential
from keras.optimizers import SGD
from keras.models import load_model
from keras.utils import plot_model

#all data path
Indian_pines_dataset="Data/indianpines_ds_raw.hdr"  
Indian_pines_corrected_dataset="Data/Indian_pines_corrected"
Indian_pines_gt_dataset="Data/indianpines_ts.tif" 
Indian_pines__corrected_gt_dataset="Data/Indian_pines_gt"
Pavia_dataset="Data/pavia_ds.hdr"   
Pavia_gt_dataset="Data/pavia_ts.tif"  

#read all data to ndarry
Indian_pines_gt=misc.imread(Indian_pines_gt_dataset)
Indian_pines=sp.open_image(Indian_pines_dataset).load()
Indian_pines_corrected=sio.loadmat(Indian_pines_corrected_dataset)['indian_pines_corrected']
Indian_pines__corrected_gt=sio.loadmat(Indian_pines__corrected_gt_dataset)['indian_pines_gt']
PaviaU_gt=misc.imread(Pavia_gt_dataset)
PaviaU=sp.open_image(Pavia_dataset).load()
Indian_modify_gt=sio.loadmat('indian_modify_gt.mat')['indian_pines_gt']
Pavia_modify_gt=sio.loadmat('pavia_modify_gt.mat')['paviaU']

# get classes num of label
def get_class_num(dataset):
    dict_k = {}
    for i in range(dataset.shape[0]):
        for j in range(dataset.shape[1]):
            # if output_image[i][j] in [m for m in range(1,17)]:
            if dataset[i][j] in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]:
                if dataset[i][j] not in dict_k:
                    dict_k[dataset[i][j]] = 0
                dict_k[dataset[i][j]] += 1
    return dict_k, len(dict_k)

# get all nonzero sample and label


def get_nozero_data(dataset, labelset):
    nozero_num = len(labelset.nonzero()[0])
    k = 0
    new_data_set = np.zeros((nozero_num, dataset.shape[2]))
    new_label_set = np.zeros((nozero_num, 1))
    for i in range(labelset.shape[0]):
        for j in range(labelset.shape[1]):
            if(labelset[i][j] != 0):
                new_data_set[k, :] = dataset[i][j]
                new_label_set[k, -1] = labelset[i][j]
                k = k+1
    return new_data_set, new_label_set

# 扩展数据，便于划分patch


def extend_dataset(dataset, windowSize=5):
    new_dataset = np.zeros(
        (dataset.shape[0] + windowSize-1, dataset.shape[1] + windowSize-1, dataset.shape[2]))
    offset = int((windowSize - 1) / 2)
    new_dataset[offset:dataset.shape[0] + offset,
                offset:dataset.shape[1] + offset, :] = dataset
    return new_dataset

# 创建patch：每一个patch都是一个windowSize*windowSize*特征向量长度的数组。原数据每一个元素分别成为patch中心,


def create_patche(dataset, labelset, windowSize=5, removeZero=True):
    offset = int((windowSize - 1) / 2)
    new_dataset = extend_dataset(dataset, windowSize)
    patches_data = np.zeros(
        (dataset.shape[0] * dataset.shape[1], windowSize, windowSize, dataset.shape[2]))
    patches_label = np.zeros((dataset.shape[0] * dataset.shape[1]))
    patchIndex = 0
    for row in range(offset, new_dataset.shape[0] - offset):
        for col in range(offset, new_dataset.shape[1] - offset):
            patch = new_dataset[row - offset:row +
                                offset + 1, col - offset:col + offset + 1]
            patches_data[patchIndex, :, :, :] = patch
            patches_label[patchIndex] = labelset[row-offset, col-offset]
            patchIndex = patchIndex + 1
    if removeZero:
        patches_data = patches_data[patches_label > 0, :, :, :]
        patches_label = patches_label[patches_label > 0]
        patches_label -= 1

    return patches_data, patches_label  # patches_data为四维数组，patches_label为列向量

# data process
def data_process(dataset, labelset, feature_rate=0.8, windowSize=5, test_rate=0.2, method=1):
    num_class = get_class_num(labelset)[1]
    if method == 0:  # 拍平->PCA->normalized->split
        new_data_set = np.reshape(dataset, (-1, dataset.shape[2]))
        new_label_set = np.reshape(labelset, (-1, 1))
        #pca
        feature_num = int(new_data_set.shape[1]*feature_rate)
        pca = PCA(n_components=feature_num)
        pca_data = pca.fit_transform(new_data_set)
        # 标准化
        scaled_data = preprocessing.scale(pca_data)
        new_data_set= scaled_data
        # 划分
        training_data, test_data, training_label, test_label = train_test_split(new_data_set, new_label_set, test_size=test_rate, random_state=42)
    elif method == 1:  #拍平->pca->normalized->升维->创建patch(不去0)
        new_data_set = np.reshape(dataset, (-1, dataset.shape[2]))
        new_label_set=labelset
        # pca
        feature_num = int(new_data_set.shape[1]*feature_rate)
        pca = PCA(n_components=feature_num)
        pca_data = pca.fit_transform(new_data_set)
        # 标准化
        scaled_data = preprocessing.scale(pca_data)
        #升维
        new_data_set = np.reshape(scaled_data, (dataset.shape[0], dataset.shape[1], feature_num))
        # 创建patch
        new_data_set, new_label_set = create_patche(new_data_set, new_label_set, windowSize=5,removeZero=False)
        #划分
        training_data, test_data, training_label, test_label = train_test_split(new_data_set, new_label_set, test_size=test_rate, random_state=42)
    elif method == 2:  #  #拍平->pca->normalized->升维->创建patch(不去0)
        new_data_set = np.reshape(dataset, (-1, dataset.shape[2]))
        new_label_set=labelset
        # pca
        feature_num = int(new_data_set.shape[1]*feature_rate)
        pca = PCA(n_components=feature_num)
        pca_data = pca.fit_transform(new_data_set)
        # 标准化
        scaled_data = preprocessing.scale(pca_data)
        #升维
        new_data_set = np.reshape(scaled_data, (dataset.shape[0], dataset.shape[1], feature_num))
        temp_dataset=new_data_set
        temp_labelset=new_label_set
        # 创建patch
        new_data_set, new_label_set = create_patche(temp_dataset, temp_labelset, windowSize=5,removeZero=False)
        nozero_dataset, nozero_labelset = create_patche(temp_dataset, temp_labelset, windowSize=5,removeZero=True)
        #划分
        training_data, test_data, training_label, test_label = train_test_split(nozero_dataset, nozero_labelset, test_size=test_rate, random_state=42)
    else:
        print("请选择正确的数据预处理方法，程序即将退出.....")
        return

    print( new_data_set.shape,
           new_label_set.shape,
           training_data.shape,
           test_data.shape,
           training_label.shape,
           test_label.shape,
           feature_num,
           num_class
        )
    return new_data_set, training_data, test_data, training_label, test_label, feature_num,num_class


# set model
def set_model(datasetname, new_data_set,
              training_data, test_data,
              training_label, test_label,
              num_class, feature_num, method=0):
    if method == 0:   #普通神经网络
        # 设置参数
        y_train = keras.utils.to_categorical(training_label, num_classes=num_class+1)
        y_test  = keras.utils.to_categorical(test_label, num_classes=num_class+1)
        model = Sequential()
        model.add(Dense(64, activation='relu', input_dim=feature_num))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_class+1, activation='softmax'))
        #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        # 训练模型
        model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
        model.fit(training_data, y_train, epochs=5, batch_size=32)
        # 测试模型
        score = model.evaluate(test_data, y_test, batch_size=32)
        print("正确率为："+str(score[1]))
        # 保存模型
        model.save(datasetname+"_model.h5")
        model = load_model(datasetname+"_model.h5")
        # 预测所有数据
        classes = model.predict(new_data_set, batch_size=32)
        predict_label = np.zeros((classes.shape[0], 1))
        for i in range(classes.shape[0]):
            predict_label[i, 0] = classes[i, :].argmax()
    elif method == 1:  #SVM
        #设置参数
        model = svm.SVC(kernel='linear')
        #训练模型
        model.fit(training_data, training_label)
        # 保存模型
        joblib.dump(model, datasetname+"_model.m")
        # 预测所有数据
        predict_label = joblib.load(datasetname+"_model.m").predict(new_data_set)
    elif method == 2:  # CNN
        # 设置参数
        training_data = np.reshape(training_data, (training_data.shape[0], training_data.shape[3], training_data.shape[1], training_data.shape[2]))
        test_data = np.reshape(test_data, (test_data.shape[0], test_data.shape[3], test_data.shape[1], test_data.shape[2]))
        y_train = keras.utils.to_categorical(training_label, num_classes=num_class)
        y_test = keras.utils.to_categorical(test_label, num_classes=num_class)
        input_shape = training_data[0].shape
        C1 = 3*feature_num
        model = Sequential()
        model.add(Conv2D(C1, (3, 3), activation='relu', input_shape=input_shape))
        model.add(Conv2D(3*C1, (3, 3), activation='relu'))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(6*feature_num, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_class, activation='softmax'))
        #sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
        #训练模型
        model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
        model.fit(training_data, y_train, batch_size=32, epochs=2)
        #保存模型
        #plot_model(model, to_file='model.png', show_shapes=True)
        model.save(datasetname+"_model.h5")
        #测试模型
        score = model.evaluate(test_data, y_test, batch_size=32)
        print("score is :"+str(score[1]))
        #预测所有数据
        model = load_model(datasetname+"_model.h5")
        new_data_set = np.reshape(new_data_set, (new_data_set.shape[0], new_data_set.shape[3], new_data_set.shape[1], new_data_set.shape[2]))
        classes = model.predict(new_data_set, batch_size=32)
        predict_label = np.zeros((classes.shape[0], 1))
        print(predict_label.shape)
        for i in range(classes.shape[0]):
            predict_label[i, 0] = classes[i, :].argmax()
    else:
        print("请选择正确的模型，程序即将退出.....")
        return
    return predict_label



#分类
def classify(datasetname, data_process_method=1, model_method=1):
    if datasetname == "Indian_pines":
        dataset = Indian_pines
        labelset = Indian_pines_gt
        score_label = np.reshape(Indian_modify_gt, (-1, 1))
    elif datasetname == "PaviaU":
        dataset = PaviaU
        labelset = PaviaU_gt
        score_label = np.reshape(Pavia_modify_gt, (-1, 1))
    else:
        print("输入参数错误，程序即将退出")
        return
    row = labelset.shape[0]
    col = labelset.shape[1]
    # 数据预处理
    new_data_set,training_data, test_data, training_label, test_label,feature_num,num_class=data_process(dataset, labelset,method=data_process_method)
    # 训练模型
    predict_label=set_model(datasetname, new_data_set,
                            training_data, test_data,
                            training_label, test_label,
                            num_class, feature_num, method=model_method)
    #模型评价
    ##混淆矩阵
    cm = confusion_matrix(score_label, predict_label)
    print("混淆矩阵如下：")
    print(cm)
    #cm = cm.astype(int)
    np.savetxt(datasetname+"_result.txt", cm)
    with open(datasetname+"_result.txt", 'a', encoding='utf-8') as f:
        f.write("********以上为混淆矩阵********"+"\n")
        target_names = ['class 0', 'class 1', 'class 2', 'class 3', 'class 4', 'class 5',
                        'class 6', 'class 7', 'class 8', 'class 9', 'class 10', 'class 11',
                        'class 12', 'class 13', 'class 14', 'class 15', 'class 16'
                        ]
        classes_score = classification_report(
            score_label, predict_label, target_names=target_names)
        f.write(classes_score)
        ##kappa
        kappa = cohen_kappa_score(score_label, predict_label)
        print("kappa  is："+str(kappa))
        f.write("kappa is："+str(kappa)+'\n')
        ##总体分类精度
        OA = accuracy_score(score_label, predict_label)
        print("OA is: "+str(OA))
        f.write("OA is: "+str(OA)+'\n')
        # ##平均分类精度
        # AA=average_precision_score(score_label,predict_label)
        # print("AA is:"+str(AA))
        # f.write("AA is:"+str(AA)+'\n')

    # 绘图
    result = np.reshape(predict_label, (row, col))
    #result = result.astype(int)
    image = Image.fromarray(result)
    image.save(datasetname+"_predict.tif")
    print("预测结果已保存为："+datasetname+"_predict.tif")
    sp.save_rgb(datasetname+"_predict.jpg", result, colors=sp.spy_colors)
    print("预测效果可查看图片："+datasetname+"_predict.jpg")
    
    #print(result)


if __name__ == '__main__':
    classify("Indian_pines", data_process_method=0, model_method=1)
    #classify("PaviaU",data_process_method=2,model_method=2)

