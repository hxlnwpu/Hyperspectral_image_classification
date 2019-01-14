import scipy.io as sio
from PIL import Image
import numpy as np

def performence_get(test_file_path,dataset_id = 1):
    if dataset_id == 0:
        data_gt = sio.loadmat('result/indian_modify_gt.mat')['indian_pines_gt']
        matrix = np.zeros((16, 16), dtype=np.float32)
    elif dataset_id == 1:
        data_gt = sio.loadmat('result/pavia_modify_gt.mat')['paviaU']
        matrix = np.zeros((9, 9), dtype=np.float32)
    else:
        print('dataset id error')
        exit()
    data_pre = np.array(Image.open(test_file_path))
    print('The data shape you given is: ',(data_pre.shape))
    data_gt = np.asarray(data_gt,dtype=np.int32)
    data_pre = np.asarray(data_pre,dtype=np.int32)

    for i in range(data_gt.shape[0]):
        for j in range(data_gt.shape[1]):
            if data_gt[i,j] == 0:
                continue
            matrix[data_gt[i,j]-1,data_pre[i,j]-1] += 1
    ac_list = []
    for i in range(len(matrix)):
        ac = matrix[i, i] / sum(matrix[:, i])
        ac_list.append(ac)
        print(i+1,'class:','(', matrix[i, i], '/', sum(matrix[:, i]), ')', ac)
    print('confusion matrix:')
    print(np.int_(matrix))
    print('total right num:', np.sum(np.trace(matrix)))
    accuracy = np.sum(np.trace(matrix)) / np.sum(matrix)
    print('oa:', accuracy)
    # kappa
    kk = 0
    for i in range(matrix.shape[0]):
        kk += np.sum(matrix[i]) * np.sum(matrix[:, i])
    pe = kk / (np.sum(matrix) * np.sum(matrix))
    pa = np.trace(matrix) / np.sum(matrix)
    kappa = (pa - pe) / (1 - pe)
    ac_list = np.asarray(ac_list)
    aa = np.mean(ac_list)
    print('aa:',aa)
    print('kappa:', kappa)