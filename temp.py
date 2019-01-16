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