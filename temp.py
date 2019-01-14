import spectral as sp
from numpy import *
from scipy import misc
from sklearn import svm
from PIL import Image

#all data path
Indian_pines_dataset=r".\Data\indianpines_ds_raw.hdr"   
Indian_pines_gt_dataset=r".\Data\indianpines_ts.tif" 
Pavia_dataset=r".\Data\pavia_ds.hdr"   
Pavia_gt_dataset=r".\Data\pavia_ts.tif"  


#read all data to ndarry
Indian_pines_gt=misc.imread(Indian_pines_gt_dataset)
Indian_pines=sp.open_image(Indian_pines_dataset).load()
PaviaU_gt=misc.imread(Pavia_gt_dataset)
PaviaU=sp.open_image(Pavia_dataset).load()


print(Indian_pines_gt.shape,Indian_pines.shape,PaviaU_gt.shape,PaviaU.shape)


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