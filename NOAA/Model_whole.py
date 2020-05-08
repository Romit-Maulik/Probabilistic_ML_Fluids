# Model_noaa_whole.py
# 2020/05/02 Kai Fukami (Keio University, kai.fukami@keio.jp)

## Probabilistic model for data estimation with NOAA SST data.
## Authors:
# Romit Maulik (Argonne National Lab.), Kai Fukami (Keio University), Nesar Ramachandra (Argonne National Lab.), Koji Fukagata (Keio University), Kunihiko Taira (UCLA)

## We provide no guarantees for this code.  Use as-is and for academic research use only; no commercial use allowed without permission. For citation, please use the reference below:
# Ref: R. Maulik, K. Fukami, N. Ramachandra, K. Fukagata, and K. Taira,
#     "Probabilistic neural networks for fluid flow model-order reduction and data recovery,"
#     in Review, 2020
#
# The code is written for educational clarity and not for speed.
# -- version 1: May 2, 2020


from keras.layers import Input, Add, Dense, Flatten, Reshape
from keras.models import Model
from keras import backend as K
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from scipy import signal, interpolate
import matplotlib.pyplot as plt
import pickle,sys
from tqdm import tqdm as tqdm

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import matplotlib.pylab as plt
np.random.seed(10)

import tensorflow as tf
# tf.set_random_seed(10)


import tensorflow.keras as keras
import tensorflow_probability as tfp
tfd = tfp.distributions

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from tensorflow.keras import optimizers, models, regularizers
from tensorflow.keras import backend as K

# Activate TF2 behavior:
# from tensorflow.python import tf2
# if not tf2.enabled():
#   import tensorflow.compat.v2 as tf
#   tf.enable_v2_behavior()
#   assert tf2.enabled()

import h5py
import numpy as np

f = h5py.File('./sst_weekly.mat','r') # can be downloaded from https://drive.google.com/drive/folders/1pVW4epkeHkT2WHZB7Dym5IURcfOP4cXu?usp=sharing
lat = np.array(f['lat'])
lon = np.array(f['lon'])
sst = np.array(f['sst'])
time = np.array(f['time'])


sst2 = np.zeros((len(sst[:,0]),len(lat[0,:]),len(lon[0,:])))
for i in tqdm(range(len(sst[:,0]))):
    sst2[i,:,:] = np.flipud((sst[i,:].reshape(len(lat[0,:]),len(lon[0,:]),order='F')))


sst1 = np.nan_to_num(sst)

#location can be decided with np.rand although it is omitted here

X_train = np.zeros((1040,10))
y_train = np.zeros((1040,180*360))
X_train[:,0] = sst2[:1040,120,90]
X_train[:,1] = sst2[:1040,40,170]
X_train[:,2] = sst2[:1040,60,230]
X_train[:,3] = sst2[:1040,50,305]
X_train[:,4] = sst2[:1040,55,325]
X_train[:,5] = sst2[:1040,52,345]
X_train[:,6] = sst2[:1040,100,345]
X_train[:,7] = sst2[:1040,102,350]
X_train[:,8] = sst2[:1040,110,348]
X_train[:,9] = sst2[:1040,125,335]
y_train[:,:] = sst1[:1040,:]


X_test = np.zeros((874,10))
y_test = np.zeros((874,180*360))
X_test[:,0] = sst2[1040:,120,90]
X_test[:,1] = sst2[1040:,40,170]
X_test[:,2] = sst2[1040:,60,230]
X_test[:,3] = sst2[1040:,50,305]
X_test[:,4] = sst2[1040:,55,325]
X_test[:,5] = sst2[1040:,52,345]
X_test[:,6] = sst2[1040:,100,345]
X_test[:,7] = sst2[1040:,102,350]
X_test[:,8] = sst2[1040:,110,348]
X_test[:,9] = sst2[1040:,125,335]
y_test[:,:] = sst1[1040:,:]

TrainshuffleOrder = np.arange(X_train.shape[0])
np.random.shuffle(TrainshuffleOrder)
X_train = X_train[TrainshuffleOrder]
y_train = y_train[TrainshuffleOrder]

num_components = 1
input_shape = 10
output_shape = [180*360]
act = 'relu'
model = keras.Sequential([
    keras.layers.Dense(units=64, activation=act, input_shape=(input_shape,)),
    keras.layers.Dense(units=128, activation=act),
    keras.layers.Dense(units=256, activation=act),
    keras.layers.Dense(units=512, activation=act),
    keras.layers.Dense(units=1024, activation=act),
    keras.layers.Dense(units=2048, activation=act),
    keras.layers.Dense(tfp.layers.MixtureNormal.params_size(num_components, output_shape)),
    tfp.layers.MixtureNormal(num_components, output_shape)

def negloglik(y_true, y_pred):
    return -y_pred.log_prob(y_true)

model.compile(loss=negloglik, optimizer='adam', metrics=[])

from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
model_cb=ModelCheckpoint('./mdn_noaa_model_whole.h5', monitor='val_loss',save_best_only=True,verbose=1)
early_cb=EarlyStopping(monitor='val_loss', patience=300,verbose=1)
cb = [model_cb, early_cb]
history = model.fit(X_train,y_train,nb_epoch=50000,batch_size=64,verbose=1,callbacks=cb,shuffle=True,validation_data=[X_test, y_test])
import pandas as pd
df_results = pd.DataFrame(history.history)
df_results['epoch'] = history.epoch
df_results.to_csv(path_or_buf='./mdn_noaa_model_whole.csv',index=False)
