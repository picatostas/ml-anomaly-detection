# %%
import numpy as np
import matplotlib.pyplot as plt
import h5py
# %%
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, CuDNNLSTM, BatchNormalization, Input, SimpleRNN
from keras.optimizer_v2.adam import Adam
# %%
h5 = h5py.File('inse_final.hdf5', 'r')

train_x = h5['train_x'] [:]
train_y = h5['train_y'] [:]
test_x = h5['test_x'][:]
test_y = h5['test_y'][:]

stride = h5.attrs['stride']
train_test_split = h5.attrs['train_test_split']

h5.close()
# %%
print(f"train data: {len(train_x)} validation: {len(test_x)} \n")
print(f"TRAIN, Failures: {np.count_nonzero(train_y == 0)}, Normal: {np.count_nonzero(train_y == 1)}\n")
print(f"TEST,  Failures: {np.count_nonzero(test_y == 0)}, Normal: {np.count_nonzero(test_y == 1)}\n")
print(f"INSE train_x: {train_x.shape}, train_y: {train_y.shape}, test_x: {test_x.shape}, test_y: {test_y.shape}")
# %%
model = Sequential()
model.add(CuDNNLSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(CuDNNLSTM(128, return_sequences=True))
model.add(CuDNNLSTM(128))
model.add(Dense(1))
print(model.summary())
# %%
opt = Adam(learning_rate=1e-3, decay=1e-5)
model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])
model.fit(train_x, train_y, epochs=10, batch_size=1, validation_data=(test_x, test_y))

# %%
