# %%
import glob
import os
import pandas as pd
import numpy as np
import random
import h5py
files = glob.glob('./logs_rev/*.csv')

# container for stride bins
sequence_data = []

total_bins = 0

def normalize(data, data_max=None, data_min=None):
    if type(data) != np.ndarray:
        print("Error, give me a ndarray")
        return
    _max = data_max if data_max is not None else data.max()
    _min = data_min if data_min is not None else data.min()

    pp = (_max - _min)
    if pp == 0:
        return np.zeros_like(data)
    else:
        return (data - _min)/pp

for file in files:
    df = pd.read_csv(file, delimiter=',')
    # TODO: Classify according to different failures, for the time being we will
    # only train for normal/fail
    class_name = df['class'][0]
    if class_name == 'normal':
        df = df.replace('normal', 1)
    else:
        df = df.replace(class_name, 0)

    # the only input thats not binary is the ultrasound level sensor, so its the only
    # one that needs normalizing. Based on datasheet, it goes from 0-10000
    df['ultrasound'] = normalize(
        df['ultrasound'].values, data_max=10000, data_min=0)

    stride_bins = 0
    stride = 5
    bin_size = 100
    data_len = len(df)

    data_idx = 0

    while ((data_idx + bin_size) <= data_len):
        data_idx += stride
        stride_bins += 1

    total_bins += stride_bins

    data_idx = 0
    for bin_idx in range(stride_bins):
        data_start = data_idx
        data_stop = data_idx + bin_size
        sequence_data.append(df[data_start:data_stop].values)
        data_idx += stride

    print(f"file:{os.path.basename(file)[:-4]}\tlen:{data_len}\tstride bins:{stride_bins}")

print(f"total bins: {total_bins}, sequence_len: {len(sequence_data)}")
sequence_data = np.array(sequence_data)

random.shuffle(sequence_data)
data_x = sequence_data[:,:,:-1] # we remove the results column
# since we are already setting each class as a numerical value
# we can take the mean per sequence as output result
data_y = sequence_data[:,:,-1].mean(axis=1)

# %% Export the data with no split
h5 = h5py.File('inse_final_no_split.hdf5', 'w')

h5['data_x'] = data_x
h5['data_y'] = data_y

h5.attrs['stride'] = stride

h5.close()
# %% split the data here or do it in the train script
train_test_split = 0.65
split_idx = int(len(sequence_data)*train_test_split)
train_x, train_y = data_x[0:split_idx,:], data_y[0:split_idx]
test_x,  test_y =  data_x[split_idx:, :], data_y[split_idx: ]

print(f"train data: {len(train_x)} validation: {len(test_x)} \n")
print(f"TRAIN, Failures: {np.count_nonzero(train_y == 0)}, Normal: {np.count_nonzero(train_y == 1)}\n")
print(f"TEST,  Failures: {np.count_nonzero(test_y == 0)}, Normal: {np.count_nonzero(test_y == 1)}\n")
print(f"INSE train_x: {train_x.shape}, train_y: {train_y.shape}, test_x: {test_x.shape}, test_y: {test_y.shape}")

h5 = h5py.File('inse_final_split.hdf5', 'w')

h5['train_x'] = train_x
h5['train_y'] = train_y
h5['test_x'] = test_x
h5['test_y'] = test_y

h5.attrs['stride'] = stride
h5.attrs['train_test_split'] = train_test_split

h5.close()

# %%
