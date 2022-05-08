# %%
import glob
import pandas as pd
import random
import numpy as np
# %%
from keras.models import Sequential
from keras.layers import Dense, Dropout, CuDNNLSTM, RepeatVector, TimeDistributed
from keras.optimizer_v2.adam import Adam
from tensorflow.keras.utils import to_categorical

# %%
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

def label_conversion(labels):
    label_num = 0
    converted_labels = np.zeros(len(labels))
    labels_dict = {}
    for i, label in enumerate(labels):
        try:
            labels_dict[label]
        except Exception:
            labels_dict[label] = label_num
            label_num += 1
        converted_labels[i] = labels_dict[label]
    return converted_labels, label_num, labels_dict

def label_assignment(labels, labels_dict):
    assigned_labels = np.zeros(len(labels))
    for i, label in enumerate(labels):
        assigned_labels[i] = labels_dict[label]
    return assigned_labels


# %%
files = glob.glob('./logs_rev/*.csv')
train_x, train_y, test_x, test_y = [], [], [], []
for file in files:
    # container for stride bins
    data_bin = []
    df = pd.read_csv(file, delimiter=',')
    split = 0.35
    # the only input thats not binary is the ultrasound level sensor, so its the only
    # one that needs normalizing. Based on datasheet, it goes from 0-10000
    df['ultrasound'] = normalize(
        df['ultrasound'].values, data_max=10000, data_min=0)
    # simplify examples
    class_name = df['class'][0]
    if class_name != 'normal':
        df = df.replace(class_name, 'error')

    stride = 30
    bin_size = 100
    data_len = len(df)

    data_idx = 0
    while ((data_idx + bin_size) <= data_len):
        data_start = data_idx
        data_stop = data_idx + bin_size
        data_bin.append(df[data_start:data_stop].values)
        data_idx += stride

    data_bin = np.array(data_bin)
    random.shuffle(data_bin)

    # only inputs
    x_bin = data_bin[:, :, :-1]
    # we only want 1 output per bin
    y_bin = data_bin[:, 0, -1]

    # To ensure that there is a proportional distribution of examples
    # in both train and test, do the split as per log file
    for idx, _ in enumerate(data_bin):
        if idx < int(len(data_bin)*split):
            test_x.append(x_bin[idx])
            test_y.append(y_bin[idx])
        else:
            train_x.append(x_bin[idx])
            train_y.append(y_bin[idx])


train_x = np.array(train_x, dtype=float)
test_x = np.array(test_x, dtype=float)

# %%
data_labels, label_num, labels_dict = label_conversion(train_y)
train_y = to_categorical(data_labels, label_num)
test_labels = label_assignment(test_y, labels_dict)
test_y = to_categorical(test_labels, label_num)

print(labels_dict.keys())

# %% MODEL 1
model = Sequential()
model.add(CuDNNLSTM(64, input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(Dropout(0.2))
model.add(CuDNNLSTM(64))
model.add(Dropout(0.2))
model.add(Dense(label_num, activation='softmax'))
model.summary()
# %%

opt = Adam(learning_rate=1e-3, decay=1e-5)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.fit(train_x, train_y, epochs=100, validation_data=(test_x, test_y))

# %%
