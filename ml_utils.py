import numpy as np
import glob
import random
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import (
    Dense,
    Dropout,
    CuDNNLSTM,
    BatchNormalization,
    Input,
    RepeatVector,
    TimeDistributed,
    Conv1D,
    GaussianNoise,
    Flatten,
    MaxPooling1D
)


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


def load_data(split=0.35, logs_path=None, stride=50, bin_size=100, transpose=True, simplify_classes=True):

    files = glob.glob(logs_path + '*.csv')
    train_x, train_y, test_x, test_y = [], [], [], []
    for file in files:
        # container for stride bins
        data_bin = []
        df = pd.read_csv(file, delimiter=',')
        # the only input thats not binary is the main_level level sensor, so its the only
        # one that needs normalizing. Based on datasheet, it goes from 0-10000
        df['main_level'] = normalize(
            df['main_level'].values, data_max=10000, data_min=0)
        # simplify examples
        class_name = df['class'][0]
        class_to_use = class_name

        if simplify_classes:
            if class_name != 'normal':
                df = df.replace(class_name, 'error')
                class_to_use = 'error'

        data_len = len(df)

        data_idx = 0

        while ((data_idx + bin_size) <= data_len):
            data_start = data_idx
            data_stop = data_idx + bin_size
            data_bin.append(df[data_start:data_stop].values)
            data_idx += stride
        data_bin = np.array(data_bin)

        # If the data_len < log_len, skip
        if len(data_bin) < 1:
            continue

        random.shuffle(data_bin)
        # only inputs
        x_bin = data_bin[:, :, :-1]
        x_bin_t = np.zeros(
            shape=(x_bin.shape[0], x_bin.shape[2], x_bin.shape[1]))
        for ix, data in enumerate(x_bin):
            x_bin_t[ix] = data.T if transpose else data
        # To ensure that there is a proportional distribution of examples
        # in both train and test, do the split as per log file
        for idx, _ in enumerate(data_bin):
            if idx < int(len(data_bin)*split):
                test_x.append(x_bin_t[idx])
                test_y.append(class_to_use)
            else:
                train_x.append(x_bin_t[idx])
                train_y.append(class_to_use)

    train_x = np.array(train_x, dtype=float)
    test_x = np.array(test_x, dtype=float)

    data_labels, label_num, labels_dict = label_conversion(train_y)
    train_y = to_categorical(data_labels, label_num)
    test_labels = label_assignment(test_y, labels_dict)
    test_y = to_categorical(test_labels, label_num)

    return (train_x, train_y), (test_x, test_y)


def plot_results(history, title="traning results"):

    fig, axs = plt.subplots(1, 2, figsize=(10, 6))
    fig.suptitle(title)
    axs[0].set_xlabel("epochs")
    axs[1].set_xlabel("epochs")
    axs[1].set_ylim(bottom=0, top=1)
    axs[0].plot(history.history['loss'], label='Training loss')
    axs[0].plot(history.history['val_loss'], label='Validation loss')
    axs[1].plot(history.history['accuracy'], label='Training accuracy')
    axs[1].plot(history.history['val_accuracy'], label='Validation accuracy')
    axs[0].legend()
    axs[1].legend()
    axs[0].grid()
    axs[1].grid()
    plt.tight_layout()
    file_name = title.replace(',', '').replace(':', '')
    plt.savefig('./multi_dim_test/' + file_name + '.jpg')
    plt.show()


model_names = ['LSTM_A', 'LSTM_B', 'CONV1D']

def load_model(name, input_shape, output_dim):

    if name not in model_names:
        print('Model doesn\'t exist, please check the models')
        return

    # Common
    model = Sequential()
    model.add(Input(shape=input_shape))

    if name == 'LSTM_A':
        model.add(CuDNNLSTM(128, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(CuDNNLSTM(128))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.2))

    if name == 'CONV1D':
        model.add(Conv1D(filters=8, kernel_size=1,
                  padding='valid', activation='relu'))
        model.add(GaussianNoise(stddev=0.1))
        model.add(BatchNormalization())
        model.add(Conv1D(filters=16, kernel_size=1,
                  padding='valid', activation='relu'))
        model.add(Dropout(rate=0.1))
        model.add(Flatten())

    if name == 'LSTM_B':
        model.add(Conv1D(filters=32, kernel_size=4, padding='valid', activation='relu'))
        model.add(CuDNNLSTM(20))
        model.add(Dropout(rate=0.3))
        model.add(Flatten())

    # All models have the same output layer
    model.add(Dense(output_dim, activation='softmax'))

    # print(model.summary())
    return model
