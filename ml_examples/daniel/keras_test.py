import os
import traceback
import numpy as np
import yaml
import sys
import h5py

import tensorflow as tf
from keras.utils.layer_utils import count_params
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers as Opt
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.layers import Input, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from keras.layers import (
    BatchNormalization,
    Conv1D,
    Conv2D,
    ConvLSTM2D,
    Dense,
    Dropout,
    Flatten,
    GaussianNoise,
    MaxPool2D,
    MaxPooling1D,
    Reshape,
    Activation,
    LSTM,
)

USE_LSTM = True

def main():
    train_x = np.zeros((200,10,10))
    train_x[0:100,:,:] = np.random.uniform(.8,1.0,[100,10,10])
    train_x[100:,:,:] = np.random.uniform(.0,0.2,[100,10,10])
    train_y_bin = ['high'] * 100 + ['low'] * 100
    test_x = np.zeros((200,10,10))
    test_x[0:100,:,:] = np.random.uniform(.8,1.0,[100,10,10])
    test_x[100:,:,:] = np.random.uniform(.0,0.2,[100,10,10])
    test_y_bin = ['high'] * 100 + ['low'] * 100

    train_labels, label_num, labels_dict, num2label = label_conversion(train_y_bin)
    train_y = to_categorical(train_labels, label_num)

    test_labels = label_assignment(test_y_bin, labels_dict)
    test_y = to_categorical(test_labels, label_num)

    input_dim = list(train_x.shape[1:])
    output_dim = 2

    if not USE_LSTM:
        inputs = Input(shape=input_dim)
        x = Conv1D(filters=8, kernel_size=1, padding='valid', activation='relu')(inputs)
        x = GaussianNoise(stddev=0.1)(x)
        x = BatchNormalization()(x)
        x = Conv1D(filters=16, kernel_size=1, padding='valid', activation='relu')(x)
        x = Dropout(rate=0.1)(x)
        x = Flatten()(x)
        predictions = Dense(output_dim, activation='softmax')(x)
    else:
        input_dim.append(1)
        inputs = Input(shape=input_dim)
        x = TimeDistributed(Conv1D(filters=32, kernel_size=4, padding='valid', activation='relu'))(inputs)
        x = TimeDistributed(Flatten())(x)
        x = LSTM(units=20, activation='relu', recurrent_activation='relu', return_sequences=True)(x)
        x = Dropout(rate=0.3)(x)
        x = Flatten()(x)
        predictions = Dense(output_dim, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=predictions)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=Opt.Adam(),
        metrics=["accuracy"],
    )

    model.call = tf.function(model.call, experimental_relax_shapes=True)

    model.summary()

    y_ints = [d.argmax() for d in train_y]
    class_weights = class_weight.compute_class_weight(
        class_weight = "balanced",
        classes = np.unique(y_ints),
        y = y_ints
    )
    class_weights = dict(enumerate(class_weights))

    K.set_value(model.optimizer.lr, 0.0001)

    history = model.fit(train_x,
                        train_y,
                        epochs=40,
                        batch_size=64,
                        callbacks=None,
                        validation_data=[test_x, test_y],
                        verbose=1,
                        class_weight=class_weights,
                        )

    for i in range(10):
        if np.random.rand() > 0.5:
            ground_truth = 'high'
            d = np.random.uniform(.8,1.0,[10,10])
        else:
            ground_truth = 'low'
            d = np.random.uniform(.0,0.2,[10,10])
        d = np.expand_dims(d, 0)
        d = np.expand_dims(d, -1)

        prediction = model(d, training=False)[0]
        print("\nGround truth: {}".format(ground_truth))
        for p in range(len(prediction)):
            print("{}: {:.2f}%".format(num2label[p], 100*prediction[p]))


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

    num2label = {}
    for key in labels_dict:
        num2label[labels_dict[key]] = key

    return converted_labels, label_num, labels_dict, num2label

def label_assignment(labels, labels_dict):
    assigned_labels = np.zeros(len(labels))
    for i, label in enumerate(labels):
        assigned_labels[i] = labels_dict[label]
    return assigned_labels

def labelnum2text(num, label_dict):
    for key in label_dict:
        if label_dict[key] == num:
            return key
    return None

if __name__ == "__main__":
    main()
