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

import layer_definitions

not_time_distributed = [
    "GaussianNoise",
    "Dropout",
    "BatchNormalization",
    "LSTM",
]
eager_execution = False
optimizer = "Adam"

def main():

    # layer_list = load_from_yaml('./1D_CNN.yaml')
    layer_list = load_from_yaml('./LSTM.yaml')
    h5 = h5py.File('inse_final.hdf5', 'r')

    train_x = h5['train_x'] [:]
    train_y = h5['train_y'] [:]
    test_x = h5['test_x'][:]
    test_y = h5['test_y'][:]

    h5.close()

    model = init_model(layer_list, input_dim=train_x.shape[1:], output_dim=1)
    model.summary()
    train(model, x=train_x, y=train_y, train_params={'eval_data' : (test_x, test_y),
                                                     'epochs': 10,
                                                     'batch_size': 64,
                                                     'optimizer': 'adam'})

def load_from_yaml(filename):
    data = None
    try:
        with open(filename, "r") as f_handle:
            data = yaml.load(f_handle, Loader=yaml.FullLoader)
    except Exception as e:
        print("Failed to load data\n", e)
    return data

def init_model(layer_list, input_dim=(10,5), output_dim=2, steps=5):
    if not isinstance(input_dim, list):
        input_dim = list(input_dim)

    print("\nInitiating model with {:d}x{:d} inputs"
          " and {:d} outputs".format(*input_dim, output_dim))

    if layer_list is None:
        print("No layers defined!")
        return {"loaded": False, "model_message": "No Layers defined"}

    nr_layers = len(layer_list)
    layer_callbacks = layer_definitions.get_layers()

    lstm_mode = False
    time_series = False
    print("Building model with {} layers...".format(nr_layers))
    if steps > 1:
        input_dim[-1] = input_dim[-1] - steps + 1
        input_dim = [steps] + input_dim
        lstm_mode = True
        time_series = True
        print("Building TimeDistributed model with {} steps!".format(steps))

    # Add single channel dimension
    if input_dim[-1] != 1:
        input_dim = input_dim + [1]

    inputs = Input(shape=input_dim)

    x = None
    nr_layers = len(layer_list)
    for idx, layer in enumerate(layer_list):
        if not layer.get("is_active", True):
            continue
        try:
            cb = layer_callbacks[layer["name"]]["class"]
        except KeyError:
            print("Layer {} not found in layer_definitions.py!".format(layer["name"]))

        # Wrap layers in TimeDistributed until first LSTM layer
        if layer["name"] == "lstm":
            lstm_mode = False
            time_series = False
            layer["params"].pop("steps", None)
        if lstm_mode:
            if layer["class"] not in not_time_distributed:
                time_series = True
            else:
                time_series = False

        try:
            options = {}
            if layer["params"] is not None:
                for entry in layer["params"]:
                    opt = layer["params"][entry]
                    if isinstance(opt, list):
                        options[entry] = tuple(opt)
                    else:
                        options[entry] = opt
            print("{}: Adding {} with\n{}".format(idx + 1, layer["name"], options))
            if idx == 0 and nr_layers > 1:
                if time_series:
                    x = TimeDistributed(cb(**options))(inputs)
                else:
                    x = cb(**options)(inputs)
            elif idx > 0 and idx < nr_layers - 1:
                if time_series:
                    x = TimeDistributed(cb(**options))(x)
                else:
                    x = cb(**options)(x)
            else:
                options.pop("units", None)
                predictions = cb(output_dim, **options)(x)
        except Exception as e:
            traceback.print_exc()
            error = "\nLayer nr. {} failed. Error adding {}\n{}".format(idx + 1, layer["name"], e)
            sys.exit(0)

        if layer["name"] == "lstm":
            layer["params"]["steps"] = steps

    model = Model(inputs=inputs, outputs=predictions)
    if eager_execution:
        model.call = tf.function(model.call, experimental_relax_shapes=True)

    if optimizer.lower() == "adam":
        opt_handle = Opt.Adam()
    elif optimizer.lower() == "adagrad":
        opt_handle = Opt.Adagrad()
    elif optimizer.lower() == "adadelta":
        opt_handle = Opt.Adadelta()
    elif optimizer.lower() == "rmsprop":
        opt_handle = Opt.RMSprop()
    else:
        print("Unknown optimizer {}. Using Adam!".format(optimizer))
        opt_handle = Opt.Adam()

    print("Setting model optimizer to {}".format(optimizer))
    model.compile(
        loss="categorical_crossentropy",
        optimizer=opt_handle,
        metrics=["accuracy"],
    )

    return model

def train(model, x, y, train_params):
    epochs = train_params["epochs"]
    batch_size = train_params["batch_size"]

    if "eval_data" in train_params:
        eval_data = train_params["eval_data"]
        if isinstance(eval_data, float):
            x, xTest, y, yTest = train_test_split(x, y, test_size=eval_data)
            eval_data = (xTest, yTest)
    else:
        eval_data = None

    y_ints = [d.argmax() for d in y]
    class_weights = class_weight.compute_class_weight(class_weight="balanced",
                                                      classes=np.unique(y_ints),
                                                      y=y_ints)
    class_weights = dict(enumerate(class_weights))

    optimizer = None
    if train_params.get("optimizer"):
        optimizer = train_params["optimizer"]

    if "learning_rate" in train_params:
        K.set_value(model.optimizer.lr, train_params["learning_rate"])

    history = model.fit(x,
                        y,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=eval_data,
                        class_weight=class_weights,
                        )

    return history

def set_learning_rate(model, rate):
    K.set_value(model.optimizer.lr, rate)

def eval(model, x, y):
    test_loss, test_acc = model.evaluate(x, y)
    print("\nTest result:")
    print("Loss: ", test_loss, "Accuracy: ", test_acc)

def predict(model, x):
    if len(x.shape) == len(self.model.input_shape) - 1:
        if x.shape[0] == self.model.input_shape[1]:
            x = np.expand_dims(x, 0)
        else:
            x = np.expand_dims(x, -1)
    if len(x.shape) == len(self.model.input_shape) - 2:
        x = np.expand_dims(x, 0)
        x = np.expand_dims(x, -1)

    if len(x.shape) != len(model.input_shape):
        print("Wrong data shapes:\n Model: {}\n Test: {}\n".format(model.input_shape,
                                                                    x.shape,))
        return None

    if eager_execution:
        prediction = model(x, training=False)
    else:
        prediction = model.predict(x)
    result = list()

    num2label = {}
    for key in self.labels_dict:
        num2label[self.labels_dict[key]] = key

    for pred in prediction:
        confidence = 0
        prediction_label = ""
        res = {}
        category = {}
        for p in range(len(pred)):
            label = num2label[p]
            if pred[p] > confidence:
                prediction_label = label
                confidence = pred[p]
            category[label] = [pred[p], p]
        res["label_predictions"] = category
        res["number_labels"] = len(pred)
        res["prediction"] = prediction_label
        res["confidence"] = confidence
        res["label_num"] = np.argmax(pred)
        result.append(res)
    return result

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

def labelnum2text(num, label_dict):
    for key in label_dict:
        if label_dict[key] == num:
            return key
    return None

if __name__ == "__main__":
    main()
