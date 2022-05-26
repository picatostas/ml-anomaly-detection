import os
import traceback
import numpy as np
import yaml
import sys
import h5py
import glob
import random
import pandas as pd

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
eager_execution = True
optimizer = "Adam"
artificial_data = False

def main():
    layer_list = load_from_yaml('1D_CNN.yaml')
    # layer_list = load_from_yaml('1D_CNN_simple.yaml')
    # layer_list = load_from_yaml('LSTM.yaml')

    if not artificial_data:
        train_x, train_y, test_x, test_y = load_data_split_tranposed(split=0.35)
    else:

        train_x = np.zeros((200,10,2))
        train_x[0:100,:,:] = np.random.uniform(.8,1.0,[100,10,2])
        train_x[100:,:,:] = np.random.uniform(.0,0.2,[100,10,2])
        train_y = ['high'] * 100 + ['low'] * 100
        test_x = np.zeros((200,10,2))
        test_x[0:100,:,:] = np.random.uniform(.8,1.0,[100,10,2])
        test_x[100:,:,:] = np.random.uniform(.0,0.2,[100,10,2])
        test_y = ['high'] * 100 + ['low'] * 100

    train_labels, label_num, labels_dict = label_conversion(train_y)
    train_y = to_categorical(train_labels, label_num)
    test_labels = label_assignment(test_y, labels_dict)
    test_y = to_categorical(test_labels, label_num)

    model = init_model(layer_list, train_x.shape[1:], label_num)
    model.summary()

    train_params = {
        "epochs": 100,
        "batch_size": 64,
        "eval_data": [test_x, test_y],
        "optimizer": 'adam',
        "learning_rate": 0.001,
    }
    train(model, train_x, train_y, train_params)

    if artificial_data:
        for i in range(100):
            if np.random.rand() > 0.5:
                ground_truth = 'high'
                d = np.random.uniform(.8,1.0,[10,2])
            else:
                ground_truth = 'low'
                d = np.random.uniform(.0,0.2,[10,2])

            result = predict(model, d, labels_dict)
            print("\nGround truth: {}".format(ground_truth))
            print("Prediction: {} ({}%)".format(result[0]['prediction'], result[0]['confidence']))

def print_result(results):
    for res in results:
        for key in res:
            if key == 'label_predictions':
                for l in res['label_predictions']:
                    print(l, res['label_predictions'][l])
            else:
                print("{}: {}".format(key, res[key]))


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


def load_data_split(split=0.25):
    files = glob.glob('./logs_rev/*.csv')
    # container for stride bins
    train_x, train_y, test_x, test_y = [], [], [], []
    for file in files:
        data_bin = []
        df = pd.read_csv(file, delimiter=',')

        # the only input thats not binary is the ultrasound level sensor, so its the only
        # one that needs normalizing. Based on datasheet, it goes from 0-10000
        df['ultrasound'] = normalize(
            df['ultrasound'].values, data_max=10000, data_min=0)
        # simplify examples
        class_name = df['class'][0]
        if class_name != 'normal':
            df = df.replace(class_name, 'error')
        # stride = 5
        # bin_size = 100
        stride = 100
        bin_size = 291
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

        for idx, _ in enumerate(data_bin):
            if idx < int(len(data_bin)*split):
                test_x.append(x_bin[idx])
                test_y.append(y_bin[idx])
            else:
                train_x.append(x_bin[idx])
                train_y.append(y_bin[idx])

    train_x = np.array(train_x, dtype=float)
    test_x = np.array(test_x, dtype=float)

    return train_x, train_y, test_x, test_y


def load_data_split_tranposed(split=0.25):
    files = glob.glob('../../exported_logs/*.csv')
    # container for stride bins
    train_x, train_y, test_x, test_y = [], [], [], []
    for file in files:
        data_bin = []
        df = pd.read_csv(file, delimiter=',')

        # the only input thats not binary is the ultrasound level sensor, so its the only
        # one that needs normalizing. Based on datasheet, it goes from 0-10000
        df['main_level'] = normalize(
            df['main_level'].values, data_max=10000, data_min=0)
        df['aux_level'] = normalize(
            df['aux_level'].values, data_max=0b1111, data_min=0)
        class_name = df['class'][0]
        class_to_use = class_name
        if class_name != 'normal':
            df = df.replace(class_name, 'error')
            class_to_use = 'error'
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
        # only inputs
        x_bin = data_bin[:, :, :-1]
        x_bin_t = np.zeros(shape=(x_bin.shape[0], x_bin.shape[2], x_bin.shape[1]))
        for ix, data in enumerate(x_bin):
            x_bin_t[ix] = data.T
        # we only want 1 output per bin
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


    return train_x, train_y, test_x, test_y


def load_from_yaml(filename):
    data = None
    try:
        with open(filename, "r") as f_handle:
            data = yaml.load(f_handle, Loader=yaml.FullLoader)
    except Exception as e:
        print("Failed to load data\n", e)
    return data

def init_model(layer_list, input_dim, output_dim, steps=1):
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

    opt_handle = get_optimizer(optimizer)

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
            x, x_test, y, y_test = train_test_split(x, y, test_size=eval_data)
            eval_data = (x_test, y_test)
    else:
        eval_data = None

    y_ints = [d.argmax() for d in y]
    class_weights = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_ints),
        y=y_ints
    )
    class_weights = dict(enumerate(class_weights))
    print(f"class weights: {class_weights}")

    cb = []
    plot_cb = train_params.get("plot_cb", None)
    stop_cb = train_params.get("stop_cb", None)
    save_best = train_params.get("save_best", None)
    steps = int(np.ceil(x.shape[0] / batch_size))
    func = TrainCallback(plot_cb=plot_cb,
                         steps_per_epoch=steps,
                         stop_cb=stop_cb,
                         save_best=save_best,
                         parent=None)
    cb.append(func)

    if plot_cb is not None:
        verbose = 0
    else:
        verbose = 1

    if "dropout" in train_params:
        dropout = train_params["dropout"]
        if isinstance(dropout, dict):
            if dropout["monitor"] in ["acc", "val_acc", "loss", "val_loss"]:
                cb_early_stop = EarlyStopping(monitor=dropout["monitor"],
                                              min_delta=dropout["min_delta"],
                                              patience=dropout["patience"],
                                              verbose=verbose,
                                              mode="auto"
                                              )
            cb.append(cb_early_stop)

    optimizer = None
    if train_params.get("optimizer"):
        optimizer = train_params["optimizer"]

    if "learning_rate" in train_params:
        K.set_value(model.optimizer.lr, train_params["learning_rate"])

    history = model.fit(x,
                        y,
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=cb,
                        validation_data=eval_data,
                        verbose=verbose,
                        class_weight=class_weights,
                        )

    return history


def get_optimizer(optimizer):
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

    return opt_handle


def set_learning_rate(model, rate):
    K.set_value(model.optimizer.lr, rate)


def eval(model, x, y):
    test_loss, test_acc = model.evaluate(x, y)
    print("\nTest result:")
    print("Loss: ", test_loss, "Accuracy: ", test_acc)


def predict(model, x, labels_dict):
    if len(x.shape) == len(model.input_shape) - 1:
        if x.shape[0] == model.input_shape[1]:
            x = np.expand_dims(x, 0)
        else:
            x = np.expand_dims(x, -1)
    if len(x.shape) == len(model.input_shape) - 2:
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
    for key in labels_dict:
        num2label[labels_dict[key]] = key

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


class TrainCallback(Callback):
    def __init__(self, plot_cb=None, steps_per_epoch=None, stop_cb=None, save_best=None, parent=None):
        self.parent = parent
        self.plot = plot_cb
        self.stop_cb = stop_cb
        self.steps_per_epoch = steps_per_epoch
        self.epoch = 0
        self.batch = 0
        self.save_best = save_best
        self.val_loss = np.inf

    def on_batch_end(self, batch, logs=None):
        self.batch += 1
        self.send_data(logs)

    def on_epoch_end(self, epoch, logs=None):
        self.epoch += 1
        self.send_data(logs)

        if self.save_best:
            try:
                if logs["val_loss"] < self.val_loss:
                    self.val_loss = logs["val_loss"]
                    fname = "model_epoch_{}_val_loss_{:.04f}".format(self.epoch, self.val_loss)
                    fname = os.path.join(self.save_best["folder"], fname)
                    self.parent.save_model(fname,
                                           self.save_best["feature_list"],
                                           self.save_best["sensor_config"],
                                           self.save_best["frame_settings"],
                                           )
            except Exception as e:
                print(e)

    def send_data(self, data):
        if "steps_per_epoch" not in data:
            data["steps_per_epoch"] = self.steps_per_epoch

        if self.plot is not None:
            self.plot(data)

        if self.stop_cb is not None:
            try:
                stop_training = self.stop_cb()
                if stop_training:
                    self.stopped_epoch = self.epoch
                    self.model.stop_training = True
            except Exception as e:
                print("Failed to call stop callback! ", e)
                pass


if __name__ == "__main__":
    main()
