# %%
from keras.optimizer_v2.adam import Adam
from ml_utils import load_data, plot_results, load_model, model_names
from sklearn.utils import class_weight
import numpy as np
import matplotlib.pyplot as plt
# %%
logs_path = './exported_logs/'
# %% Load data once
(train_x, train_y), (test_x, test_y) = load_data(split=0.35,
                                                 logs_path=logs_path,
                                                 stride=50,
                                                 bin_size=150,
                                                 transpose=True,
                                                 simplify_classes=False)

# %% Train for data static parameters

model = load_model('LSTM_A', input_shape=(train_x.shape[1:]), output_dim=train_y.shape[1])
opt = Adam(learning_rate=1e-3, decay=1e-5)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

y_ints = [d.argmax() for d in train_y]
class_weights = class_weight.compute_class_weight(
    class_weight = "balanced",
    classes = np.unique(y_ints),
    y = y_ints
)
class_weights = dict(enumerate(class_weights))

history = model.fit(train_x, train_y, epochs=100, validation_data=(test_x, test_y), verbose=0, class_weight=class_weights)
plot_results(history, title=f"model: LSTM_A epochs: 100")

# %% loop over split, bin_size and stride.
ep = 60
name = 'LSTM_A'
fig, axs = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle(f"{name} over different parameters, epochs:{ep}")
for x in axs:
    for y in x:
        y.set_xlabel("epochs")

axs[0][0].set_ylabel('training loss')
axs[0][1].set_ylabel('training accuracy')
axs[0][1].set_ylim(bottom=0, top=1)
axs[1][0].set_ylabel('val loss')
axs[1][1].set_ylabel('val accuracy')
axs[1][1].set_ylim(bottom=0, top=1)

for split in [0.25, 0.35]:
    for bin_size in [150, 200]:
        for stride in [50, 100]:
            (train_x, train_y), (test_x, test_y) = load_data(split=split,
                                                    logs_path=logs_path,
                                                    stride=stride,
                                                    bin_size=bin_size,
                                                    transpose=True,
                                                    simplify_classes=True)
            model = load_model(name, input_shape=(train_x.shape[1:]), output_dim=train_y.shape[1])
            opt = Adam(learning_rate=1e-3, decay=1e-5)
            model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

            y_ints = [d.argmax() for d in train_y]
            class_weights = class_weight.compute_class_weight(
                class_weight = "balanced",
                classes = np.unique(y_ints),
                y = y_ints
            )
            class_weights = dict(enumerate(class_weights))

            history = model.fit(train_x, train_y, epochs=ep, validation_data=(test_x, test_y), verbose=0, class_weight=class_weights)
            axs[0][0].plot(history.history['loss'], label=f"splt{split}_bin{bin_size}_stride{stride}")
            axs[0][1].plot(history.history['accuracy'], label=f"splt{split}_bin{bin_size}_stride{stride}")
            axs[1][0].plot(history.history['val_loss'], label=f"splt{split}_bin{bin_size}_stride{stride}")
            axs[1][1].plot(history.history['val_accuracy'], label=f"splt{split}_bin{bin_size}_stride{stride}")
for x in axs:
    for y in x:
        y.legend()
        y.grid()
plt.tight_layout()
plt.savefig(f'./multi_dim_test/{name}_parameter_comparsion_{ep}epochs.jpg')
plt.show()

# %% Test all models over different epochs
for ep in [10, 30, 100]:
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"Model comparison, epochs:{ep}")

    for x in axs:
        for y in x:
            y.set_xlabel("epochs")

    axs[0][1].set_ylim(bottom=0, top=1)
    axs[1][1].set_ylim(bottom=0, top=1)

    axs[0][0].set_ylabel('training loss')
    axs[0][1].set_ylabel('training accuracy')
    axs[1][0].set_ylabel('val loss')
    axs[1][1].set_ylabel('val accuracy')

    for name in model_names:
        model = load_model(name, input_shape=(train_x.shape[1:]), output_dim=train_y.shape[1])
        opt = Adam(learning_rate=1e-3, decay=1e-5)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        y_ints = [d.argmax() for d in train_y]
        class_weights = class_weight.compute_class_weight(
            class_weight = "balanced",
            classes = np.unique(y_ints),
            y = y_ints
        )
        class_weights = dict(enumerate(class_weights))
        history = model.fit(train_x, train_y, epochs=ep, validation_data=(test_x, test_y), verbose=0, class_weight=class_weights)
        axs[0][0].plot(history.history['loss'], label=name)
        axs[0][1].plot(history.history['accuracy'], label=name)
        axs[1][0].plot(history.history['val_loss'], label=name)
        axs[1][1].plot(history.history['val_accuracy'], label=name)
    for x in axs:
        for y in x:
            y.legend()
            y.grid()
    plt.tight_layout()
    plt.savefig(f'./test_results/model_comparison_{ep}epochs.jpg')
    plt.show()

# %% Test all models over different epochs using all classes

(train_x, train_y), (test_x, test_y) = load_data(split=0.35,
                                                 logs_path=logs_path,
                                                 stride=50,
                                                 bin_size=150,
                                                 transpose=True,
                                                 simplify_classes=False)

ep = 60
fig, axs = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle(f"Model comparison, epochs:{ep}")

for x in axs:
    for y in x:
        y.set_xlabel("epochs")

axs[0][1].set_ylim(bottom=0, top=1)
axs[1][1].set_ylim(bottom=0, top=1)

axs[0][0].set_ylabel('training loss')
axs[0][1].set_ylabel('training accuracy')
axs[1][0].set_ylabel('val loss')
axs[1][1].set_ylabel('val accuracy')

for name in model_names:
    model = load_model(name, input_shape=(train_x.shape[1:]), output_dim=train_y.shape[1])
    opt = Adam(learning_rate=1e-3, decay=1e-5)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    y_ints = [d.argmax() for d in train_y]
    class_weights = class_weight.compute_class_weight(
        class_weight = "balanced",
        classes = np.unique(y_ints),
        y = y_ints
    )
    class_weights = dict(enumerate(class_weights))
    history = model.fit(train_x, train_y, epochs=ep, validation_data=(test_x, test_y), verbose=0, class_weight=class_weights)
    axs[0][0].plot(history.history['loss'], label=name)
    axs[0][1].plot(history.history['accuracy'], label=name)
    axs[1][0].plot(history.history['val_loss'], label=name)
    axs[1][1].plot(history.history['val_accuracy'], label=name)
for x in axs:
    for y in x:
        y.legend()
        y.grid()
plt.tight_layout()
plt.savefig(f'./test_results/model_comparison_{ep}epochs_all_classes.jpg')
plt.show()

# %%
