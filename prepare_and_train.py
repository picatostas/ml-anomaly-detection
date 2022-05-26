# %%
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, CuDNNLSTM,BatchNormalization, Input, RepeatVector, TimeDistributed, Conv1D, GaussianNoise, Flatten, MaxPooling1D
from keras.optimizer_v2.adam import Adam
from ml_utils import load_data, plot_results
# %%
logs_path = './exported_logs/'

(train_x, train_y), (test_x, test_y) = load_data(split=0.35,
                                                 logs_path=logs_path,
                                                 stride=50,
                                                 bin_size=100,
                                                 transpose=True,
                                                 simplify_classes=True)

# %% MODEL 1

model = Sequential()
model.add(CuDNNLSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(Dropout(0.2))
model.add(CuDNNLSTM(128))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(train_y.shape[1], activation='softmax'))
model.summary()

# %%

opt = Adam(learning_rate=1e-3, decay=1e-5)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
history = model.fit(train_x, train_y, epochs=100, validation_data=(test_x, test_y))
# %%
# plot the results
plot_results(history)

# %%
