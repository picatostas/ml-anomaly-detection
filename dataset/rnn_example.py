# %%
from keras.models import Sequential
from keras.layers import Dense, Dropout, CuDNNLSTM
from keras.datasets import mnist
from keras.optimizer_v2.adam import Adam
# %%
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# data is 8 bits
x_train = x_train/255
x_test = x_test/255
print(x_train.shape)
model = Sequential()
model.add(CuDNNLSTM(128, input_shape=(x_train.shape[1:]), return_sequences=True))
model.add(Dropout(0.2))
model.add(CuDNNLSTM(128))
model.add(Dropout(0.2))
model.add(Dense(32))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))
model.summary()
# %%
opt = Adam(learning_rate=1e-3, decay=1e-5)
model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test))
# %%
