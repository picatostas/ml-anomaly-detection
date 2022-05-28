# %% Mnist example
from keras.models import Sequential
from keras.layers import Dense, Dropout, CuDNNLSTM
from keras.datasets import mnist
from keras.optimizer_v2.adam import Adam
# %%
(train_x, train_y), (test_x, test_y) = mnist.load_data()
# data is 8 bits
train_x = train_x/255
test_x = test_x/255
print(train_x.shape)
model = Sequential()
model.add(CuDNNLSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
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
model.fit(train_x, train_y, epochs=3, validation_data=(test_x, test_y))
# %%
