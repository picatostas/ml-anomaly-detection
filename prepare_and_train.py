# %%
from keras.optimizer_v2.adam import Adam
from ml_utils import load_data, plot_results, load_model

# %%
logs_path = './exported_logs/'

(train_x, train_y), (test_x, test_y) = load_data(split=0.35,
                                                 logs_path=logs_path,
                                                 stride=50,
                                                 bin_size=150,
                                                 transpose=True,
                                                 simplify_classes=True)

# %%

model = load_model('LSTM_B', input_shape=(train_x.shape[1:]), output_dim=train_y.shape[1])

# %%

opt = Adam(learning_rate=1e-3, decay=1e-5)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
history = model.fit(train_x, train_y, epochs=100, validation_data=(test_x, test_y), verbose=0)
plot_results(history)

# %%
