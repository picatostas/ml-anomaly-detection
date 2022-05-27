# %%
from keras.optimizer_v2.adam import Adam
from ml_utils import load_data, plot_results, load_model, model_names

# %%
logs_path = './exported_logs/'

(train_x, train_y), (test_x, test_y) = load_data(split=0.35,
                                                 logs_path=logs_path,
                                                 stride=50,
                                                 bin_size=150,
                                                 transpose=True,
                                                 simplify_classes=True)

# %%


model = load_model('LSTM_A', input_shape=(train_x.shape[1:]), output_dim=train_y.shape[1])
opt = Adam(learning_rate=1e-3, decay=1e-5)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
history = model.fit(train_x, train_y, epochs=100, validation_data=(test_x, test_y), verbose=0)
plot_results(history, title=f"model: LSTM_A epochs: 100")

# %%
for name in model_names:
    for ep in [10, 30, 100]:
        model = load_model(name, input_shape=(train_x.shape[1:]), output_dim=train_y.shape[1])
        opt = Adam(learning_rate=1e-3, decay=1e-5)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        history = model.fit(train_x, train_y, epochs=ep, validation_data=(test_x, test_y), verbose=0)
        plot_results(history, title=f"model: {name} epochs: {ep}")
# %%
