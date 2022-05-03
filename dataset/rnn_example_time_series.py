# %%
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import CuDNNLSTM, LSTM, Input, Dropout, Dense, RepeatVector, TimeDistributed
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
import seaborn as sns


dataframe = pd.read_csv('./rnn_tests/GE.csv')
df = dataframe[['Date', 'Close']]
df['Date']
df['Date'] = pd.to_datetime(df['Date'])
sns.lineplot(x=df['Date'], y=df['Close'])
print("Start date is:", df['Date'].min())
print("End date is:", df['Date'].max())
train, test = df.loc[df['Date'] <=
                     '2021-10-30'], df.loc[df['Date'] > '2021-10-30']
scaler = StandardScaler()
train['Close'] = scaler.transform(train[['Close']])
test['Close'] = scaler.transform(test[['Close']])
seq_size = 30


def to_sequences(x, y, seq_size=1):
    x_values = []
    y_values = []

    for i in range(len(x) - seq_size):
        x_values.append(x.iloc[i:(i+seq_size)].values)
        y_values.append(y.iloc[i:(i+seq_size)].values)
    return np.array(x_values), np.array(y_values)

trainX, trainY = to_sequences(train[['Close']], train['Close'], seq_size)
testX, testY = to_sequences(test[['Close']], test['Close'], seq_size)

# %%
model = Sequential()

model.add(CuDNNLSTM(128, input_shape=(trainX.shape[1], trainX.shape[2])))
model.add(Dropout(rate=0.2))
model.add(RepeatVector(trainX.shape[1]))
model.add(CuDNNLSTM(128, return_sequences=True))
model.add(Dropout(rate=0.2))
model.add(TimeDistributed(Dense(trainX.shape[2])))
model.compile()
model.compile(optimizer='adam', loss='mae')
model.summary()
history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_split=0.1, verbose=1)
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()

# %%
