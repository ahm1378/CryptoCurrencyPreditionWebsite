from keras.models import Sequential
from keras.layers import Dense, LSTM, LeakyReLU, Dropout
from trading.models import TbsCrypto, TblHistCryptoDaily
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np


def get_normalize_data(name, table):
    cryptoes = table.objects.filter(fk_crypto_code__crypto_name=name).order_by('date').reverse()[0:5]
    dict_cryptoes = cryptoes.values('date', 'open', 'high', 'low', 'volume', 'close')
    data = pd.DataFrame.from_records(dict_cryptoes)
    data = data.loc[::-1]
    data['target'] = data['close']
    date = data[['date']]
    price = data[['target']]

    return price, date


def univariate_data(dataset, start_index, end_index, history_size, target_size):
    data = []
    labels = []
    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size
    for i in range(start_index, end_index):
        indices = range(i - history_size, i)
        # Reshape data from (history_size,) to (history_size, 1)
        data.append(np.reshape(dataset[indices], (history_size, 1)))
        labels.append(dataset[i + target_size])
    return np.array(data), np.array(labels)


def fit_model(data):
    min_max_scaler = MinMaxScaler()
    norm_data = min_max_scaler.fit_transform(data.values)
    num_units = 64
    activation_function = 'sigmoid'
    loss_function = 'mse'
    batch_size = 5
    model = Sequential()
    model.add(LSTM(units=num_units, activation=activation_function))
    model.add(LeakyReLU(alpha=0.5))
    model.add(Dropout(0.1))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss=loss_function)
    past_history = 5
    future_target = 0
    TRAIN_SPLIT = int(len(norm_data) * 0.8)
    x_train, y_train = univariate_data(norm_data,
                                       0,
                                       TRAIN_SPLIT,
                                       past_history,
                                       future_target)

    model.compile(loss='mse', optimizer='Adam')

    model.fit(
        x_train,
        y_train,
        validation_split=0.1,
        batch_size=batch_size,
        epochs=270,
        shuffle=False
    )
    return model