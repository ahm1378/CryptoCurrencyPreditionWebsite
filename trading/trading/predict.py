import datetime

import tensorflow as tf
from trading.learning import get_normalize_data
from trading.models import TblHistCryptoWeekly, TblHistCryptoMonthly, TblHistCryptoDaily
from sklearn.preprocessing import MinMaxScaler
from dateutil.relativedelta import *


def get_model(time_interval):
    model = tf.keras.models.load_model('trading/models/{}/model'.format(time_interval))
    return model


def get_predict_prices(name, table, time_interval):
    price_val, date_val = get_normalize_data(name, table)
    price = price_val.values
    date = date_val.values
    min_max_scaler = MinMaxScaler()
    norm_data = min_max_scaler.fit_transform(price)
    s_reshape = norm_data.reshape(1, 5, 1)
    model = get_model(time_interval)
    predict_price = min_max_scaler.inverse_transform(model.predict(s_reshape))
    result = {
        'date': date[0][0]+relativedelta(days=+1),
        'close': predict_price[0][0],
        'last_price': price_val['target'].values[-1]
    }
    return result


def predict_daily(name):
    table = TblHistCryptoDaily
    time_interval = "daily"
    predicted = get_predict_prices(name, table=table, time_interval=time_interval)
    predicted['date'] += relativedelta(days=+1)
    return predicted


def predict_weekly(name):
    table = TblHistCryptoWeekly
    time_interval = "weekly"
    predicted = get_predict_prices(name, table=table, time_interval=time_interval)
    predicted['date'] += relativedelta(weeks=+1)
    return predicted


def predict_monthly(name):
    table = TblHistCryptoWeekly
    time_interval = "monthly"
    predicted = get_predict_prices(name, table=table, time_interval=time_interval)
    predicted['date'] += relativedelta(months=+1)
    return predicted
