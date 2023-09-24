from django.shortcuts import render
from trading.models import TbsCrypto,TblHistCryptoDaily, TblHistCryptoWeekly, TblHistCryptoMonthly
from trading.extract_data import get_coin_data
# Create your views here.
from django.utils.timezone import get_current_timezone
from datetime import datetime


def add_to_tbs_crypto(names):
    for name in names:
        TbsCrypto.objects.create(crypto_name=name)


def add_daily_monthly_weekly_data(names, table):
    table_name = table._meta.model_name
    time_zone = table_name[13:len(table_name)].upper()
    for name in names:
        data_all = get_coin_data(name, time_zone)
        tz = get_current_timezone()
        for data in data_all:
            coin = TbsCrypto.objects.get(crypto_name=name)

            table.objects.create(
                fk_crypto_code=coin,
                date=tz.localize(datetime.strptime(data['date'], '%Y-%m-%d')),
                open=data['open'],
                close=data['close'],
                high=data['low'],
                volume=data['volume'],
                low=data['low']
            )

names = ['BTC','ETH',"ADA"]
add_to_tbs_crypto(names)
add_daily_monthly_weekly_data(names,TblHistCryptoDaily)
add_daily_monthly_weekly_data(names,TblHistCryptoMonthly)
add_daily_monthly_weekly_data(names,TblHistCryptoWeekly)