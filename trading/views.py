import json

from django.http import HttpResponse
from sklearn.preprocessing import MinMaxScaler

from trading.models import TblHistCryptoDaily, TblHistCryptoWeekly, TblHistCryptoMonthly, TbsCrypto
from django.shortcuts import render
import plotly.offline as opy
import plotly.graph_objs as go
import pandas as pd
from trading.predict import predict_daily, predict_weekly, predict_monthly, get_model,get_normalize_data
import requests
from django.core.cache import cache
from django.utils.timezone import get_current_timezone
from datetime import datetime
from trading.translate import COINTRANSLATION


def draw_history_interview(request, name):

    data_five_min = cache.get_or_set('five_min_data{}'.format(name), get_interday_data(name, '5min'), 8 * 60)
    data_one_hour = cache.get_or_set('one_hour_data{}'.format(name), get_interday_data(name, '60min'), 1 * 60 * 60)
    data_thirty_min = cache.get_or_set('thirty_minute_data{}'.format(name), get_interday_data(name, '30min'), 1 * 60 * 30)
    daily_data = cache.get_or_set('daily_data{}'.format(name), TblHistCryptoDaily.objects.filter(fk_crypto_code__crypto_name=name).order_by('date').values('date', 'close','open','high','low','volume'), 1*60*60*24)
    weekly_data = cache.get_or_set('weekly_data{}'.format(name), TblHistCryptoWeekly.objects.filter(fk_crypto_code__crypto_name=name).order_by('date').values('date', 'close','open','high','low', 'volume'), 1*60*60*24*7)
    monthly_data = cache.get_or_set('monthly_data{}'.format(name), TblHistCryptoMonthly.objects.filter(fk_crypto_code__crypto_name=name).order_by('date').values('date', 'close','high','low', 'volume'), 1*60*60*24)
    data_five_min_df = pd.DataFrame.from_records(data_five_min)
    data_thirty_min_df = pd.DataFrame.from_records(data_thirty_min)
    data_one_hour_df = pd.DataFrame.from_records(data_one_hour)
    daily_data_df = pd.DataFrame.from_records(daily_data)
    monthly_data_df = pd.DataFrame.from_records(monthly_data)
    weekly_data_df = pd.DataFrame.from_records(weekly_data)
    marker = dict(color='LightSkyBlue', size=120, line=dict(color='MediumPurple', width=12))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=monthly_data_df['date'], y=monthly_data_df['close'], marker=marker, visible=False,
                             line=dict(
                                 color='rgb(32, 178, 170)',

                             )))
    fig.add_trace(go.Scatter(x=monthly_data_df['date'], y=monthly_data_df['volume'], marker=marker, visible=False,fill='tozeroy' ,fillcolor='rgb(209,202,205)',
                             line=dict(
                                 color='rgb(32, 178, 170)'

                             )))

    fig.add_trace(go.Scatter(x=weekly_data_df['date'], y=weekly_data_df['close'], marker=marker,
                             visible=False, line=dict(
            color='rgb(32, 178, 170)',


        )))
    fig.add_trace(go.Scatter(x=daily_data_df['date'], y=daily_data_df['close'], marker=marker, line=dict(
        color='rgb(32, 178, 170)'

    )))

    fig.add_trace(go.Scatter(x=data_one_hour_df['date'], y=data_one_hour_df['close'], marker=marker, visible=False, line=dict(
        color='rgb(32, 178, 170)'

    )))
    fig.add_trace(go.Scatter(x=data_one_hour_df['date'], y=data_thirty_min_df['close'], marker=marker, visible=False, line=dict(
        color='rgb(32, 178, 170)'

    )))
    fig.add_trace(go.Scatter(x=data_five_min_df['date'], y=data_five_min_df['close'], marker=marker, visible=False, line=dict(
        color='rgb(32, 178, 170)'

    )))

    fig.update_layout(
        autosize=False,
        width=1200,
        height=500,

        )


    # figure = go.Figure(data=data)
    fig.update_layout(
        xaxis_title="Date",


        font=dict(
            family="Times New Roman",
            size=18,
            color="LightSeaGreen"
        ))

    fig.update_layout(
        paper_bgcolor='rgb(23,22,27)',

        )

    fig.update_layout(plot_bgcolor="rgb(23,22,27)")
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                active=0,
                x=0.57,
                y=1.2,
                buttons=list([
                    dict(label="ماهانه",
                         method="update",
                         args=[{"visible": [True,True, False, False, False, False, False]},
                               {"title": "Price",
                                }]),
                    dict(label="هفتگی",
                         method="update",
                         args=[{"visible": [False, False, True, False, False, False, False]},
                               {"title": "Price"
                               }]),
                    dict(label="روزانه",
                         method="update",
                         args=[{"visible": [False,False, False, True, False, False, False]},
                               {"title": "Price"
                                }]),
                    dict(label="یک ساعت",
                         method="update",
                         args=[{"visible": [False,False, False, False, True, False, False]},
                               {"title": "Price"
                                }]),
                    dict(label="30 دقیقه",
                         method="update",
                         args=[{"visible": [False,False, False, False, False, True, False]},
                               {"title": "Price"
                                }]),
                    dict(label="5 دقیقه",
                         method="update",
                         args=[{"visible": [False,False, False, False, False, False, True]},
                               {"title": "Price"
                                }]),


                ]),
            )
        ])


    fig.update_xaxes(showgrid=False,
ticks='outside',
showline=True,)
    fig.update_yaxes(showgrid=False,
ticks='outside',
showline=True,)
    div = opy.plot(fig, auto_open=True, output_type='div', config={"displayModeBar": False})
    context = dict()
    context['plot_history'] = div
    history = cache.get_or_set('five_min_data{}'.format(name), get_interday_data(name, '5min'), 8 * 60)
    context['name'] = name
    context['name_persion'] = COINTRANSLATION[name]
    context['history'] = history[0:10]
    context['last_price'] = daily_data_df[['close']].values[-1][0]
    return render(request=request, template_name='trading_coin/tradingview.html', context=context)


def draw_andicators(refrence, value):
    layout = go.Layout(paper_bgcolor='rgb(23,22,27)')
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "پیش بینی قیمت", 'font': {'size': 24}},
        delta={'reference': refrence, 'increasing': {'color': "RebeccaPurple"}},
        gauge={
            'axis': {'range': [None, max(value,refrence)], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, min(value, refrence)], 'color': 'cyan'},
                {'range': [min(value, refrence), max(value, refrence)], 'color': 'red'}],
            'threshold': {
                'line': {'color': "green", 'width': 4},
                'thickness': 0.75,
                'value': value}}),layout=layout)
    fig.update_layout(font={'color': "white", 'family': "Times New Roman"})
    div = opy.plot(fig, auto_open=True, output_type='div', config={"displayModeBar": False})
    return div


def get_interday_data(name,interval, market='USD', api_key="44WET5M504MPIRR0"):
    url = 'https://www.alphavantage.co/query'
    parameters = {
        'function': 'CRYPTO_INTRADAY',
        'symbol': name,
        'market': market,
        'apikey': api_key,
        'interval': interval
    }
    re = requests.get(url, params=parameters)
    data = re.json()
    result = []
    data['prices'] = data.pop('Time Series Crypto ({})'.format(interval))
    tz = get_current_timezone()
    for key, value in data['prices'].items():
        #2021-11-01 20:50:00
        result.append(
            {'date': tz.localize(datetime.strptime(key,'%Y-%m-%d %H:%M:%S')),
             "open": float(value['1. open']),
             "high": float(value['2. high']),
             "close": float(value['4. close']),
             "low": float(value['3. low']),
             "volume": float(value['5. volume']),
             }
        )
    return result


def get_current_price(name):
    url = 'https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE&from_currency={}&to_currency=BUSD&apikey=44WET5M504MPIRR0'.format(name)
    response = requests.get(url)
    data = response.json()
    rate=data['Realtime Currency Exchange Rate']['5. Exchange Rate']

    return rate


def get_all_price_symbol():
    coins = TbsCrypto.objects.all().values('crypto_name')
    result = dict()
    for coin in coins:
        result["{}_price".format(coin['crypto_name'])] = get_current_price(coin['crypto_name'])
    return result


def render_market(request):
    prices = cache.get_or_set('prices', get_all_price_symbol(), 120*4)
    return render(request, template_name='trading_coin/markets.html', context=prices)


def trading_view(request):
    return render(request=request, template_name='trading_coin/tradingview.html')


def draw_plot_preview(data, mode_predict=False ,value =0):
    layout = go.Layout(xaxis={'title': 'date'}, yaxis={'title': 'price'},
                       template='plotly_white')
    marker = dict(color='LightSkyBlue', size=120, line=dict(color='MediumPurple', width=12))
    figure = go.Figure([go.Scatter(x=data['date'], y=data['close'], marker=marker, fill='tozeroy' ,fillcolor='rgb(209,202,205)',line=dict(
                        color='rgb(16, 59, 188)',
                        width=2
                    ),)], layout=layout)

    figure.update_layout(
        autosize=False,
        width=1550,
        height=500,

        )


    # figure = go.Figure(data=data)
    figure.update_layout(
        title="Plot Title",
        xaxis_title="Date",
        yaxis_title="Price",
        legend_title="Legend Title",
        font=dict(
            family="Times New Roman",
            size=18,
        ))
    figure.update_layout(
        title={
            'text': "ETH Predict Price",
            'y': 0.9,
            'x': 0.5,

            'xanchor': 'center',
            'yanchor': 'top',
        })
    if mode_predict:
        figure.add_trace(
            go.Indicator(
                mode="number+delta",
                value= value,
                number={'prefix': "$"},
                delta={'position': "top", 'reference': data['close'].values[-1]},
                domain={'x': [0, 1], 'y': [0, 1]}))

    div = opy.plot(figure, auto_open=True, output_type='div', config={"displayModeBar": False})
    return div


def predict_daily_price(name):
    result = predict_daily(name)
    div = draw_andicators(result['last_price'], result['close'])
    return div


def render_prediction(request, name):
    div = predict_daily_price(name)
    context = dict()
    context['last_price'] = TblHistCryptoDaily.objects.filter(fk_crypto_code__crypto_name=name).order_by('date').reverse().values('close')[0]['close']


    if request.GET.get('predict') == 'روزانه':
        div = predict_daily_price(name)

    elif request.GET.get('predict') == 'هفتگی':
        div = predict_weekly_price(name)

    elif request.GET.get('predict') == 'ماهانه':
        div = predict_monthly_price(name)
    elif request.GET.get('predict') == 'پنج دقیقه':
        eth_data = cache.get_or_set('five_min_data',get_interday_data(name,'5min')[0:5], 8*60)
        df_eth = pd.DataFrame(eth_data)
        df_eth = df_eth.loc[::-1]
        price_dataframe = df_eth[['close']]
        price_values= price_dataframe.values
        min_max_scaler = MinMaxScaler()
        norm_data = min_max_scaler.fit_transform(price_values)
        s_reshape = norm_data.reshape(1, 5, 1)
        model = get_model('daily')
        predict_price = min_max_scaler.inverse_transform(model.predict(s_reshape))
        close_price = predict_price[0][0]
        last_price = df_eth['close'].values[-1]
        div = draw_andicators(last_price, close_price)
    elif request.GET.get('predict') == 'یک ساعت':
        eth_data = cache.get_or_set('one_hour_data', get_interday_data(name,'60min')[0:5], 1*60*60)
        df_eth = pd.DataFrame(eth_data)
        df_eth = df_eth.loc[::-1]
        price_dataframe = df_eth[['close']]
        price_values= price_dataframe.values
        min_max_scaler = MinMaxScaler()
        norm_data = min_max_scaler.fit_transform(price_values)
        s_reshape = norm_data.reshape(1, 5, 1)
        model = get_model('daily')
        predict_price = min_max_scaler.inverse_transform(model.predict(s_reshape))
        close_price = predict_price[0][0]
        last_price = df_eth['close'].values[-1]
        div = draw_andicators(last_price, close_price)

    elif request.GET.get('predict') == ' 30دقیقه ':
        eth_data = cache.get_or_set('thirty_minute_data', get_interday_data(name,'30min')[0:5], 1*60*30)
        df_eth = pd.DataFrame(eth_data)
        df_eth = df_eth.loc[::-1]
        price_dataframe = df_eth[['close']]
        price_values= price_dataframe.values
        min_max_scaler = MinMaxScaler()
        norm_data = min_max_scaler.fit_transform(price_values)
        s_reshape = norm_data.reshape(1, 5, 1)
        model = get_model('daily')
        predict_price = min_max_scaler.inverse_transform(model.predict(s_reshape))
        close_price = predict_price[0][0]
        last_price = df_eth['close'].values[-1]
        div = draw_andicators(last_price, close_price)
    history = cache.get_or_set('five_min_data', get_interday_data(name,'5min'), 5*60)

    context['name'] = name
    context['name_persion'] = COINTRANSLATION[name]
    context['plot'] = div
    context['history'] = history[0:10]
    return render(request=request, template_name='trading_coin/predict.html', context=context)


def last_20th_history_price(request, name):
    history = cache.get_or_set('history', get_interday_data(name, "5min")[0:10], 5 * 60)
    context = dict()
    context['name'] = name
    context['history'] = history
    return render(request=request, template_name='trading_coin/footer_price.html', context=context)


def predict_weekly_price(name):
    result = predict_weekly(name)
    div = draw_andicators(result['last_price'], result['close'])
    return div


def predict_monthly_price(name):
    result = predict_monthly(name)
    div = draw_andicators(result['last_price'], result['close'])
    return div



def create_comment(request):

    data = json.loads(request.body)
    user = request.user
    data = TblHistCryptoDaily.objects.filter(fk_crypto_code__crypto_name="ETH").order_by('date').values('date','open','close','low','high')
    result =[]
    for daton in data:
        result.append({
            'x': daton[0],
            'y': daton[1:]
        })

    try:
        response = {"result": result}
        return HttpResponse(json.dumps(response), status=201)
    except:
        response = {"error": 'error'}
        return HttpResponse(json.dumps(response), status=400)


