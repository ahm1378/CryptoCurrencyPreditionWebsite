from sklearn.preprocessing import MinMaxScaler

from trading.models import TblHistCryptoDaily, TblHistCryptoWeekly, TblHistCryptoMonthly, TbsCrypto
from django.shortcuts import render
import plotly.offline as opy
import plotly.graph_objs as go
import pandas as pd
from trading.predict import predict_daily, predict_weekly, predict_monthly, get_model
import requests
from django.core.cache import cache

from trading.translate import COINTRANSLATION


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

    for key, value in data['prices'].items():
        result.append(
            {'date': key,
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
        eth_data = cache.get_or_set('five_min_data',get_interday_data('ETH','5min')[0:5], 8*60)
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
        eth_data = cache.get_or_set('one_hour_data', get_interday_data('ETH','60min')[0:5], 1*60*60)
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
        eth_data = cache.get_or_set('thirty_minute_data', get_interday_data('ETH','30min')[0:5], 1*60*60)
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
    history = cache.get_or_set('history', get_interday_data(name,'5min')[0:10], 5*60)

    context['name'] = name
    context['name_persion'] = COINTRANSLATION[name]
    context['plot'] = div
    context['history'] = history
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






