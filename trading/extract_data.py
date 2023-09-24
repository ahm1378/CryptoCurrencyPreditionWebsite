import requests


def get_coin_data(name, time_result, market='USD', api_key="44WET5M504MPIRR0", ):
    url = 'https://www.alphavantage.co/query'
    parameters = {
        'function': 'DIGITAL_CURRENCY_{}'.format(time_result),
        'symbol': name,
        'market': market,
        'apikey': api_key
    }
    re = requests.get(url, params=parameters)
    data = re.json()

    result = []
    data['prices'] = data.pop('Time Series (Digital Currency {})'.format(time_result.capitalize()))

    for key, value in data['prices'].items():
        result.append(
            {'date':key,
             "open": float(value['1a. open (USD)']),
             "high": float(value['2a. high (USD)']),
             "close": float(value['4a. close (USD)']),
             "low": float(value['3b. low (USD)']),
             "volume": float(value['5. volume']),
             }
        )
    return result