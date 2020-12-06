#!/usr/bin/env python3
# coding: utf-8

import os
import io
import time
import pytz
import requests

import pandas as pd
import numpy as np
import datetime as dt

from urllib.parse import parse_qs
from bs4 import BeautifulSoup


BASE_URL = 'http://200.105.169.186'
RETRY_M = 2
SLEEP_T = 3
TIMEOUT = 20


COLUMNS = [
    'fecha',
    'estacion',
    'viento_direccion',
    'viento_velocidad',
    'temperatura',
    'temperatura_de_rocio',
    'humedad_relativa',
    'presion_atmosferica',
    'precipitacion'
]
UNITS = [
    'GMT -4',
    '',
    '',
    'Nudos',
    'Grados Centigrados',
    'Grados Centigrados',
    'Porcentaje',
    'Milibar',
    'mm/hora'
]
FAILOVER_COLUMNS = [
    'Tiempo',
    't0',
    't1',
    'Velocidad',
    'Temperatura',
    't2',
    'Humedad Relativa',
    'Presión',
    'Precipitación'
]


def fetch_failover(estacion_id, estacion_name):
    URL = BASE_URL + '/grafest/jpGraph/graficas/jsongraf.php?x={}'.format(estacion_id)

    try:
        req = requests.get(URL, timeout=TIMEOUT)
        estacion_df = pd.read_json(req.content, orient='records')

    except:
        return

    estacion_df = estacion_df.set_index('name')
    estacion_df = pd.DataFrame([*estacion_df['data']], index=estacion_df.index).T

    estacion_df[['t0', 't1', 't2']] = [estacion_name, np.nan, np.nan]
    estacion_df = estacion_df[FAILOVER_COLUMNS]
    estacion_df.columns = COLUMNS

    time_index = pd.to_datetime(estacion_df['fecha']).dt.tz_localize('UTC')
    estacion_df['fecha'] = time_index.dt.tz_convert('America/La_Paz')

    estacion_df = estacion_df.set_index(['fecha', 'estacion'])

    return estacion_df.sort_index()


def fetch_single(estacion_id, estacion_name, _try=1):
    URL = BASE_URL + '/grafest/jpGraph/graficas/datos.php?x={}'.format(estacion_id)
    print(URL)

    try:
        req = requests.get(URL, timeout=TIMEOUT)
        estacion_df = pd.read_html(req.content, decimal=',', thousands='.')

    except Exception as e: # :S
        if _try > RETRY_M:
            return fetch_failover(estacion_id, estacion_name)

        time.sleep(_try * SLEEP_T)
        return fetch_single(estacion_id, estacion_name, _try=_try + 1)

    if len(estacion_df) == 0:
        return

    estacion_df = estacion_df[0]
    estacion_df.columns = COLUMNS

    time_index = pd.to_datetime(
        estacion_df['fecha'],
        dayfirst=not estacion_df['fecha'].str.match(r'^\d{4}.*').any()
    ).dt.tz_localize('UTC')
    estacion_df['fecha'] = time_index.dt.tz_convert(pytz.FixedOffset(-240))

    estacion_df = estacion_df.set_index(['fecha', 'estacion'])

    return estacion_df


def fetch():
    URL = BASE_URL + '/grafest/jpGraph/graficas/scriptEstaciones_prueba.php'
    req = requests.get(URL, timeout=TIMEOUT * 3)
    mdf = pd.DataFrame([])

    for table_row in BeautifulSoup(req.content).find_all('tr')[1:]:
        link = table_row.find('a')

        if link is None or not link.has_attr('href'):
            continue

        link_data = parse_qs(link['href'])

        estacion_id = link_data['f'][0]
        estacion_name = table_row.find_all('td')[2].text
        estacion_df = fetch_single(estacion_id, estacion_name)

        if estacion_df is None:
            continue

        time.sleep(SLEEP_T)
        mdf = pd.concat([mdf, estacion_df])

    mdf = mdf.sort_index()
    mdf = mdf.replace('*', np.nan)

    return mdf


WIND_ROSE = [
    'E', 'ENE', 'NE', 'NNE',
    'N', 'NNW', 'NW', 'WNW',
    'W', 'WSW', 'SW', 'SSW',
    'S', 'SSE', 'SE', 'ESE'
]
WIND_ROSE = dict(zip(WIND_ROSE, np.linspace(0, 2 * np.pi, len(WIND_ROSE) + 1)))
WIND_ROSE.update({_:np.nan for _ in ['C', 'VRB', np.nan]})


def mean_weather(df):
    wdf = df[['viento_direccion', 'viento_velocidad']].dropna()

    wdf['v_x'] = (1e-3 + wdf['viento_velocidad']) * wdf['viento_direccion'].apply(np.cos)
    wdf['v_y'] = (1e-3 + wdf['viento_velocidad']) * wdf['viento_direccion'].apply(np.sin)

    wdf = wdf.mean()

    direction = np.arctan2(*zip(wdf[['v_y', 'v_x']]))
    direction = pd.Series(direction)
    direction[direction < 0] = direction[direction < 0]  + 2 * np.pi

    wdf['viento_direccion'] = [*WIND_ROSE.keys()][
        (pd.Series(WIND_ROSE.values()) - direction.to_numpy()).abs().argmin()
    ]

    df = df.mean()

    df['viento_direccion'] = wdf['viento_direccion']
    df['viento_velocidad'] = np.sqrt(wdf['v_x'] ** 2 + wdf['v_y'] ** 2)

    return df


def resample_df(df, freq):
    df['viento_direccion'] = df['viento_direccion'].map(WIND_ROSE)
    df['viento_velocidad'] = df['viento_velocidad'].astype(float)

    df = df.groupby([
        pd.Grouper(level='fecha', freq=freq),
        pd.Grouper(level='estacion'),
    ]).apply(mean_weather)

    return df


def merge_data(mdf):
    today_data = pd.read_csv('./hoy.csv', parse_dates=True, index_col=[
        'fecha', 'estacion'
    ])
    today_data = today_data.astype(
        dict(zip(today_data.columns[1:], [*[np.float64] * 6]))
    )

    today_date = today_data.index.get_level_values(0)[0]
    today_date = today_date - dt.timedelta(
        hours=today_date.hour,
        minutes=today_date.minute,
        seconds=today_date.second,
    )
    tomorrow_date = today_date + dt.timedelta(days=1)

    today_data = pd.concat([today_data, mdf])
    today_data = today_data[
        ~today_data.index.duplicated(keep='last')
    ].sort_index()
    today_data = today_data[today_date:tomorrow_date - dt.timedelta(seconds=1)]

    tomorrow_data = mdf[tomorrow_date:]

    if len(tomorrow_data):
        month_data_file = './data/{0}.{1:0=2d}.csv'.format(
            today_date.year, today_date.month
        )

        options = {}
        if os.path.exists(month_data_file):
            options['mode'] = 'a'
            options['header'] = False

        today_data = resample_df(today_data, freq='H')
        today_data = today_data.unstack(level=2)

        today_data = today_data[tomorrow_data.columns]
        today_data = today_data.astype(
            dict(zip(today_data.columns[1:], [*[np.float64] * 6]))
        )
        today_data = today_data.round(1)

        today_data.to_csv(month_data_file, **options)

        today_data = tomorrow_data
        today_date = tomorrow_date

    today_data.to_csv('./hoy.csv')

    return today_date, today_data


def update_status(data, date):
    status = pd.DataFrame([*data.index])
    status.columns = ['fecha_ultima_actualizacion', 'estacion']
    status['fecha_ultima_actualizacion'] = pd.to_datetime(
        status['fecha_ultima_actualizacion']
    )

    status = status.groupby('estacion').max()
    status = status.sort_values('fecha_ultima_actualizacion')

    stored_status = pd.read_csv('./status.csv', index_col='estacion')
    stored_status['fecha_ultima_actualizacion'] = pd.to_datetime(
        stored_status['fecha_ultima_actualizacion']
    )

    status = pd.concat([stored_status, status])
    status = status[~status.index.duplicated(keep='last')].sort_values(
        'fecha_ultima_actualizacion'
    )

    status.to_csv('./status.csv')

    with io.open('./README.md', mode='w', encoding='utf-8') as f:
        f.write(
            '### Datos Metereologicos de Bolivia   \n\n'
            '#### Estado {:10.10}:   \n\n'.format(str(date))
        )

        status['fecha_ultima_actualizacion'] = status[
            'fecha_ultima_actualizacion'
        ].astype(str)
        status.to_markdown(f)

        f.write('\n\n')

        f.write(
            '#### Meta:   \n\n'
            'Fuente: http://senamhi.gob.bo/index.php/tiempo_real   \n'
            'Unidades:   \n\n'
        )

        units = pd.DataFrame(UNITS, index=COLUMNS, columns=['unidad'])
        units.index = units.index.rename('variable')
        units.to_markdown(f)

        f.write('\n\n')


if __name__ == '__main__':
    data = fetch()
    date, _ = merge_data(data)
    update_status(data, date)
