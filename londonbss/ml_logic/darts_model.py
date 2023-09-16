import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.models import AutoARIMA

stations = ['wormwood_street__liverpool_street', 'wenlock_road___hoxton', 'finsbury_circus__liverpool_street', 'eagle_wharf_road__hoxton']

for station in stations:
    full_df = pd.read_csv(f'processed_all_{station}_2020-01-01_2023-06-19_full_data_4.csv')
    full_df['startdate'] = pd.to_datetime(full_df['startdate']).dt.tz_localize(None)
    full_df.drop(columns=['year', 'month', 'day','hour', 'weekday'], inplace=True)

    model_loaded = AutoARIMA.load(f"{station}_model_correct_data_encoded.pkl")

    series = TimeSeries.from_dataframe(full_df, time_col='startdate', value_cols=station, fill_missing_dates=True, freq='H', fillna_value=0)

    train, val = series.split_before(pd.Timestamp('20230615'))

    covariates = ['elisabeth_line', 'lockdown','strike', 'school_holidays', 'daytime', 'London_zone_Central',
       'London_zone_North', 'London_zone_West', 'London_zone_South_West',
       'London_zone_South_East', 'London_zone_East', 'Event', 'temperature',
       'rainfall', 'snowfall', 'cloudcover', 'wind_speed', 'wind_direction']

    cov_series = TimeSeries.from_dataframe(full_df, time_col='startdate', value_cols=covariates, fill_missing_dates=True, freq='H', fillna_value=0)

    predictionX = model_loaded.predict(len(val),future_covariates=cov_series)
    predictionX

    station_name = ' '.join(x.title() for x in (', '.join(x.title() for x in station.split('__')).split('_')))

    plt.figure(figsize=(30,10))
    predictionX[:24].plot(label='forecast', lw=5.0, c='r')
    train[-168:].plot(label='actual', c='b')
    val[:24].plot(label='actual', c='b')
    plt.title(f'{station_name} 24 hr Prediction')
    plt.legend()
