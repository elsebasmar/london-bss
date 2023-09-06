import dateutil.parser
import requests
import numpy as np
import pandas as pd
from datetime import datetime
from londonbss.params import *


# GET LIST OF LONDON EVENTS
def get_raw_features(start_date='2022-01-01', end_date='2023-01-01'):
    ## Bank holidays
    url='https://www.gov.uk/bank-holidays.json'
    res=requests.get(url).json()

    bank_holidays_1=pd.DataFrame(res['england-and-wales']['events'])
    bank_holidays_1['date']=pd.to_datetime(bank_holidays_1['date'])
    bank_holidays_1_filtered=bank_holidays_1[bank_holidays_1['date'].dt.to_period('Y')<='2023']
    bank_holidays_1_filtered=bank_holidays_1_filtered.drop(columns=['notes','bunting'])
    bank_holidays_1_filtered.loc[42, 'date'] = "2023-01-01"
    url_2=f'{LOCAL_DATA_PATH}/UK_Bank_Holidays_2.csv'
    bank_holidays_2=pd.read_csv(url_2)
    bank_holidays_2=bank_holidays_2[bank_holidays_2['Bank Holiday']==1]
    bank_holidays_2['Date']=pd.to_datetime(bank_holidays_2['Date'])
    bank_holidays_2_filtered=bank_holidays_2[bank_holidays_2['Date'].dt.to_period('Y')>='2014']
    bank_holidays_2_filtered=bank_holidays_2_filtered[bank_holidays_2_filtered['Date'].dt.to_period('Y')<='2018']
    bank_holidays_2_filtered=bank_holidays_2_filtered.drop(columns='Bank Holiday')
    bank_holidays_2_filtered=bank_holidays_2_filtered.rename(columns={"Name": "title", "Date": "date"})
    bank_holidays_2_filtered=bank_holidays_2_filtered.replace(['May Day (Early May Bank Holiday)'], ['Early_May_bank_holiday'])
    bank_holidays_2_filtered=bank_holidays_2_filtered.replace(['Spring Bank Holiday'], ['Spring_bank_holiday'])
    bank_holidays_2_filtered=bank_holidays_2_filtered.replace(['Summer Bank Holiday'], ['Summer_bank_holiday'])
    bank_holidays_2_filtered=bank_holidays_2_filtered.replace(['Christmas'], ['Christmas_Day'])
    bank_holidays_2_filtered=bank_holidays_2_filtered.replace(["New Year's Day"], ['New_Years_Day'])

    bank_holidays_fv=pd.concat([bank_holidays_2_filtered,bank_holidays_1_filtered],axis=0)
    bank_holidays_fv=bank_holidays_fv.sort_values(by=['date'])
    bank_holidays_fv=bank_holidays_fv.rename(columns={"date": "start_date"})
    bank_holidays_fv['end_date']=bank_holidays_fv['start_date']
    bank_holidays_fv['Location']='London-wide'
    bank_holidays_fv['Latitude']='London-wide'
    bank_holidays_fv['Longitude']='London-wide'
    bank_holidays_fv=bank_holidays_fv.reset_index(drop=True)

    ## Events
    url_3=f'{LOCAL_DATA_PATH}/London_Events_v4.csv'
    london_events=pd.read_csv(url_3)
    london_events=london_events.dropna()
    london_events_filtered=london_events[london_events['start_date']!='Cancelled']
    london_events_filtered['start_date']=pd.to_datetime(london_events_filtered['start_date'])
    london_events_filtered['end_date']=pd.to_datetime(london_events_filtered['end_date'])

    ## All events concatenation
    all_events_df=pd.concat([bank_holidays_fv,london_events_filtered],axis=0)
    all_events_df["date"] = all_events_df["start_date"]
    all_events_df=all_events_df.reset_index(drop=True)
    # all_events_df.to_csv('raw_data/all_events_df.csv')

    # GET WEATHER DATA
    ## Get daily data from the API
    url = 'https://archive-api.open-meteo.com/v1/archive'

    params_daily_dict ={
            "latitude":"51.5085", #London latitude - should remain hardcoded
            "longitude":"-0.1780971", #London longitude -should remain hardcoded
            "start_date":start_date, #could be defined in .env and used in the other files
            "end_date":end_date, #could be defined in .env and used in the other files
            "timezone":"Europe/London", #Europe/London - specific to this api
            "daily":"sunrise,sunset" # specific to this api
    }

    daily_weather_response = requests.get(
            url,
            params=params_daily_dict).json()

    sun_df =pd.DataFrame()
    sun_df["sunrise"] = daily_weather_response["daily"]["sunrise"]
    sun_df["sunset"] = daily_weather_response["daily"]["sunset"]

    def date_parser(time):
        date_parser = dateutil.parser.isoparse(time)
        return date_parser

    sun_df["sunrise_datetime"] = sun_df.apply(lambda x: date_parser(x["sunrise"]), axis = 1)
    sun_df["sunset_datetime"] = sun_df.apply(lambda x: date_parser(x["sunset"]), axis = 1)
    sun_df['date'] = sun_df['sunrise_datetime'].dt.date

    ## Get hourly data from the API
    params_hourly_dict ={
            "latitude":"51.5085", #London latitude - should remain hardcoded
            "longitude":"-0.1780971", #London longitude -should remain hardcoded
            "start_date":start_date, #could be defined in .env and used in the other files
            "end_date":end_date, #could be defined in .env and used in the other files
            "timezone":"Europe/London", #Europe/London - specific to this api
            "hourly":"temperature_2m,precipitation,rain,snowfall,cloudcover,windspeed_10m,winddirection_10m" # specific to this api
    }

    hourly_weather_response = requests.get(
        url,
        params=params_hourly_dict).json()

    weather_data = pd.DataFrame()
    weather_data["timestamp"] = hourly_weather_response["hourly"]["time"]
    weather_data["temperature"] = hourly_weather_response["hourly"]["temperature_2m"]
    weather_data["rainfall"] = hourly_weather_response["hourly"]["rain"]
    weather_data["snowfall"] = hourly_weather_response["hourly"]["snowfall"]
    weather_data["cloudcover"] = hourly_weather_response["hourly"]["cloudcover"]
    weather_data["wind_speed"] = hourly_weather_response["hourly"]["windspeed_10m"]
    weather_data["wind_direction"] = hourly_weather_response["hourly"]["winddirection_10m"]

    weather_data["timestamp"] = weather_data.apply(lambda x: date_parser(x["timestamp"]), axis = 1) # recode timestamp to datetime format

    weather_data['date'] = weather_data['timestamp'].dt.date # add date column for future merge

    ## Merge daily and hourly dfs
    weather_data = weather_data.merge(sun_df)
    # weather_data.to_csv('raw_data/weather_data.csv')

    ## Add date details
    weather_data["year"] = weather_data["timestamp"].dt.year
    weather_data["month"] = weather_data["timestamp"].dt.month
    weather_data["day"] = weather_data["timestamp"].dt.day
    weather_data["hour"] = weather_data["timestamp"].dt.hour
    weather_data["weekday"] = weather_data["timestamp"].dt.weekday

    def daytime_encoding(timestamp, sunrise_datetime, sunset_datetime):
        if timestamp < sunrise_datetime:
            daytime_encoding = "nighttime"
        elif timestamp >= sunrise_datetime and timestamp < sunset_datetime:
            daytime_encoding = "daytime"
        else:
            daytime_encoding = "nighttime"
        return daytime_encoding

    weather_data["daytime"] = weather_data.apply(lambda x: daytime_encoding(x["timestamp"], x["sunrise_datetime"], x["sunset_datetime"]), axis = 1)

    # MERGE EVENTS AND WEATHER DFS
    #all_events_df.drop("Unnamed: 0",axis =1, inplace=True)
    new_column_list = ('event_title', 'event_start_date', 'event_end_date', 'event_location', 'event_latitude', 'event_longitude', 'date')
    all_events_df.columns = new_column_list
    all_events_df["event_start_date"]= pd.to_datetime(all_events_df["event_start_date"])
    all_events_df["event_start_date"]= pd.to_datetime(all_events_df["event_start_date"])
    all_events_df['date'] = all_events_df['event_start_date'].dt.date

    weather_events_data = weather_data.merge(all_events_df, on="date", how="left")
    weather_events_data["date"]= pd.to_datetime(weather_events_data["date"])

    # ADD ADDITIONAL COLUMNS
    ## Add Elisabeth line
        # Elisabeth line opening = 24/05/2022

    weather_events_data["elisabeth_line"] = True
    weather_events_data.loc[weather_events_data["date"] < "2022-05-24", "elisabeth_line"] = False

    ## Add lockdown column
        # lockdown 1 = 24/03/2020 to 28/05/2020
        # lockdown 2 = 05/11/2020 to 02/12/2020
        # lockdown 3 = 04/01/2021 to 12/04/2021

    lockdown1_start = datetime.strptime("2020-03-24", '%Y-%m-%d')
    lockdown1_end = datetime.strptime("2020-05-28", '%Y-%m-%d')
    lockdown2_start = datetime.strptime("2020-11-05", '%Y-%m-%d')
    lockdown2_end = datetime.strptime("2020-12-02", '%Y-%m-%d')
    lockdown3_start = datetime.strptime("2021-01-04", '%Y-%m-%d')
    lockdown3_end = datetime.strptime("2021-04-12", '%Y-%m-%d')

    def lockdown_date(date):
        if date > lockdown1_start and date <= lockdown1_end:
            return True
        elif date > lockdown2_start and date <= lockdown2_end:
            return True
        elif date > lockdown3_start and date <= lockdown3_end:
            return True
        else:
            return False

    weather_events_data["lockdown"] = weather_events_data["date"].apply(lockdown_date)


    ## School holidays
    school_holidays = pd.read_csv(f'{LOCAL_DATA_PATH}/school_holidays.csv')
    school_holidays["date"] = pd.to_datetime(school_holidays["date"])
    weather_events_data = weather_events_data.merge(school_holidays, on="date", how="left")


    ## Add tube and train strikes
    strikes = pd.read_csv(f'{LOCAL_DATA_PATH}/strikes.csv')
    strikes["date"] = pd.to_datetime(strikes["date"])
    weather_events_data = weather_events_data.merge(strikes, on="date", how="left")
    weather_events_data = weather_events_data.set_index("timestamp")

    ## Drop columns
    weather_events_data= weather_events_data.drop(columns=['sunrise', 'sunset', 'sunrise_datetime',
        'sunset_datetime'])

    # ## Export final df to raw_data folder
    # weather_events_data.to_csv('raw_data/final_features_df.csv')
    return weather_events_data
