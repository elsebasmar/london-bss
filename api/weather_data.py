# WEATHER DATA FROM OPEN METEO API
import requests
import pandas as pd

## Getting daily sunrise and sunset times from the Open Meteo API
def get_daily_data(start_date, end_date):
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
        'https://archive-api.open-meteo.com/v1/archive',
        params=params_daily_dict).json()

    sunset_time = daily_weather_response["daily"]["sunset"]
    sunrise_time = daily_weather_response["daily"]["sunrise"]
    # print(sunset_time)
    return sunrise_time,sunset_time

# get_daily_data("2023-08-20", "2023-08-23")


## Getting hourly weather data from the Open Meteo API
#def get_hourly_data(start_date, end_date):
start_date = "2023-08-20"
end_date = "2023-08-23"

url = 'https://archive-api.open-meteo.com/v1/archive'

params_hourly_dict ={
        "latitude":"51.5085", #London latitude - should remain hardcoded
        "longitude":"-0.1780971", #London longitude -should remain hardcoded
        "start_date":start_date, #could be defined in .env and used in the other files
        "end_date":end_date, #could be defined in .env and used in the other files
        "timezone":"Europe/London", #Europe/London - specific to this api
        "hourly":"temperature_2m,precipitation,rain,snowfall,cloudcover,windspeed_10m,winddirection_10m" # specific to this api
}

hourly_weather_response = requests.get(
    'https://archive-api.open-meteo.com/v1/archive',
    params=params_hourly_dict).json()

timestamp_api = hourly_weather_response["hourly"]["time"]
temperature_api = hourly_weather_response["hourly"]["temperature_2m"]
precipitation_api = hourly_weather_response["hourly"]["precipitation"]
rain_api = hourly_weather_response["hourly"]["rain"]
snow_api = hourly_weather_response["hourly"]["snowfall"]
cloudcover_api = hourly_weather_response["hourly"]["cloudcover"]
windspeed_api = hourly_weather_response["hourly"]["windspeed_10m"]
winddirection_api = hourly_weather_response["hourly"]["winddirection_10m"]

    #print(temperature, precipitation, rain, snow, cloudcover, windspeed, winddirection)
    #return temperature_api, precipitation_api, rain_api, snow_api, cloudcover_api, windspeed_api, winddirection_api

#get_hourly_data("2023-08-20", "2023-08-21")

weather_data = pd.DataFrame()
weather_data["timestamp"] = timestamp_api
weather_data["temperature"] = temperature_api
weather_data["precipitation"] = precipitation_api
weather_data["rain"] = rain_api
weather_data["snow"] = snow_api
weather_data["cloudcover"] = cloudcover_api
weather_data["wind_speed"] = windspeed_api
weather_data["wind_direction"]= winddirection_api

#print(weather_data.head(2))
#print(weather_data.shape)
