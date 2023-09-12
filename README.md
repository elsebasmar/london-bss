# Project description

This project built a prediction model of bike flow in a bike-sharing system to estimate the number of bicycles and docks required in each station at a given point.

## Under the hood: all things that happen behind the screen

### Step 1: Collecting data
Inputting:
* All Santander trips between each station
* All major events, bank holidays, and disruptions
* Historical weather data

### Step 2: Model
Using a SARIMAX that models:
* seasonality
* exogenous variables to take into account additional factors


### Step 3: Prediction
Returning a prediction of:
* available bikes at the origin station
* available docks at the destination station
* recommendation of alternative route if necessary

## Website(built using streamlit)
### On our website:

#### Enter input information, including an origin, a destination stations and a departure time.

<img width="1511" alt="Screenshot 2023-09-12 at 12 13 53" src="https://github.com/elsebasmar/london-bss/assets/86128324/75b3cf3c-9f9f-45eb-9eef-d5fcee4db36d">

<br />

#### Getting the prediction, including the number of bikes and docks available at the stations(origin and destiantion), trip duration, weather and closest bike stations.

<img width="1135" alt="Screenshot 2023-09-12 at 12 25 04" src="https://github.com/elsebasmar/london-bss/assets/86128324/0533b47e-d05f-4f4d-afa6-29abc8457323">
<img width="1161" alt="Screenshot 2023-09-12 at 12 30 05" src="https://github.com/elsebasmar/london-bss/assets/86128324/8266b67f-9c48-4c34-ae87-5b225fc055a7">


















