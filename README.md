## Project description

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













