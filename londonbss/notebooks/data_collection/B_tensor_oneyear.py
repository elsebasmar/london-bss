import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests
import io
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns


import pandas as pd
query = '''
  SELECT *
  FROM `wagon-bootcamp-396316.londonbss.data2022`
'''
df = pd.read_gbq(query, project_id="wagon-bootcamp-396316")

print('done loading data')

new_names={}
for column_name in df.columns:
    new_names[column_name]=column_name.replace('_','').lstrip().replace(' ','')
new_names
df=df.rename(columns=new_names)



df = df.reset_index(drop=True)
df = df.sort_values(by=['StartDate'])
df = df.reset_index(drop=True)
df=df[['StartDate', 'StartStationName', 'EndStationName', 'Nooftrips']]



total_stations = pd.concat([df['StartStationName'],df['EndStationName']])
total_uniq_stations = np.unique(total_stations).tolist()

total_hours = np.unique(df['StartDate']).tolist()

i = len(total_uniq_stations)
k = len(total_hours)
trips_3darray = np.zeros((i,i,k))
trips_3darray=trips_3darray.astype(np.int8)


from sklearn.preprocessing import OrdinalEncoder
df["date_code"] = 0
df["so_code"] = 0
df["sd_code"] = 0
ord_enc = OrdinalEncoder()
df["date_code"] = ord_enc.fit_transform(df[["StartDate"]])
df["so_code"] = ord_enc.fit_transform(df[["StartStationName"]])
df["sd_code"] = ord_enc.fit_transform(df[["EndStationName"]])

coords = df[['so_code','sd_code','date_code']]
print(coords.shape)

values = df['Nooftrips']
print(values.shape)



coords = np.array(coords, dtype=np.int64).T
values = np.array(values, dtype=np.float64)

import sparse
tensor = sparse.COO(coords, data=values)

import tensorly as tl
from tensorly.contrib.sparse.decomposition import tucker

tucker_rank = [30, 30, 35]
core, factors = tucker(tensor, rank=tucker_rank, init='random', tol=10e-5,random_state=12345)
print('done factor')
tucker_rec = tl.tucker_to_tensor((core, factors))
