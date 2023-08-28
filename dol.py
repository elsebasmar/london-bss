import pandas as pd
data=pd.read_csv('tripcsvbame.csv')

# note that we just only do first 2 dataset since we still do not know how to web scrap
dat=data['usage-stats/'][:2]
frames=[]
for x in dat:
    frames.append(pd.read_csv(x[1:]))

result = pd.concat(frames)

# convert datetime

date_format = '%d/%m/%Y %H:%M'
result['End Date']=pd.to_datetime(result['End Date'],format=date_format)
result['Start Date']=pd.to_datetime(result['Start Date'],format=date_format)

# dropna
result=result.dropna()

#desire 3d data
new_result=result.set_index('Start Date').groupby([pd.Grouper(freq='h'), 'StartStation Name', 'EndStation Name']).count()
