import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

from matplotlib.dates import DateFormatter

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()  

######################################
#### Utility Functions ###############
######################################

def str_to_datetime(s):
    split = s.split('-')
    year, month, day = int(split[0]), int(split[1]), int(split[2])
    return datetime.datetime(year=year, month=month, day=day)

#############################################
### Program Starts Here #####################
#############################################

df_orig = pd.read_csv('data/BTC-USD.csv')
df = df_orig[['Date', 'Close']]

df['Date'] = df['Date'].apply(str_to_datetime)

df = df[(df['Date'] > '2022-04-26') & (df['Date'] < '2022-05-02')]

print(df.head())

myFmt = DateFormatter("%d-%m-%y")

fig = plt.figure(figsize=(12, 10))
plt.subplot(211)
plt.plot(df['Date'], df['Close'])
plt.legend(['BTCUSD Price'])
plt.ylim(ymin = 25000)
plt.gca().xaxis.set_major_formatter(myFmt) 

plt.show()

