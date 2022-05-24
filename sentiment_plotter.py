import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

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

mask = (df['Date'] > '2022-04-26') & (df['Date'] <= '2022-05-01')
df = df.loc[mask]

df.index = df.pop('Date')

print(df)

fig = plt.figure(figsize=(12, 10))
plt.subplot(211)
plt.plot(df.index, df['Close'])
#plt.legend(['Price'])

df2 = pd.read_csv('data/processed_sentiments.csv')
df2.index = df2.pop('Date')
print(df2)

plt.subplot(212)
plt.plot(df2.index, df2['Compound'])
plt.plot(df2.index, df2['Positive'])
plt.plot(df2.index, df2['Negative'])
plt.plot(df2.index, df2['Polarity'])
#plt.legend(['Compound', 'Positive', 'Negative'])

plt.show()






