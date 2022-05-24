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

df2 = pd.read_csv('data/processed_sentiments.csv')
#df2.index = df2.pop('Date')
print(df2)

#fig = plt.figure(figsize=(12, 10))
plt.subplot(111)
#plt.plot(df2['Date'], df2['Compound'])
plt.plot(df2['Date'], df2['Positive'], color = 'red')
plt.plot(df2['Date'], df2['Negative'], color = 'black')
plt.plot(df2['Date'], df2['Neutral'], color = 'blue')
#plt.legend(['Compound', 'Positive', 'Negative', 'Neutral'])
plt.legend(['Positive', 'Negative', 'Neutral'])

plt.show()
