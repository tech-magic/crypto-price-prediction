# data downloaded from https://finance.yahoo.com/quote/BTC-USD/history

import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers

######################################
#### Utility Functions ###############
######################################

def str_to_datetime(s):
    split = s.split('-')
    year, month, day = int(split[0]), int(split[1]), int(split[2])
    return datetime.datetime(year=year, month=month, day=day)

def df_to_windowed_df(dataframe, first_date_str, last_date_str, n=3):
    first_date = str_to_datetime(first_date_str)
    last_date  = str_to_datetime(last_date_str)
    
    target_date = first_date
    
    dates = []
    X, Y = [], []
    
    last_time = False
    
    while True:
        df_subset = dataframe.loc[:target_date].tail(n+1)
        
        if len(df_subset) != n+1:
            print('Error: Window of size ' + n + ' is too large for date ' + target_date)
            return
            
        values = df_subset['Close'].to_numpy()
        x, y = values[:-1], values[-1]
        
        dates.append(target_date)
        X.append(x)
        Y.append(y)
        
        next_week = dataframe.loc[target_date:target_date+datetime.timedelta(days=7)]
        next_datetime_str = str(next_week.head(2).tail(1).index.values[0])
        next_date_str = next_datetime_str.split('T')[0]
        year_month_day = next_date_str.split('-')
        year, month, day = year_month_day
        next_date = datetime.datetime(day=int(day), month=int(month), year=int(year))
        
        if last_time:
            break
            
        target_date = next_date
        
        if target_date == last_date:
            last_time = True
    
    ret_df = pd.DataFrame({})
    ret_df['Target Date'] = dates
    
    X = np.array(X)
    for i in range(0, n):
        X[:, i]
        curr_heading = 'Target-{' + str(n-i) + '}' 
        ret_df[curr_heading] = X[:, i]
        
    ret_df['Target'] = Y
    
    return ret_df

def windowed_df_to_date_X_y(windowed_dataframe):
    df_as_np = windowed_dataframe.to_numpy()
    dates = df_as_np[:, 0]
    
    middle_matrix = df_as_np[:, 1:-1]
    X = middle_matrix.reshape((len(dates), middle_matrix.shape[1], 1))
    Y = df_as_np[:, -1]
    
    return dates, X.astype(np.float32), Y.astype(np.float32)


#############################################
### Program Starts Here #####################
#############################################

df_orig = pd.read_csv('data/BTC-USD.csv')
df = df_orig[['Date', 'Close']]

df['Date'] = df['Date'].apply(str_to_datetime)

df.index = df.pop('Date')
print(df.head())

#plt.plot(df.index, df['Close'])

windowed_df = df_to_windowed_df(df, '2014-09-20', '2022-05-21', n=3)
print(windowed_df.head())

dates, X, y = windowed_df_to_date_X_y(windowed_df)
print(dates.shape, X.shape, y.shape)

q_80 = int(len(dates) * .8)
q_90 = int(len(dates) * .9)

dates_train, X_train, y_train = dates[:q_80], X[:q_80], y[:q_80]

dates_val, X_val, y_val = dates[q_80:q_90], X[q_80:q_90], y[q_80:q_90]
dates_test, X_test, y_test = dates[q_90:], X[q_90:], y[q_90:]

fig = plt.figure(figsize=(12, 10))

plt.subplot(221)
plt.plot(dates_train, y_train, color = 'blue')
plt.plot(dates_val, y_val, color = 'orange')
plt.plot(dates_test, y_test, color = 'green')

plt.legend(['Train', 'Validate', 'Test'])
#plt.show()

model = Sequential([layers.Input((3, 1)),
                    layers.LSTM(64),
                    layers.Dense(32, activation='relu'),
                    layers.Dense(32, activation='relu'),
                    layers.Dense(1)])

model.compile(loss='mse', 
              optimizer=Adam(learning_rate=0.001),
              metrics=['mean_absolute_error'])

model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100)

train_predictions = model.predict(X_train).flatten()

plt.subplot(222)
plt.plot(dates_train, train_predictions, color = 'red')
plt.plot(dates_train, y_train, color='blue')
plt.legend(['Training Predictions', 'Training Observations'])
#plt.show()

val_predictions = model.predict(X_val).flatten()

plt.subplot(223)
plt.plot(dates_val, val_predictions, color = 'red')
plt.plot(dates_val, y_val, color='orange')
plt.legend(['Validation Predictions', 'Validation Observations'])
#plt.show()

test_predictions = model.predict(X_test).flatten()

#f2 = plt.figure()
plt.subplot(224)
plt.plot(dates_test, test_predictions, color = 'red')
plt.plot(dates_test, y_test, color = 'green')
plt.legend(['Testing Predictions', 'Testing Observations'])
plt.show()

