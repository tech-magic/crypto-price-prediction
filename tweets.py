import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import nltk
import sys

import ssl
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from pathlib import Path

#################################
### Utility Functions
#################################

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('vader_lexicon')

def str_to_datetime(s):
    only_date = s.split(' ')[0]
    split = only_date.split('-')
    year, month, day = int(split[0]), int(split[1]), int(split[2])
    return datetime.datetime(year=year, month=month, day=day)

def containsAnExpectedHash(a):
    try:
        if (a is not None):
            a_lower = a.lower()
            for x in listOfHashtags:
                if (x.lower() in a_lower):
                    return True
    except:
        pass
    return False

#############################################
### Program Starts Here #####################
#############################################

df_orig = pd.read_csv('data/Bitcoin_tweets.csv')
print(list(df_orig))

df = df_orig[['user_name', 'date', 'text', 'hashtags']].copy()
df.drop_duplicates(subset=['text'])

df = df.tail(100000)

print('Total # of Records to process => ' + str(len(df)))

listOfHashtags = ['BTC' , 'BitCoin']

data = []

for index, row in df.iterrows():
    if ((index % 1000) == 0):
        print('Processed ' + str(index) + ' rows.')
    if(containsAnExpectedHash(row['hashtags'])):
        try:
            curr_date = str_to_datetime(row['date'])
            curr_text = row['text']
            analysis = TextBlob(curr_text)
            score = SentimentIntensityAnalyzer().polarity_scores(curr_text)
            neg = score['neg']
            neu = score['neu']
            pos = score['pos']
            comp = score['compound']
            polarity = analysis.sentiment.polarity 
            #print(polarity, neg, neu, pos, comp)
            data.append([ curr_date, curr_text, pos, neg, neu, comp, polarity ])
        except:
            continue
            #print("Oops!", sys.exc_info()[0], "occurred.")

df = pd.DataFrame(data, columns = ['Date', 'Text', 'Positive', 'Negative', 'Neutral', 'Compound', 'Polarity'])

df = df.groupby('Date')[[ 'Positive', 'Negative', 'Neutral', 'Compound', 'Polarity' ]].mean()

print(df)

outputPath = Path('data/processed_sentiments.csv')
df.to_csv(outputPath)
