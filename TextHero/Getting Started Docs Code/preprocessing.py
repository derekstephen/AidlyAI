# coding=utf-8

'''
Getting Started - Preprocessing
https://texthero.org/docs/getting-started#getting-started
'''

import texthero as hero
import pandas as pd

df = pd.read_csv(
   "https://github.com/jbesomi/texthero/raw/master/dataset/bbcsport.csv"
)

print('-- dataframe head 2 --')
print(df.head(2))
print('----------------------')

'''
Clean

Clean method does the following:
    1. fillna(s) Replace not assigned values with empty spaces.
    2. lowercase(s) Lowercase all text.
    3. remove_digits() Remove all blocks of digits.
    4. remove_punctuation() Remove all string.punctuation (!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~).
    5. remove_diacritics() Remove all accents from strings.
    6. remove_stopwords() Remove all stop words.
    7. remove_whitespace() Remove all white space between words.

You can achieve same results with Pandas Pipe Function
    Code: df['clean_text'] = df['text'].pipe(hero.clean)
'''

df['clean_text'] = hero.clean(df['text'])

print('-- dataframe clean text --')
print(df.head(2))
print('----------------------')

'''
Custom Pipeline

Alternatively can be done with following code:
   df['clean_text'] = df['text'].pipe(hero.clean, custom_pipeline)
'''

from texthero import preprocessing

custom_pipeline = [preprocessing.fillna,
                   preprocessing.lowercase,
                   preprocessing.remove_whitespace]
df['clean_text'] = hero.clean(df['text'])

print('-- dataframe clean text custom pipeline --')
print(df.head(2))
print('----------------------')
