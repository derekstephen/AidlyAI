# coding=utf-8

'''
Example 3
Simple pipeline for text cleaning
https://github.com/jbesomi/texthero#3-simple-pipeline-for-text-cleaning
'''
import texthero as hero
import pandas as pd

text = "This sèntencé    (123 /) needs to [OK!] be cleaned!   "

s = pd.Series(text)

print(s)

s = hero.remove_digits(s)

print(s)

s = hero.remove_brackets(s)

print(s)

s = hero.remove_diacritics(s)

print(s)

s = hero.remove_punctuation(s)

print(s)

s = hero.remove_whitespace(s)

print(s)

s = hero.remove_stopwords(s)

print(s)
