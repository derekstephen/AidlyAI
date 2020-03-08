#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 02:55:49 2020

@author: ddetommaso12
"""

# Load Libraries & Dependencies
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import string
import nltk
import re

# First time download stop words
nltk.download('stopwords')

# Load Stop Words
stop_words = stopwords.words('english')

# Import data
df = pd.read_csv("./Data/MISSION.csv")

# Remove unnecessary columns and rename mission column
df = df[['EIN', 'NAME', 'F9_03_PZ_MISSION']]
df = df.rename(columns={'F9_03_PZ_MISSION': 'MISSION'})

# PREPARE DATA FOR POS Analysis
# TODO: Make all text lowercase, remove punctuation, split words, remove stop words

# Convert to lowercase
df = df.apply(lambda x: x.astype(str).str.lower())



print(df)


