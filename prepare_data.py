# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 17:37:37 2020

@author: Derek
"""

# Load Libraries & Dependencies
from nltk.corpus import stopwords

import pandas as pd
import nltk


def prep_text(mission):
    """Preps mission statement by tokenizing sentences and words."""
    sentences = nltk.sent_tokenize(mission)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    return sentences


# First time download stop words + punkt
nltk.download('punkt')
nltk.download('stopwords')

# Load Stop Words
stop_words = stopwords.words('english')

# Import data
df = pd.read_csv("./Data/MISSION.csv")

# Split Mission to Sentences and then Words
df_missions = df["F9_03_PZ_MISSION"].apply(lambda x: prep_text(str(x).lower()))
df["MISSION"] = df_missions

# Flatten Separated Words to one List
df["WORDS"] = df["MISSION"].apply(lambda column: [y for x in column for y in x])

# Remove Stop Words
df["WORDS"] = df["WORDS"].apply(lambda x: [item for item in x if item not in stop_words])
