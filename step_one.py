# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 02:55:49 2020
@author: ddetommaso12
"""

# Load Libraries & Dependencies
from nltk.corpus import stopwords, wordnet
from nltk.stem import snowball, WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import pandas as pd
import nltk


# PREPARE DATA CODE


def prep_text(mission):
    """Preps mission statement by tokenizing sentences and words."""
    sentences = nltk.sent_tokenize(mission)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    return sentences


# First time download stop words
nltk.download('stopwords')

# Load Stop Words
stop_words = stopwords.words('english')

# Import data
df = pd.read_csv("./Data/MISSION.csv")

# Separate Mission to Sentences
df["MISSION"] = df["F9_03_PZ_MISSION"].apply(lambda x: nltk.sent_tokenize(str(x).lower()))

# Split Mission to Sentences and then Words
df["WORD"] = df["F9_03_PZ_MISSION"].apply(lambda x: prep_text(str(x).lower()))

# Flatten Separated Words to one list
df["WORD"] = df["WORD"].apply(lambda column: [y for x in column for y in x])

# Remove Stop Words
df["WORD"] = df["WORD"].apply(lambda x: [item for item in x if item not in stop_words])


# END PREPARE DATA CODE

# START STEP ONE CODE


def get_wordnet_pos(treebank_tag):
    """Convert POS String to POS for Lemmatizer."""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:  # Default Option
        return wordnet.NOUN


def computeTFDict(mission):
    """ Returns a tf dictionary for each mission whose keys are all 
    the unique words in the mission and whose values are their 
    corresponding tf.
    """
    #Counts the number of times the word appears in review
    missionTFDict = {}
    for word in mission:
        if word in missionTFDict:
            missionTFDict[word] += 1
        else:
            missionTFDict[word] = 1
    #Computes tf for each word           
    for word in missionTFDict:
        missionTFDict[word] = missionTFDict[word] / len(mission)
    return missionTFDict





# First time download wordnet
nltk.download('wordnet')

# Create Porter Stemmer
stemmer = snowball.SnowballStemmer('english')

# Stem mission statements
df["STEMMER"] = df["WORD"].apply(lambda x: [stemmer.stem(word) for word in x])

# Get POS for each word to use in Lemmatizer
df["POS"] = df["WORD"].apply(lambda x: [nltk.pos_tag(x)])

# Create WordNet Lemmatization
wordnet_lemmatizer = WordNetLemmatizer()

# Flatten POS to one list
df["POS"] = df["POS"].apply(lambda column: [y for x in column for y in x])

# Lemmatization of Words
df["LEMMATIZATION"] = df["POS"].apply(lambda x: [wordnet_lemmatizer.lemmatize(pair[0], pos=get_wordnet_pos(pair[1])) for pair in x])

# Calc TF Dictionary for mission
df["TF"] = df["LEMMATIZATION"].apply(lambda x: computeTFDict(x))





# END STEP ONE CODE

# Separate Data to view easily

df_imp = df[["EIN", "NAME", "F9_03_PZ_MISSION", "MISSION", "WORD", "POS", "STEMMER", "LEMMATIZATION", "TF"]]
