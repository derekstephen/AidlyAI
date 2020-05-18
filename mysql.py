# -*- coding: utf-8 -*-
"""
Created on Thu May 17 23:26:37 2020

@author: Derek
"""

# Load Libraries & Dependencies
from nltk.corpus import stopwords
import nltk

import pandas as pd
import pymysql


def checkTableExists(dbcon, tablename):
    """Checks if MySQL Table Exists in Database."""
    pass


# Create MySQL Connection1
connection = pymysql.connect(host='', user='', password='', db='')
print('Connection Opened')

# Try MySQL Code
try:
    with connection.cursor() as cursor:
        pass
except RuntimeError as err:
    print('ERROR: ', err)
finally:
    connection.close()
    print('Connection Closed')
