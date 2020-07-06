# coding=utf-8
'''
Getting Started - Representation
https://texthero.org/docs/getting-started#representation
'''

import texthero as hero
import pandas as pd

df = pd.read_csv(
   "https://github.com/jbesomi/texthero/raw/master/dataset/bbcsport.csv"
)

df['clean_text'] = df['text'].pipe(hero.clean)

'''
TF-IDF Representation
'''

df['tfidf_clean_text'] = hero.tfidf(df['clean_text'])

'''
Dimensionality reduction with PCA

Used to visualize the data by maping each point to a two-dimensional representation with PCA.
Returns the combination of attributes that better account variance in the data.
'''

df['pca_tfidf_clean_text'] = hero.pca(df['tfidf_clean_text'])

'''
All steps above can be completed in one step.
'''
df['pca'] = (
            df['text']
            .pipe(hero.clean)
            .pipe(hero.tfidf)
            .pipe(hero.pca)
   )
