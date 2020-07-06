# coding=utf-8

'''
Example 1
Text Cleaning, TF-IDF representation and visualization
https://github.com/jbesomi/texthero#1-text-cleaning-tf-idf-representation-and-visualization
'''
import texthero as hero
import pandas as pd

df = pd.read_csv(
   "https://github.com/jbesomi/texthero/raw/master/dataset/bbcsport.csv"
)

df['pca'] = (
   df['text']
   .pipe(hero.clean)
   .pipe(hero.tfidf)
   .pipe(hero.pca)
)
hero.scatterplot(df, 'pca', color='topic', title="PCA BBC Sport news")
