# coding=utf-8

'''
Example 2
Text preprocessing, TF-IDF, K-means and visualization
https://github.com/jbesomi/texthero#2-text-preprocessing-tf-idf-k-means-and-visualization
'''
import texthero as hero
import pandas as pd

df = pd.read_csv(
    "https://github.com/jbesomi/texthero/raw/master/dataset/bbcsport.csv"
)

df['tfidf'] = (
    df['text']
    .pipe(hero.clean)
    .pipe(hero.tfidf)
)

df['kmeans_labels'] = (
    df['tfidf']
    .pipe(hero.kmeans, n_clusters=5)
    .astype(str)
)

df['pca'] = df['tfidf'].pipe(hero.pca)

hero.scatterplot(df, 'pca', color='kmeans_labels', title="K-means BBC Sport news")
