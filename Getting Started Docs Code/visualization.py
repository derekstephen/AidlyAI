# coding=utf-8
'''
Getting Started - Visualization
https://texthero.org/docs/getting-started#visualization
'''

import texthero as hero
import pandas as pd

df = pd.read_csv(
   "https://github.com/jbesomi/texthero/raw/master/dataset/bbcsport.csv"
)

df['clean_text'] = df['text'].pipe(hero.clean)

df['pca'] = (
            df['text']
            .pipe(hero.clean)
            .pipe(hero.tfidf)
            .pipe(hero.pca)
   )

'''
texthero.visaulization provides helper functions to visualize the tarnsformed dataframe
'''
hero.scatterplot(df, col='pca', color='topic', title="PCA BBC Sport news")

'''
Top Words

We can 'visualize' the most common words for each topic using top_words
'''
NUM_TOP_WORDS = 5
top_words = df.groupby('topic')['text'].apply(lambda x: hero.top_words(x)[:NUM_TOP_WORDS])

print('-- dataframe top words --')
print(top_words)
print('----------------------')
