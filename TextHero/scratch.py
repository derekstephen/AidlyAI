import texthero as hero
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

df = pd.read_csv("../Data/berks_NGOs.csv")

df['clean_mission'] = df['Mission_Statement'].pipe(hero.clean)
df['tokenize'] = df['clean_mission'].pipe(hero.tokenize)



