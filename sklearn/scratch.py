from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import texthero as hero

df = pd.read_csv("C:\\Users\\ddeto\\PycharmProjects\\AidlyAI\\Data\\berks_NGOs.csv")\

df['clean_mission'] = df['Mission_Statement'].pipe(hero.clean)

NUM_TOPICS = 10

vectorizer = CountVectorizer(min_df=5, max_df=0.9,
                             stop_words='english', lowercase=True,
                             token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')
data_vectorized = vectorizer.fit_transform(df["clean_mission"])

# Build a Latent Dirichlet Allocation Model
lda_model = LatentDirichletAllocation(n_components=NUM_TOPICS, max_iter=10, learning_method='online')
lda_Z = lda_model.fit_transform(data_vectorized)

text = "To put Judeo-Christian principles into practice through programs that build healthy spirit, mind, and body for all."
x = lda_model.transform(vectorizer.transform([text]))[0]
print(x, x.sum())

text2 = "To put Judeo-Christian principles into practice through programs that build healthy spirit, mind, and body for all."
clean_text2 = "put judeo christian principles practice programs build healthy spirit mind body"

x2 = lda_model.transform(vectorizer.transform([text2]))[0]
print(x2, x2.sum())

x2 = lda_model.transform(vectorizer.transform([clean_text2]))[0]
print(x2, x2.sum())


