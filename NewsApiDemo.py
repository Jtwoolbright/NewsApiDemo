import spacy
from newsapi import NewsApiClient

nlp_eng = spacy.load('en_core_web_sm')
newsapi = NewsApiClient (api_key='a14ae82fe9694e75b32ee020ea0619a5')

temp = newsapi.get_everything(q='coronavirus', language='en', from_param='2020-09-28', to='2020-10-01', sort_by='relevancy', page_size=100)
articles = temp['articles']

import pandas as pd

dataset = []

for i in range(100):
    title = articles[i].get("title")
    description = articles[i].get("description")
    content = articles[i].get("content")
    dataset.append({'title':title, 'desc':description, 'content':content})
        
df = pd.DataFrame(dataset)
df = df.dropna()

from collections import Counter

def get_keywords_eng(text):
    result = []
    content_text = nlp_eng(text)
    pos_tag = ['NOUN', 'PROPN', 'VERB']
    for token in content_text:
        if (token.text in nlp_eng.Defaults.stop_words or token.is_punct):
            continue
        if (token.pos_ in pos_tag):
            result.append(token.text)
    return result

results = []
for content in df.content.values:
    results.append([('#' + x[0]) for x in Counter(get_keywords_eng(content)).most_common(5)])

df['keywords'] = results

from wordcloud import WordCloud
import matplotlib.pyplot as plt

text = str(results)
wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
