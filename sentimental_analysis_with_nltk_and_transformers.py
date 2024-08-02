import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

plt.style.use('ggplot')

df = pd.read_csv('data2/Reviews.csv')
#print(df.shape)
df = df.head(5000)

ax = df['Score'].value_counts().sort_index().plot(kind='bar', title='Count of Reviews by Stars', figsize=(10, 5))
ax.set_xlabel('Review Stars')
#plt.show
 
example = df['Text'][50]
#print(example)
tokens = nltk.word_tokenize(example)
#print(tokens[:10])
tagged = nltk.pos_tag(tokens)
#print(tagged[:10])

entities = nltk.chunk.ne_chunk(tagged)
#print(entities)



# VADER Model

from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm import tqdm

sia = SentimentIntensityAnalyzer()
#print(sia)
#sia.polarity_scores(example)
#sia.polarity_scores("I love Furkan abi and TS!")
#sia.polarity_scores("This is the worst thing I have ever seen!")
#sia.polarity_scores(example)

results = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    text = row['Text']
    myid = row['Id']
    results[myid] = sia.polarity_scores(text)

vaders = pd.DataFrame(results).T
vaders = vaders.reset_index().rename(columns={'index': 'Id'})
vaders = vaders.merge(df, how='left')
#vaders.head()

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
sns.barplot(data=vaders, x='Score', y='pos', ax=axs[0], color='green')
sns.barplot(data=vaders, x='Score', y='neu', ax=axs[1], color='gray')
sns.barplot(data=vaders, x='Score', y='neg', ax=axs[2], color='red')
axs[0].set_title('Positive')
axs[1].set_title('Neutral')
axs[2].set_title('Negative')
plt.tight_layout
#plt.show()



# RoBERTa Model

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax

MODEL = f'cardiffnlp/twitter-roberta-base-sentiment'
tokeniser = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

print(example)
sia.polarity_scores(example)

def polarity_scores_roberta(example):
    encoded = tokeniser(example, return_tensors='pt')
    output = model(**encoded)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg': scores[0],
        'roberta_neu': scores[1],
        'roberta_pos': scores[2]
    }
    return scores_dict

res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    try:
        text = row['Text']
        myid = row['Id']
        vader_result = sia.polarity_scores(text)
        vader_result_rename = {}
        for key, value in vader_result.items():
            vader_result_rename[f'vader_{key}'] = value
        roberta_result = polarity_scores_roberta(text)
        both = {**vader_result_rename, **roberta_result}
        res[myid] = both
    except RuntimeError:
        print(f'Broke for id {myid}')

print(res)

results_df = pd.DataFrame(res).T
results_df = results_df.reset_index().rename(columns={'index': 'Id'})
results_df = results_df.merge(df, how='left')

sns.pairplot(data=results_df, vars=['vader_pos', 'vader_neu', 'vader_neg', 'roberta_pos', 'roberta_neu', 'roberta_neg'], hue='Score', palette='tab10')
plt.show()

# positive 1-star review
results_df.query('Score == 1').sort_values('roberta_pos', ascending=False)['Text'].values[0]
results_df.query('Score == 1').sort_values('vader_pos', ascending=False)['Text'].values[0]

# negative 5-star review
results_df.query('Score == 5').sort_values('roberta_neg', ascending=False)['Text'].values[0]
results_df.query('Score == 5').sort_values('vader_neg', ascending=False)['Text'].values[0]



# The Transformers Pipeline

from transformers import pipeline

sent_pipeline = pipeline('sentiment-analysis', device=0)
#sent_pipeline('I love Furkan abi and TS!')
#sent_pipeline('This is the worst thing I have ever seen!')
input_text = str(input('Enter a text to analyse: '))
sent_pipeline(input_text)
