# wk8_PHW_201835518_전소영
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd
import numpy as np

corpus = [
    # query
    'corona vaccine fake news',
    # document 1
    'WHO is gathering the latest international multilingual scientific findings and knowledge on COVID-19. The global literature cited in the WHO COVID-19 database is updated daily (Monday through Friday) from searches of bibliographic databases, hand searching, and the addition of other expert-referred scientific articles. This database represents a comprehensive multilingual source of current literature on the topic. While it may not be exhaustive, new research is added regularly.',
    # document 2
    'A COVID-19 vaccine candidate made of tiny artificial particles could be more powerful than other leading varieties at triggering a protective immune response. When the team injected mice with the nanoparticle vaccine, the animals produced virus-blocking antibodies at levels comparable to or greater than those produced by people who had recovered from COVID-19. Mice that received the vaccine produced about ten times more of these antibodies than did rodents vaccinated only with the spike protein, on which many COVID-19 vaccine candidates rely.',
    # document 3
    'The rise of fake news in the American popular consciousness is one of the remarkable growth stories in recent years—a dizzying climb to make any Silicon Valley unicorn jealous. Just a few years ago, the phrase was meaningless. Today, according to a new Pew Research Center study, Americans rate it as a larger problem than racism, climate change, or terrorism.',
    # document 4
    '“Falsehood flies, and the Truth comes limping after it,” Jonathan Swift once wrote. It was hyperbole three centuries ago. But it is a factual description of social media, according to an ambitious and first-of-its-kind study published Thursday in Science. The massive new study analyzes every major contested news story in English across the span of Twitter’s existence—some 126,000 stories, tweeted by 3 million users, over more than 10 years—and finds that the truth simply cannot compete with hoax and rumor. By every common metric, falsehood consistently dominates the truth on Twitter, the study finds: Fake news and false rumors reach more people, penetrate deeper into the social network, and spread much faster than accurate stories.',
    # document 5
    'The anti-vaccination movement has gained traction online in recent years, and campaigners opposed to vaccination have moved their focus to making claims relating to the coronavirus. First, a video containing inaccurate claims about coronavirus vaccine trials, made by osteopath Carrie Madej, that has proved popular on social media. Carrie \'s video makes a false claim that the vaccines will change recipients\' DNA (which carries genetic information).\"The Covid-19 vaccines are designed to make us into genetically modified organisms.\" She also claims—without any evidence—that vaccines will \"hook us all up to an artificial intelligence interface\".'
]


vectorizer = TfidfVectorizer()
matrix = vectorizer.fit_transform(corpus)

print(vectorizer.vocabulary_)
# print(matrix.shape)
# print(matrix.toarray())

cosine_sim = linear_kernel(matrix, matrix)
print(cosine_sim)

data = pd.DataFrame(cosine_sim, columns=['Query', 'Document 1', 'Document 2', 'Document 3', 'Document 4', 'Document 5'])
print(data)
df = data.iloc[[0], :]
df = df.iloc[:, 1:]

print("\n\n#### Ranking ####")
print(df.rank(method='max', ascending=False, axis=1))
