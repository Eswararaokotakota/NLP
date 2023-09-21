import nltk

paragraph = """Term Frequency - Inverse Document Frequency (TF-IDF) is a widely used statistical method in natural language processing and information retrieval.
It measures how important a term is within a document relative to a collection of documents (i.e., relative to a corpus). 
Words within a text document are transformed into importance numbers by a text vectorization process.
There are many different text vectorization scoring schemes, with TF-IDF being one of the most common.
As its name implies, TF-IDF vectorizes/scores a word by multiplying the words Term Frequency (TF) with the Inverse Document Frequency (IDF)."""

import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
sentences = nltk.sent_tokenize(paragraph)
corpus=[]

for i in range(len(sentences)):
    review = re.sub('[^a-zA-Z]', " ", sentences[i]) ##Removing the other things than small and capital (a to z)
    review = review.lower() ##Lowering the words
    review = review.split()  ##splitting every word (gives a list of words)
    review = [lemmatizer.lemmatize(word) for word in review if not word in set(stopwords.words('english'))] ##stemming the words after removing the stopwords
    review =" ".join(review)##Joining the words and making our sentences back 
    corpus.append(review)
print(len(stopwords.words("english")))
print(sentences)
print(corpus)

from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer()
TfIdf = cv.fit_transform(corpus).toarray() ##will be given to the model for training purpose
print(TfIdf)##will be the final TF-IDF vectorized data
