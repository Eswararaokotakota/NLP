import nltk

paragraph ="""The bag-of-words model is a simplifying representation used in natural language processing and information retrieval (IR). 
In this model, a text (such as a sentence or a document) is represented as the bag (multiset) of its words, disregarding grammar and even word order but keeping multiplicity. The bag-of-words model has also been used for computer vision.
The bag-of-words model is commonly used in methods of document classification where the (frequency of) occurrence of each word is used as a feature for training a classifier.
An early reference to "bag of words" in a linguistic context can be found in Zellig Harris's 1954 article on Distributional Structure.
The Bag-of-words model is one example of a Vector space model."""

import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

stemmer = PorterStemmer()
lammetizer = WordNetLemmatizer()
sentences = nltk.sent_tokenize(paragraph)
corpus = []

for i in range(len(sentences)):
    review = re.sub('[^a-zA-Z]', " ", sentences[i]) ##Removing the other things than small and capital (a to z) and replace with blank
    review = review.lower() ##Lowering the words
    review = review.split()  ##splitting every word (gives a list of words)
    review = [stemmer.stem(word) for word in review if not word in set(stopwords.words('english'))] ##stemming the words after removing the stopwords
    review =" ".join(review)##Joining the words and making our sentences back 
    corpus.append(review)
    ##Finally we have done the cleaning process of the data. like Removing the unwanter symbols and lowering every words and doing stemming. Hence the data is cleaned
print(sentences) ##original sentences
print(corpus) ##Cleaned data    
##creating the Bag of Words model (creating a document matrix)
from sklearn.feature_extraction.text import CountVectorizer ##this library is used to create bag of words
cv = CountVectorizer()
bow = cv.fit_transform(corpus).toarray() ##This is the BagOfWords data which we will provide this to the model to the machine learning model 
##(for projects like sentimental analysis,possitive&negative comment detection etc. )
print(bow)