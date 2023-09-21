import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

paragraph = """What Is Lemmatization? Lemmatization is a text pre-processing technique used in 
natural language processing (NLP) models to break a word down to its root meaning to identify similarities. 
For example, a lemmatization algorithm would reduce the word better to its root word, or lemme, good"""

sentense = nltk.sent_tokenize(paragraph)
lemmatizer = WordNetLemmatizer()

for i in range(len(sentense)):
    words = nltk.word_tokenize(sentense[i])
    words=[lemmatizer.lemmatize(word) for word in words if word not in set(stopwords.words('english'))]
    sentense[i] = ' '.join(words)
    
print(sentense)