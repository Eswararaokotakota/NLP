import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
paragraph = """Stemming, in Natural Language Processing (NLP), refers to the process of reducing a word to its word stem that affixes to suffixes and prefixes or the roots. 
While a stemming algorithm is a linguistic normalization process in which the variant forms of a word are reduced to a standard form."""

sentences = nltk.sent_tokenize(paragraph)##will gives the sentences of paragraph in list format
stemmer = PorterStemmer() ##creating a stemming object
for i in range(len(sentences)):
    words = nltk.word_tokenize(sentences[i]) ##superating the sentences in to words
    words = [stemmer.stem(word) for word in words if word not in set(stopwords.words('english'))]
    ##Here we are getting the words
    ##[""which are not listed in the predefined stopwords(in nltk library(distionary called lemma)) are stemmed and then stored in the words object]
    ##(stopwords means: of, is, the, and, then, all, etc.)
    sentences[i]=' '.join(words) #here we are again making the sentences without stop words and with the stemmed words(ex:hystory -- histori)
    
print(sentences)