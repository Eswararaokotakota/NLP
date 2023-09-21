##In this code we are training a nlp model to detect spam message.
##and this is my first code of training an nlp model from (https://www.youtube.com/watch?v=fA5TSFELkC0&list=PLZoTAELRMXVMdJ5sqbCK2LiM0HhQVWNzm&index=10&ab_channel=KrishNaik)
import pandas as pd

messages = pd.read_csv(r"E:\Coding\Python\NLP\Projects\Spam_Classifier\SpamClassifier_Data\smsspamcollection\SMSSpamCollection", sep='\t',  
                       names=["label", "message"]) #we have given the names to the each column as |label|message|
##"\t" means here it will superate the data into two columns where tab space exists in the text. 
##In case tab space exists at lable and message open the input file and look it by opening it in notepad


import re
import nltk
# nltk.download('stopwords')  # if we dosent dowloaded the (nltk.download)

####Data cleaning####
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer ##i have used already stemmer for the first time and got accuracy about 0.9847
from nltk.stem import WordNetLemmatizer  #This lammatizer and 
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
corpus=[]
print("Please wait, It will take some time to process all messages...")
for i in range(len(messages)):
    review = re.sub('[^a-zA-Z]'," ", messages['message'][i])
    review = review.lower()
    review = review.split()
    # review = [stemmer.stem(word) for word in review if not word in stopwords.words('english')] ### here we can also use lemmatization for better results
    ##stemming is used because it will take less amount of time to process compared to lemmatization
    review = [lemmatizer.lemmatize(word) for word in review if not word in stopwords.words('english')]  ##now iam using the lemmatization
    review = " ".join(review)  ##joining back our words Thus recreating the filtered sentences
    corpus.append(review)
###########Data cleaning is completed##########

##Lets do the  second level of preprocessing (here we are using Bag Of Words "basic")
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer ##here we are using tf_idf technique for level2 preprocessing
# cv = CountVectorizer(max_features=5000)
Tf = TfidfVectorizer(max_features=5000)
# bow = cv.fit_transform(corpus).toarray()
Tf_idf = Tf.fit_transform(corpus).toarray()
print("--Bag of words : Done..!")


##Now the data available has ham and spam but the computer understands it in 1,0  so lets convert them into 0's and 1's
y = pd.get_dummies(messages['label']) ##ham and smap strings are available in lables column
y = y.iloc[:,1].values #here it will takes only one column from dummies because hame has one column and spam has another column if we take one it is enough for training the model 

##Let's Split the data for Training and Testing
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(Tf_idf,y, test_size=0.20, random_state=0)
print("--Train Test Split : Done..!")

##Let's train our model
from sklearn.naive_bayes import MultinomialNB   #here we have used the basic model for nlp which is naive_bayes
spam_detect_model = MultinomialNB().fit(x_train, y_train)
print("--Model Training : Done..!")

#Now it's time to predict the data using our model
y_pred = spam_detect_model.predict(x_test)

##Let's do confusion matrix for understanding the detection performance of our model
from sklearn.metrics import confusion_matrix
confusion_m = confusion_matrix(y_test,y_pred)
print("--Confusion matrix : ","\n",confusion_m)

##Let's find the accuracy score fo our model
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
print("--Accuracy Scode : ", accuracy)

# print("Done...!")