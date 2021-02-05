import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv('C:\\Users\\WhiteGhost\\Desktop\\news.csv')  # Loading the news.csv file to Data Frame df
print("Total number of rows and column in data frame(rows, column) : ", end=" ")
print(df.shape)  # df.shape -- Get the number of rows and columns
print(df.head())  # df.head() -- head() method is used to return top n (5 by default) rows of a data frame or series
labels = df.label
print(labels.head())
# text column of news.csv file is represented by df['text']
x_train, x_test, y_train, y_test = train_test_split(df['text'], labels, test_size=0.2, random_state=7)
# max_df i.e maximum document frequency, it will ignore the term that have higher df than 0.7
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(x_train)  # Fit and Transform vectorizer on the train set
tfidf_test = tfidf_vectorizer.transform(x_test)       # Transform vectorizer on the test set
pac = PassiveAggressiveClassifier(max_iter=50)  # The maximum number of passes over the training data is 50
pac.fit(tfidf_train, y_train)   # Fit linear model with Passive Aggressive algorithm
y_pred = pac.predict(tfidf_test)  # Predict class labels for samples in tfidf_test
""" comments- SKLearn.metrics has a method accuracy_score(), which returns “accuracy classification score”. 
    What it does is the calculation of “How accurate the classification is" """
score = accuracy_score(y_test, y_pred)
print(f'Passive Aggressive Classifier Accuracy: {round(score*100,2)}%')
print("confusion_ matrix : ", end="\n")
print(confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL']))
mnb = MultinomialNB()
mnb.fit(tfidf_train, y_train)     # Fit linear model with MultinomialNB Bayes algorithm
y_pred = mnb.predict(tfidf_test)  # Predict class labels for samples in tfidf_test
score = accuracy_score(y_test, y_pred)
print(f'MutlinomialNB Accuracy: {round(score*100,2)}%')
print("confusion_ matrix : ", end="\n")
print(confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL']))






