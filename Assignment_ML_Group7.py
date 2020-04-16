"""

GROUP ASSIGNMENT - DEMOGRAPHIC PREDICTION 

Group Members: 
    Margot Crespin - e197033
    Andrea De Angelis - e183312
    Mathilde Gomez - e197658
    Luna Joseph - e197041
    Alexandre Lachkar - e197042

"""
### IMPORT THE REQUIRED PACKAGES ###
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve


### IMPORT THE DATASETS PROVIDED ###
train = pd.read_csv('D:\ESCP\Machine learning with python\\Final assignment\\train.csv')
test = pd.read_csv('D:\ESCP\Machine learning with python\\Final assignment\\test.csv')


### UNDERSTAND THE DATA ###
# Remove rows with missing values for both datasets 
train.isnull().sum()
train = train.dropna(axis = 0, how= 'any')

test.isnull().sum()
test = test[test['keywords'].notna()]


### WORD FREQUENCY LOOP ###
### Loop to display number of times the word appeared in the URL in an appropriate format ###

# Train dataset, performed on 50 000 rows for convenience   
train_samp = train.sample(n=50000)
train_samp.reset_index(inplace=True)
del train_samp['index']

train_samp.head(10)

train_samp['keywords'] = train_samp['keywords'].str.split(pat=';')
train_samp['new_keyword']= 'na'

for i in range(0,len(train_samp)): 
    ff = pd.DataFrame(train_samp['keywords'][i], columns = ['word_seen']) 
    ff = ff['word_seen'].str.split(pat=':', expand = True)
    ff.columns = ['word', 'seen']
    ff['ID'] = train_samp['ID'][i]
    ff['seen']=pd.to_numeric(ff.seen)
    longueur = range(0,len(ff.word))
    for a in longueur:
        if ff.seen.iloc[a] > 1:
            for n in range(0,ff.seen.iloc[a]-1):
                ff= ff.append(ff.iloc[a])
    train_samp['new_keyword'].iloc[i] = np.array(ff.word)
    print(i)

del train_samp['keywords']

# Test dataset, performer on 50 000 rows for convenience
test_samp = test.sample(n=50000)
test_samp.reset_index(inplace=True)
del test_samp['index']

test_samp.head(10)

test_samp['keywords'] = test_samp['keywords'].str.split(pat=';')
test_samp['new_keyword']= 'na'

for i in range(37912,len(test_samp)): 
    ff = pd.DataFrame(test_samp['keywords'][i], columns = ['word_seen']) 
    ff = ff['word_seen'].str.split(pat=':', expand = True)
    ff.columns = ['word', 'seen']
    ff['ID'] = test_samp['ID'][i]
    ff['seen']=pd.to_numeric(ff.seen)
    longueur = range(0,len(ff.word))
    for a in longueur:
        if ff.seen.iloc[a] > 1:
            for n in range(0,ff.seen.iloc[a].astype(np.int64)-1):
                ff= ff.append(ff.iloc[a])
    test_samp['new_keyword'].iloc[i] = np.array(ff.word)
    print(i)

del test_samp['keywords']

# Extract new data to CSV (for collaboration purposes)
train_samp.to_csv('D:\ESCP\Machine learning with python\\Final assignment\\train_samp.csv', index = True)
test_samp.to_csv('D:\ESCP\Machine learning with python\\Final assignment\\test_samp.csv', index = True)

# Rename new datasets 
train = pd.read_csv('D:\ESCP\Machine learning with python\\Final assignment\\train_samp.csv')
train = train.drop(['Unnamed: 0'], axis=1)
test = pd.read_csv('D:\ESCP\Machine learning with python\\Final assignment\\test_samp.csv')
test = test.drop(['Unnamed: 0'], axis=1)


#### VISUALIZATION ####
### Visualize data before processing it ### 

#Dummify the sex variable for convenience 
train['sex']= train['sex'].replace('M',0)
train['sex']= train['sex'].replace('F',1)

print(train['ID'].value_counts())

# Check correlations between variables to verify multi-collinearity
sns.heatmap(train.corr(),annot = True, fmt='.0g',vmin=-1, vmax=1.2, center= 0, linewidths=2, linecolor='black')

train.hist()


#### PRE-PROCESSING KEYWORDS ####
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import SnowballStemmer
from nltk.stem.snowball import FrenchStemmer

### Below are the steps we performed to run the models 
#Step 1: Cleaning the Text and removing special characters
#Step 2: Putting all letters in lower cases
#Step 3: Removing all stopwords
#Step 4: Applying Stemming on all words to  keep only the root of the words
#Step 5: Joining back the words to make it a string 
#Step 6: Applying the cleaning to a larger dataset

corpus = []
Longueur = range(0,len(train)) #len(train)
for i in Longueur:
    try:
        recherche = re.sub('[^a-zA-Z]', ' ',str(train['new_keyword'][i]))
    except KeyError:
            continue
    recherche = recherche.lower()
    recherche = recherche.split()
    #ps = PorterStemmer()
    stemmer = SnowballStemmer('french')
    recherche = [stemmer.stem(word) for word in recherche if not word in set(stopwords.words(['french','english']))]
    recherche = ' '.join(recherche)
    corpus.append(recherche)
    print(i)
    
corpus = pd.DataFrame(corpus)
print(corpus.head(10))

train.head(10)
train['new_keyword'] = corpus

#### VECTORIZE SEX ####
sex = train[{'ID', 'new_keyword', 'sex'}]

### Parameters selection ###
min_df = 1
max_df = 1.
max_features = 1000

# Using CountVectorizer #
# Create the bag of words model: sparse matrice through tokenization
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(encoding='utf-8',
                        stop_words=None,
                        lowercase=False,
                        max_df=max_df,
                        min_df=min_df,
                        max_features=max_features)

X = cv.fit_transform(sex['new_keyword']).toarray()

#Create a dependent variable
y = sex.iloc[:,0].values

### MODEL: NAIVE BAYES ###
# Split the data set
from sklearn. model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fit Naive Bayes to the training set
from sklearn.naive_bayes import MultinomialNB
classifier_sex = MultinomialNB()
classifier_sex.fit(X_train, y_train)

# Predict the Test set results
y_pred = classifier_sex.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
cm

from sklearn.metrics import f1_score

acc_score = accuracy_score(y_test, y_pred)
print('the accuracy of NB is:',  (round(acc_score,4)*100),'%')

recallLog= recall_score(y_test, y_pred, average = 'macro')
print('The recall of the NB is:' , (round(recallLog,4)*100),'%')

PrecisionLog = precision_score(y_test, y_pred)
print('The Precision of the NB is:' , (round(PrecisionLog,4)*100),'%')

AUC = roc_auc_score(y_test, y_pred)
print('The AUC of the NB is:', AUC)

F1_score = f1_score(y_test, y_pred)
print('The F1 of the NB is:', F1_score)


#### VECTORIZE AGE ####
age = train[{'ID', 'new_keyword', 'age'}]
age

# Parameters selection
min_df = 1
max_df = 1.
max_features = 1000

# Create the bag of words model: sparse matrice through tokenization
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(encoding='utf-8',
                        stop_words=None,
                        lowercase=False,
                        max_df=max_df,
                        min_df=min_df,
                        max_features=max_features)

X = cv.fit_transform(age['new_keyword']).toarray()

#Create a dependent variable
y = age.iloc[:,1].values

### MODEL: NAIVE BAYES ###
# Split data set
from sklearn. model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fit Naive Bayes to the training set
from sklearn.naive_bayes import MultinomialNB
classifier_age = MultinomialNB()
classifier_age.fit(X_train, y_train)

# Predict the Test set results
y_pred = classifier_age.predict(X_test)

# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm

y_test
y_pred
mape = np.mean(np.abs((y_test - y_pred) / y_test) * 100)
print(mape)

print("Accuracy", 100-mape)


### APPLY BOTH MODELS ON THE TEST SET ####
corpus = []
Longueur = range(0,len(test))
for i in Longueur:
    try:
        recherche = re.sub('[^a-zA-Z]', ' ',str(test['new_keyword'][i]))
    except KeyError:
            continue
    recherche = recherche.lower()
    recherche = recherche.split()
    #ps = PorterStemmer()
    stemmer = SnowballStemmer('french')
    recherche = [stemmer.stem(word) for word in recherche if not word in set(stopwords.words(['french','english']))]
    recherche = ' '.join(recherche)
    corpus.append(recherche)
    print(i)
    
corpus = pd.DataFrame(corpus)
print(corpus.head(10))

test.head(10)
test['new_keyword'] = corpus

### SEX ###
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(encoding='utf-8',
                        stop_words=None,
                        lowercase=False,
                        max_df=max_df,
                        min_df=min_df,
                        max_features=max_features)

X = cv.fit_transform(test['new_keyword']).toarray()

# Predict the test set results
test['sex'] = classifier_sex.predict(X)

### AGE ###
# Predict the test set results
test['age'] = classifier_age.predict(X)

del test['new_keyword']
test.columns = ['ID','age_pred','sex_pred']
test['sex_pred']= test['sex_pred'].replace(0,'M')
test['sex_pred']= test['sex_pred'].replace(1,'F')

### EXPORT THE FINAL DATASET 
test.to_csv('D:\ESCP\Machine learning with python\\Final assignment\\final_table_pred.csv', index = True)
