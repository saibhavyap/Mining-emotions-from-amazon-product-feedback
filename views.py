from django.shortcuts import render
from django.contrib import messages
from user.models import Usermodel

import os, sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import numpy as np
import nltk
import string
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
# Modelling
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(font_scale=1.3)

def index(request):
    return render(request, "index.html")

def Home(request):
    return index(request)

def adminlogin(request):
    return render(request, "admin/adminlogin.html")

def adminloginaction(request):
    if request.method == 'POST':
        uname = request.POST['uname']
        passwd = request.POST['upasswd']
        if uname == 'admin' and passwd == 'admin':
            data = Usermodel.objects.all()
            return render(request, "admin/adminhome.html", {'data': data})
        else:
            messages.success(request, 'Incorrect Details')
            return render(request, "admin/adminlogin.html")
    return render(request, "admin/adminlogin.html")

def showusers(request):
    data = Usermodel.objects.all()
    return render(request, "admin/adminhome.html", {'data': data})

def AdminActiveUsers(request):
    if request.method == 'GET':
        id = request.GET.get('uid')
        status = 'activated'
        print("PID = ", id, status)
        Usermodel.objects.filter(id=id).update(status=status)
        data = Usermodel.objects.all()
        return render(request, "admin/adminhome.html", {'data': data})

def logout(request):
    return render(request, "admin/adminlogin.html")

def ML(request):
    df = pd.read_csv(os.path.join(BASE_DIR, 'media/Reviews.csv'))
    f"{df.shape[0]:,} Review"
    cols = ['Text', 'Score']
    df_text = df[cols].copy()
    df_text.head()
    df_text.drop_duplicates(inplace=True)
    df_text.reset_index(drop=True,inplace=True)
    df_text.duplicated().sum()
    df_text['target'] = [0 if i <3  else 1 for i in df_text.Score ]
    sns.countplot(df_text['target'][:10000])

    #NEG_N it will store the size of negative reviews or zeros on the data frame
    NEG_N = df_text.target.value_counts()[0]
    #so, df_pos contains the positive reviews text or text with value equals to 1 in target
    df_pos = df_text[df_text['target'] == 1]['Text'].sample(NEG_N, replace=False)
    #now we will make a new dataframe where the size of positive reviews is same as the size of negative reviews
    df_text_balanced = pd.concat([df_text.iloc[df_pos.index], df_text[df_text.target == 0]])
    df_text_balanced.reset_index(drop=True,inplace=True)

    sns.countplot(df_text_balanced['target'])
    plt.ylim(0,1500)

    stop_words = list(stopwords.words('english'))
    stemmer = SnowballStemmer("english")
    lemmatizer= WordNetLemmatizer()

    #Removing every not from stop words
    stop_words.remove('not')
    for i in stop_words:
        if "n't" or "n'" in i:
            stop_words.remove(i)

    def cleaning_text(Text):
        #Removing Stop Words
        Text=[i for i in str(Text).split() if i not in stop_words]
        
        #Removing special characters
        Text=[re.sub('[^A-Za-z0-9]+', '', str(i)) for i in Text]


        #lemmatizing each word
        Text=[lemmatizer.lemmatize(y) for y in Text]
        #print(Text)

        #stemming each word
        Text=[stemmer.stem(y) for y in Text]
        
        str1 = " " 
        Cleaned_Text=str1.join(Text)
        #Remove numbers
        Cleaned_Text=''.join([i for i in Cleaned_Text if not i.isdigit()])
        # return string  
        return (Cleaned_Text)
        
    df_text_balanced.Text=df_text_balanced.Text.apply(lambda text : cleaning_text(text))

    X = df_text_balanced.iloc[:, 0].values
    y = df_text_balanced.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    ## TFIDF embedding for the Description
    vectorizer = TfidfVectorizer() ## Write your code here
    # fit on training (such vectorizer will be saved for deployment)
    vectorizer.fit(X_train)
    # transform on training data
    X_train = vectorizer.fit_transform(X_train)
    # transform on testing data
    X_test = vectorizer.transform(X_test)
    # See the dimensions of your data embeddings before entering to the model
    X_train.shape,X_test.shape

    ## initialize your Model
    clf = LogisticRegression(solver='liblinear')
    # Fit your Model on the Training Dataset
    clf.fit(X_train, y_train)
    # Predict on Test data
    preds =  clf.predict(X_test)

    # Calculate Model Accuracy
    acc = accuracy_score(preds, y_test)
    print(f"Model Accuracy = {round(acc*100,2)}%")

    preds = clf.predict(X_test)
    cm = confusion_matrix(y_test, preds)

    context = {
    'confusion_matrix': cm,
    'preds': acc
    }

    return render(request, "admin/adminml.html", context)