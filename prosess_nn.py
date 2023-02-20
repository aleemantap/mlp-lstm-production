from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Activation, Dropout
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import itertools
from tensorflow import keras
from keras import backend as K
import numpy as np
import pandas as pd
import pickle
import joblib
import re
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import encode
import io
import base64
import json


def load_data(filename):
    #df = pd.read_csv(filename, sep='\t',  names = ['Kalimat','Kategori'])
    df = pd.read_csv('train_preprocess.tsv.txt',sep='\t', header=None)
    return df


def lowercase(text):
    return text.lower()

def remove_unnecessary_char(text):
    text = re.sub('\n','', text)
    text = re.sub('rt','', text)
    text = re.sub('user','', text)
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))',' ',text)
    text = re.sub('  +',' ', text)
    return text

def remove_nonaplhanumeric(text):
    text = re.sub('[^0-9a-zA-Z]+', ' ', text)
    text = re.sub('  +',' ', text) 
    return text


# Untuk Proses Cleaning Data
def preprocess(text):
    text = lowercase(text) # 1
    text = remove_unnecessary_char(text) # 2
    text = remove_nonaplhanumeric(text) # 3
    return text

def process_text(input_text):
    try: 
        output_text = preprocess(input_text)
        return output_text
    except:
        print("Text is unreadable")

def no_process_text(input_text):
    return input_text
   
def prosess(test_size, epochs, filename, cleaning):
    
    #Tahap Preproccessing
    """ Load Data Tweet """
    #df = pd.read_csv(filename, sep='\t',  names = ['Kalimat','Kategori'])
    df = load_data(filename)
    """ hapus baris yang mengandung nan """
    df.dropna(axis=0,  inplace=True)
    df.rename(columns = {0:'text', 1:'labels' }, inplace = True)
    
    if(cleaning=='yes'):
        df['clean_text'] = df.text.apply(process_text) 
    else :
        df['clean_text'] = df.text.apply(no_process_text) 
    
    data_preprocessed = df.text_clean.tolist()
    
    # proced ekstraksi fitur
    count_vect = CountVectorizer()
    count_vect.fit(data_preprocessed)

    X = count_vect.transform(data_preprocessed)
    
    #proses dump / simpan data vount vector
    pickle.dump(count_vect, open("feature_countvetorizer_nn.pickle", "wb"))
    
    #one hot encoding label
    classes = df['labels']
    y = pd.get_dummies(classes).values
    
    X_train, X_test, y_train, y_test = train_test_split(X, classes, test_size = test_size)
    
    model = MLPClassifier(hidden_layer_sizes=(100,100,100), max_iter=epochs, alpha=0.0001,
                     solver='sgd', verbose=10,  random_state=21,tol=0.000000001)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    # simpan model NN 
    pickle.dump(clf, open("model_countvetorizer_nn.pickle", "wb"))
  
    return 1
    


def cleansing_test(sent):
     return preprocess(sent)
    
def no_cleansing_test(sent):
    return sent
    
    
def testing_raw_text_nn(x,clen):

    text = None
    if(clen=='yes'):
        text = [cleansing_test(x)]
    else :
        text = [no_cleansing_test(x)]

    # Load feature_countvetorizer_nn 
    with open('feature_countvetorizer_nn.pickle', 'rb') as f:
        cv = pickle.load(f)

    # Feature Extraction
    te = cv.transform(text)
    
    #print(te)
    
    #load model NN
    with open('model_countvetorizer_nn.pickle', 'rb') as f:
        clf = pickle.load(f)
        
    result = clf.predict(te)[0]
    
    return [text,result]


