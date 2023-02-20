from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SpatialDropout1D, LSTM, Dense
from keras import backend as K
import pickle
import re
import itertools
from tensorflow import keras
from keras import backend as K
from keras.models import load_model
import numpy as np
import pandas as pd
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
    
    
# def cleansing(text):
   # text = text.lower()
   # text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
   # return text
# def no_cleansing(text):
   # text = text.lower()
   # return text



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
    
    df = load_data(filename)
    """ hapus baris yang mengandung nan """
    df.dropna(axis=0,  inplace=True)
    df.rename(columns = {0:'text', 1:'labels' }, inplace = True)
   
    # Proses filtering
   
    if(cleaning=='yes'):
        df['clean_text'] = df.text.apply(process_text) 
    else :
        df['clean_text'] = df.text.apply(no_process_text) 
        
    #connver dataframe to list, u can use this approach or simply as "df.values". but i recomended this way
    neg = df.loc[df['labels'] == 'negative'].clean_text.tolist()
    neu = df.loc[df['labels'] == 'neutral'].clean_text.tolist()
    pos = df.loc[df['labels'] == 'positive'].clean_text.tolist()

    neg_label = df.loc[df['labels'] == 'negative'].labels.tolist()
    neu_label = df.loc[df['labels'] == 'neutral'].labels.tolist()
    pos_label = df.loc[df['labels'] == 'positive'].labels.tolist()
    
    #Count sentiment labels
    total_data = pos + neu + neg
    labels = pos_label + neu_label + neg_label

    print("Pos: %s, Neu: %s, Neg: %s" % (len(pos), len(neu), len(neg)))
    print("Total data: %s" % len(total_data))
    
    
    #make positive label variable
    df_pos = df[df['labels']=='positive']
    print(df_pos.shape)

    #make neutral label variable
    df_neu = df[df['labels']=='neutral']
    print(df_neu.shape)

    #make negative label variable
    df_neg = df[df['labels']=='negative']
    print(df_neg.shape)
    
    #Downsampling postive label becomes equal to negative 
    df_pos_downsampled = df_pos.sample(df_neg.shape[0])
    df_pos_downsampled.shape
    
    #Merge  balanced labels
    df_balanced = pd.concat([df_pos_downsampled,df_neu,df_neg])
    df_balanced.shape
    
    df_balanced['labels'].value_counts()
    
    #Convert all the balanced df to list
    neg = df_balanced.loc[df['labels'] == 'negative'].clean_text.tolist()
    neu = df_balanced.loc[df['labels'] == 'neutral'].clean_text.tolist()
    pos = df_balanced.loc[df['labels'] == 'positive'].clean_text.tolist()

    neg_label = df_balanced.loc[df['labels'] == 'negative'].labels.tolist()
    neu_label = df_balanced.loc[df['labels'] == 'neutral'].labels.tolist()
    pos_label = df_balanced.loc[df['labels'] == 'positive'].labels.tolist()
    
    total_data_balanced = pos + neu + neg
    labels_balanced = pos_label + neu_label + neg_label

    print("Pos: %s, Neu: %s, Neg: %s" % (len(pos), len(neu), len(neg)))
    print("Total data: %s" % len(total_data_balanced))
    print("Total labels: %s" % len(labels_balanced))
    
    # make tokenisasi from file
    max_features = 100000
    tokenizer = Tokenizer(num_words=max_features, split=' ', lower=True)
    token = tokenizer.fit_on_texts(total_data_balanced)
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("tokenizer.pickle has created!")

    X = tokenizer.texts_to_sequences(total_data_balanced)

    vocab_size = len(tokenizer.word_index)
    maxlen = max(len(x) for x in X)

    X = pad_sequences(X)
    with open('x_pad_sequences.pickle', 'wb') as handle:
         pickle.dump(X, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("x_pad_sequences.pickle has created!")
    
    
    #Convert Data target labels(text) to number  
    Y = pd.get_dummies(labels_balanced)
    Y = Y.values

    with open('y_labels.pickle', 'wb') as handle:
         pickle.dump(Y, handle, protocol=pickle.HIGHEST_PROTOCOL)
         print("y_labels.pickle has created!")
         
    file = open("x_pad_sequences.pickle",'rb')
    X = pickle.load(file)
    file.close()

    file = open("y_labels.pickle",'rb')
    y = pickle.load(file)
    file.close()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    #  proses iterasi training LSTM
    
    kf = KFold(n_splits=4,random_state=42,shuffle=True) 

    accuracies = []

    y = Y

    embed_dim = 100
    units = 64
    history =  None
    for iteration, data in enumerate(kf.split(X), start=1):

        #Splitting the data
        data_train   = X[data[0]]
        target_train = y[data[0]]

        data_test    = X[data[1]]
        target_test  = y[data[1]]
        #Making deep learning model architecture
        model = Sequential()
        model.add(Embedding(max_features, embed_dim, input_length=X.shape[1]))
        model.add(LSTM(units))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(3,activation='softmax'))

        adam = optimizers.Adam(learning_rate = 0.0005)
        #Compile the model 
        model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])
        #Earlystop
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
        #Training the model
        print(model.summary) 
        history = model.fit(data_train, target_train, epochs=epochs, batch_size=32, validation_data=(data_test, target_test), verbose=1,shuffle=True, callbacks=[es])
        
        #proses simpan model SLTM
    
        model.save('model_lstm.h5')
        print("Model has created!")
        
        #Predict the model 
        predictions = model.predict(X_test)
        y_pred = predictions

        # for the current fold only    
        accuracy = accuracy_score(y_test.argmax(axis=1), y_pred.argmax(axis=1))

        print("Training ke-", iteration)
        print(classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1)))
        
        pickle.dump(y_test, open("y_test_lstm.pickle", "wb"))
        pickle.dump(y_pred, open("y_pred_lstm.pickle", "wb"))
        pickle.dump(y_pred, open("X_test_lstm.pickle", "wb"))
        pickle.dump(history, open("history_lstm.pickle", "wb"))
        pickle.dump(model.summary, open("history_lstm.pickle", "wb"))
        pickle.dump(accuracy, open("accuracy_lstm.pickle", "wb"))
        print("======================================================")

        accuracies.append(accuracy)

    # this is the average accuracy over all folds
    average_accuracy = np.mean(accuracies)

    print()
    print()
    print()
    print("Rata-rata Accuracy: ", average_accuracy)
    
    # kf = KFold(n_splits=4,random_state=42,shuffle=True) 

    # accuracies = []

    # y = Y

    # embed_dim = 100
    # units = 64

    # for iteration, data in enumerate(kf.split(X), start=1):

        # #Splitting the data
        # data_train   = X[data[0]]
        # target_train = y[data[0]]

        # data_test    = X[data[1]]
        # target_test  = y[data[1]]
        # #Making deep learning model architecture
        # model = Sequential()
        # model.add(Embedding(max_features, embed_dim, input_length=X.shape[1]))
        # model.add(LSTM(units))
        # model.add(Dropout(0.2))
        # model.add(Flatten())
        # model.add(Dense(3,activation='softmax'))

        # adam = optimizers.Adam(lr = 0.0005)
        # #Compile the model 
        # model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])
        # #Earlystop
        # es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
        # #Training the model
        # print(model.summary) 
        # history = model.fit(data_train, target_train, epochs=50, batch_size=32, validation_data=(data_test, target_test), verbose=1,shuffle=True, callbacks=[es])
        # #Predict the model 
        # predictions = model.predict(X_test)
        # y_pred = predictions

        # # for the current fold only    
        # accuracy = accuracy_score(y_test.argmax(axis=1), y_pred.argmax(axis=1))

        # print("Training ke-", iteration)
        # print(classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1)))
        # print("======================================================")

        # accuracies.append(accuracy)

    # # rata - rata akurasi
    # average_accuracy = np.mean(accuracies)

    print()
    print()
    print()
    print("Rata-rata Accuracy: ", average_accuracy)
    
    
    return average_accuracy
    
    

def cleansing_test(sent):
     return preprocess(sent)
    
def no_cleansing_test(sent):
    return sent
    
    
    
def testing_raw_text_lstm(x,clen):
    
    file = open("x_pad_sequences.pickle",'rb')
    Xss = pickle.load(file)
    file.close()
    
    
    text = None
    if(clen=='yes'):
        text = [cleansing_test(x)]
    else :
        text = [no_cleansing_test(x)]
        
    sentiment = ['negative', 'neutral', 'positive']
    
    max_features = 100000
    sentiment = ['negative', 'neutral', 'positive']
    #tokenizer = Tokenizer(num_words=max_features, split=' ', lower=True)
    # Load tokenizer model tokenizer.pickle
    with open('tokenizer.pickle', 'rb') as f:
        tokenizer = pickle.load(f)
    
    predicted = tokenizer.texts_to_sequences(text)
    guess = pad_sequences(predicted, maxlen=Xss.shape[1])

    model = load_model('model_lstm.h5')
    prediction = model.predict(guess)
    polarity = np.argmax(prediction[0])
    hasil = sentiment[polarity]

    return [text[0],sentiment[polarity]]