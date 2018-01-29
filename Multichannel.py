

import pandas as pd
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
from nltk.stem import WordNetLemmatizer
import re

from keras import backend as K
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution1D, MaxPooling1D
from keras.utils import np_utils
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.layers import GRU,GlobalMaxPool1D
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.layers.merge import concatenate
from keras.models import load_model


def lemmatize_all(sentence):
    wnl = WordNetLemmatizer()
    for word, tag in pos_tag(word_tokenize(sentence)):
        if tag.startswith("NN"):
            yield wnl.lemmatize(word, pos='n')
        elif tag.startswith('VB'):
            yield wnl.lemmatize(word, pos='v')
        elif tag.startswith('JJ'):
            yield wnl.lemmatize(word, pos='a')
        elif tag.startswith('R'):
            yield wnl.lemmatize(word, pos='r')
            
        else:
            yield word


def msgProcessing(raw_msg):
    m_w=[]
    words2=[]
    raw_msg=str(raw_msg)
    raw_msg = str(raw_msg.lower())
    raw_msg=re.sub(r'[^a-zA-Z]', ' ', raw_msg)
    
    words=raw_msg.lower().split()
    #Remove words with length lesser than 3
    for i in words:
        if len(i)>=0:
            words2.append(i)
    stops=set(stopwords.words('english'))
    m_w=" ".join([w for w in words2])
    return(" ".join(lemmatize_all(m_w)))


def helperFunction(df):
    print ("Data Preprocessing!!!")
    cols=['comment_text']
    df=df[cols]
    df.comment_text.replace({r'[^\x00-\x7F]+':''},regex=True,inplace=True)
    num_msg=df[cols].size
    clean_msg=[]
    for i in range(0,num_msg):
        clean_msg.append(msgProcessing(df['comment_text'][i]))
    df['Processed_msg']=clean_msg
    X=df['Processed_msg']
    print ("Data Preprocessing Ends!!!")
    return X


def embedding(train,test):
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(train)
    t=len(tokenizer.word_index)+1
    trainsequences = tokenizer.texts_to_sequences(train)
    traindata = pad_sequences(trainsequences, maxlen=100)
    testsequences = tokenizer.texts_to_sequences(test)
    testdata = pad_sequences(testsequences, maxlen=100)
    return traindata, testdata,tokenizer





def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.strip().split()) for o in open('C:\\Users\\rohit.a\\Desktop\\kaggle\\glove.6B.50d.txt\\glove.6B.50d.txt',encoding="utf8"))

all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()

def x1(tokenizer):
    word_index = tokenizer.word_index
    nb_words = min(10000, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, 50))
    for word, i in word_index.items():
        if i >= 10000: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
    return embedding_matrix




def getTarget(y):
    ytrain=y[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values
    return ytrain




def multi_channel_model(xtrain, ytrain):
    batch_size=500
    epochs=10
    input1=Input(shape=(100,))
    embedding1=Embedding(10000, 50, weights=[embedding_matrix])(input1)
    conv1=Conv1D(filters=32, kernel_size=3, activation='relu')(embedding1)
    drop1= Dropout(0.4)(conv1)
    pool1=MaxPooling1D(pool_size=4)(drop1)
    gru1= GRU(100, dropout=0.2, recurrent_dropout=0.2)(pool1)
    
    
    input2=Input(shape=(100,))
    embedding2=Embedding(10000, 50, weights=[embedding_matrix])(input2)
    conv2=Conv1D(filters=32, kernel_size=4, activation='relu')(embedding2)
    drop2= Dropout(0.45)(conv2)
    pool2=MaxPooling1D(pool_size=4)(drop2)
    gru2= GRU(100, dropout=0.2, recurrent_dropout=0.2)(pool2)
    
    
    input3=Input(shape=(100,))
    embedding3=Embedding(10000, 50, weights=[embedding_matrix])(input3)
    conv3=Conv1D(filters=32, kernel_size=5, activation='relu')(embedding3)
    drop3= Dropout(0.5)(conv3)
    pool3=MaxPooling1D(pool_size=4)(drop3)
    gru3= GRU(100, dropout=0.2, recurrent_dropout=0.2)(pool3)
    
    
    merged= concatenate([gru1,gru2,gru3])
    dense1 = Dense(100, activation='relu')(merged)
    outputs = Dense(6, activation='sigmoid')(dense1)
    model = Model(inputs=[input1, input2, input3], outputs=outputs)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit([xtrain,xtrain,xtrain],ytrain,batch_size=batch_size,epochs=epochs)
    model.save("MultiChannel.h5")

def validate(xtest):
    model=load_model("MultiChannel.h5")
    pred=model.predict([xtest,xtest,xtest])
    return pred


def saveCSV(ytest):
    sample_submission = pd.read_csv("C:\\Users\\rohit.a\\Desktop\\kaggle\\sample_submission.csv")
    sample_submission[classes] = ytest
    sample_submission.to_csv("Multichanneltoxic.csv", index=False)

classes=["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
df= pd.read_csv("C:\\Users\\rohit.a\\Desktop\\kaggle\\train.csv",encoding="ISO-8859-1")
df2=pd.read_csv("C:\\Users\\rohit.a\\Desktop\\kaggle\\test.csv",encoding="ISO-8859-1")
df2['comment_text'].fillna('Missing',inplace=True)

X=helperFunction(df)
X2=helperFunction(df2)

xtrain,xtest,tokenizer=embedding(X,X2)
embedding_matrix=x1(tokenizer)
ytrain=getTarget(df[classes])

multi_channel_model(xtrain,ytrain)
ytest=validate(xtest)

saveCSV(ytest)

