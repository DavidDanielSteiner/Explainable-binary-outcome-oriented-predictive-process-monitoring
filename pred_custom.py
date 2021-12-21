# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 16:26:18 2021

@author: David Steiner
"""


from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_curve, matthews_corrcoef, f1_score


###Machine Learning Packakges###

#!pip install interpret
#!pip install xgboost
#!pip install catboost

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from catboost import CatBoostClassifier
from interpret.glassbox import ExplainableBoostingClassifier

###Deep Learning Packages###

#!pip install keras
#!pip install tensorflow --user

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout, Dense, Embedding, LSTM, SpatialDropout1D
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.layers import Conv1D, MaxPooling1D, Flatten
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Input, Dense, LSTM, GRU, GlobalAveragePooling1D, Embedding, Dropout, concatenate
from keras.utils.vis_utils import plot_model

from sklearn.utils import class_weight

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

import re
import tensorflow as tf
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
import time

# =============================================================================
# Evaluation Metrics
# =============================================================================

def get_evaluation_metrics(model, X_test, y_test, X_train, y_train, exec_time=False):
    """
    Evaluating ML and DL models on several quantitative metrics.
    Confusion Matrix, Accuracy, ROC-AUC, F1, MCC, Sensitivity, Specificity
    """
    
    try: # for classical machine learning models
        start_time = time.time()
        y_pred = model.predict(X_test)
        prediction_time = time.time() - start_time
        
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred_train = model.predict(X_train)


    except: #for deep learning models
        start_time = time.time()       
        y_pred = list(np.array(model.predict_classes(X_test).flat))
        prediction_time = time.time() - start_time
        
        y_pred_proba = list(np.array(model.predict(X_test).flat))
        y_pred_train = list(np.array(model.predict_classes(X_train).flat))
        
        
    print("Confusion Matrix:") 
    print(confusion_matrix(y_test,y_pred))
    print(classification_report(y_test,y_pred))

    acc =  round(accuracy_score(y_test, y_pred),3)
    print("Accuracy:",acc, "(Train:", round(accuracy_score(y_train, y_pred_train),3), ")")    

    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_proba, pos_label=1)
    auc = round(metrics.auc(fpr, tpr),3)
    print("AUC:", auc)
    
    f1 = round(f1_score(y_test, y_pred, average='macro'),3) #average='weighted'
    print("F1:", f1)
    
    mcc = round(matthews_corrcoef(y_test, y_pred),3)
    print("MCC:", mcc)
    
    cm = confusion_matrix(y_test,y_pred)
    
    sensitivity = round(cm[0,0]/(cm[0,0]+cm[0,1]),3)
    print('Sensitivity : ', sensitivity )
    
    specificity = round(cm[1,1]/(cm[1,0]+cm[1,1]),3)
    print('Specificity : ', specificity)
    
    if exec_time:
        return y_pred, auc, acc, f1, mcc, sensitivity, specificity,  prediction_time
    else:
        return y_pred, auc, acc, f1, mcc, sensitivity, specificity



# =============================================================================
# Deep Learning Training Parameters
# =============================================================================
def train_model(model, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray,
                verbose: bool = True, weighted: bool = False, epochs: int = 10, batch_size: int = 32):
    """
    Trains the deep learning models.
    Optinally usage of early stopping and class weighted training.
    """

    if weighted == True:
        class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
        class_weights = {0: class_weights[0], 1: class_weights[1]}       
    else:
        class_weights = {0: 1, 1: 1}
    print(class_weights)
    
    #if verbose:
    #    print(model.summary())
        
    #callbacks_list = [globalvar.earlystop]
        
    hist = model.fit(X_train, y_train, 
              epochs=epochs, batch_size=batch_size, 
              class_weight=class_weights,
              verbose=verbose, shuffle=False, 
              validation_data=(X_test, y_test),
              #validation_split=0.1
              callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.001)]
             )
    
    #score = model.evaluate(X_test, y_test, verbose=0)
    #print(score)
    #print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(score[0],score[1]))
    
    #model.save('Model')
    return model, hist


def plot_train_history(hist):
    """
    Plots DL train history of loss and accuracy
    """
    
    #Plot history for accuracy
    plt.subplot(1, 2, 1)
    plt.plot(hist.history['acc'])
    plt.plot(hist.history['val_acc'])
    plt.title('Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    #Plot history for loss
    plt.subplot(1, 2, 2)
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

# =============================================================================
# Deep Learning Classifiers
# =============================================================================


def get_lstm_clf(X_train):
    """
    Defines the LSTM model architecture for training as obtained by hyperparameter search
    """

    model = Sequential(name='LSTM')
    model.add(LSTM(352, input_shape=(X_train.shape[1], X_train.shape[2]), dropout=0.2, return_sequences=True))
    model.add(LSTM(512, input_shape=(X_train.shape[1], X_train.shape[2]), dropout=0.2, return_sequences=True))
    model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2]), dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy',
                  optimizer='sgd',
                  metrics=['AUC', 'acc'])
    
    print(model.summary())
    return model



def get_gru_clf(X_train):
    """
    Defines the GRU model architecture for training as obtained by hyperparameter search
    """
    
    model = Sequential(name='GRU')

    model.add(GRU(480, return_sequences=True, input_shape=(X_train.shape[1],X_train.shape[2])))  
    model.add(GRU(448,return_sequences=True))
    model.add(GRU(96))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', 
                  optimizer='sgd',
                  metrics = ['AUC', 'acc'])
    
    print(model.summary())
    return model


def get_dnn_clf(X_train):
    """
    Defines the DNN model architecture for training as obtained by hyperparameter search
    """
    
    model = Sequential()
    model.add(Dense(512,  activation='relu', input_dim=X_train.shape[1])) 
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='tanh'))
    model.add(Dense(32, activation='tanh'))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
   
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['AUC', 'acc'])
    
    print(model.summary())
    return model



def get_cnn_clf(X_train):
    """
    Defines the CNN model architecture for training as obtained by hyperparameter search
    """
    
    model = Sequential(name='CNN')

    model.add(Conv1D(8, kernel_size=18, padding='same', activation='tanh',
                     input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Conv1D(4, kernel_size=1, activation='relu',
                     input_shape=(X_train.shape[1], X_train.shape[2])))
    
    model.add(Dropout(0.3))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('tanh'))
    model.add(Dense(256))
    model.add(Activation('tanh'))
    model.add(Dropout(0.3))
    model.add(Dense(256))
    model.add(Activation('tanh'))
    model.add(Dense(64))
    model.add(Activation('tanh'))
    model.add(Dropout(0.3))
    model.add(Dense(128))
    model.add(Activation('tanh'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', 
                  metrics=['AUC', 'acc'],
                  optimizer='sgd')
    
    print(model.summary())
          
    return model
