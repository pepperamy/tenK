#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import _pickle as pickle
from collections import defaultdict
import re
import ast

from bs4 import BeautifulSoup

import sys
import os
import json
import operator

from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers

from keras.optimizers import SGD

from keras.layers import Embedding, Dense, Conv1D, MaxPooling1D, Dropout, Activation, Input, Flatten, Concatenate, GlobalAveragePooling1D, LSTM, GRU, Bidirectional, dot, multiply, Lambda, TimeDistributed, Masking

from keras.models import Model, Sequential
from keras.regularizers import l2,l1,l1_l2
from keras.callbacks import Callback,EarlyStopping, ModelCheckpoint
from sklearn.utils import class_weight

from keras.callbacks import Callback

from keras.layers.normalization import BatchNormalization
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve, classification_report,accuracy_score, auc, roc_curve, roc_auc_score, average_precision_score

from gensim.models import Word2Vec

from keras.initializers import Constant

from keras import optimizers

import matplotlib.pyplot as plt




MAX_PARA_LENGTH = 70
MAX_PARAS = 70
MAX_NB_WORDS = 50000
EMBEDDING_DIM = 100
#VALIDATION_SPLIT = 0.2
#lr = 0.0005
opt = optimizers.Adam(lr= 0.0005)
#l1_reg = l1(1e-5)
l12_reg = l1_l2(l1 = 1e-6,l2 = 1e-6)
BATCH_SIZE = 24
#metrics_auc = {}
metrics_prcs = {}
epochs_num = 200
#num_folder = 5
path = '/home/jujun/fraudprediction_10k/data/rm_name/'
hanpath = '/home/jujun/fraudprediction_10k/HAN/'
date = '20200421'
class_weights = {0: 1, 1:20.0}




def load_data(data_name):
    f = open(data_name,'rb')
    data = pickle.load(f)
    return data




labels = load_data(path + 'data_20200309_2012_rmname_smr_labels')

data = np.load(path + 'handata_20200417_smr.npy')

embedding_matrix = np.load(path + 'embedding_matrix_20200419.npy')

train_indecis = load_data(path + 'indices_train_20200214')
print(len(train_indecis))

test_indecis = load_data(path + 'indices_test_20200214')
print(len(test_indecis))

train_indecis_x, train_indecis_val = train_test_split(train_indecis, test_size=0.2, random_state=66)

X_train = data[train_indecis_x]
X_val = data[train_indecis_val]
X_test = data[test_indecis]

Y_train = pd.Series(labels)[train_indecis_x]
Y_val = pd.Series(labels)[train_indecis_val]
Y_test = pd.Series(labels)[test_indecis]

print("y train:", Y_train.value_counts())
print("y val:", Y_val.value_counts())
print("y test:", Y_test.value_counts())

val_sample_weights = []
for y in Y_val:
    if y == 1:
        val_sample_weights.append(class_weights[1])
    else: val_sample_weights.append(class_weights[0])
val_sample_weights = np.asarray(val_sample_weights)



def performance_measure(pred_yp, y):
    '''
    Given lists of predicted y probability and x, y, return a dataframe of AR, AUC, Brier, Decile Table
    '''
    
    tenc_dat = pd.DataFrame({'y_true':y,'probability':pred_yp.flatten()})
    tenc_dat.sort_values('probability',axis = 0,ascending=False, inplace = True)
    tenc_dat.index = range(0,len(tenc_dat))
    y = tenc_dat['y_true']
    point = float(len(tenc_dat))/10
    point = int(round(point))
    tenc = []
    for i in range(0,10):
        tenc.append(y[(i*point):((i+1)*point)])
    tenc[9]=tenc[9].append(y[10*point:])
    total = sum(y)
    num_of_bkr = []
    for j in range(0,10):
        num_of_bkr.append(sum(tenc[j]))
    tencile_bkr = np.array(num_of_bkr)
    rate = tencile_bkr.astype(float)/total

    return rate


class Evaluation(Callback):
    """ Show AUC after interval number of epoches """
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()
        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            logs['auc'] = score
            score_ap = average_precision_score(self.y_val, y_pred)
            logs['Avg_Prec'] = score_ap
            #tencile=performance_measure(y_pred, self.y_val)
            #logs['tencile'] = tencile
            print(" epoch:{:d} AUC: {:.4f}".format(epoch, score))
            print(" epoch:{:d} Avg_Prec: {:.4f}".format(epoch, score_ap))



class AttLayer(Layer):
    
    def __init__(self, regularizer=None,context_dim=100, name="attention",**kwargs):
        self.regularizer = regularizer
        self.context_dim=context_dim
        self.supports_masking = True
        self.name=name
        super(AttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3        
        self.W = self.add_weight(name='W', shape=(input_shape[-1], self.context_dim), initializer='normal', trainable=True, 
                                 regularizer=self.regularizer)
        self.b = self.add_weight(name='b', shape=(self.context_dim,), initializer='normal', trainable=True, 
                                 regularizer=self.regularizer)
        self.u = self.add_weight(name='u', shape=(self.context_dim,), initializer='normal', trainable=True, 
                                 regularizer=self.regularizer)        
        super(AttLayer, self).build(input_shape)  # be sure you call this somewhere!

    def call(self, x, mask=None):
        eij = K.tanh(K.dot(x, self.W) + self.b)
        eij = K.squeeze(K.dot(eij, K.expand_dims(self.u, axis=1)), axis=-1)
        ai = K.exp(eij)

        
        if mask is not None:
            ai*=K.cast(mask, K.floatx())
            
        ai /=K.cast(K.sum(ai, axis=1, keepdims=True)+K.epsilon(), K.floatx())

        return ai
        

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1])
    
    def get_config(self):
        config = {}
        base_config = super(AttLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_mask(self, inputs, mask):
        return None    


# class AttLayer(Layer):
#     def __init__(self, **kwargs):
#         self.init = initializers.get('normal')
#         #self.input_spec = [InputSpec(ndim=3)]
#         super(AttLayer, self).__init__(** kwargs)

#     def build(self, input_shape):
#         assert len(input_shape)==3
#         #self.W = self.init((input_shape[-1],1))
#         self.W = self.init((input_shape[-1],))
#         #self.input_spec = [InputSpec(shape=input_shape)]
#         self.trainable_weights = [self.W]
#         super(AttLayer, self).build(input_shape)  # be sure you call this somewhere!

#     def call(self, x, mask=None):
#         eij = K.tanh(K.dot(x, self.W))

#         ai = K.exp(eij)
#         weights = ai/K.sum(ai, axis=1).dimshuffle(0,'x')

#         weighted_input = x*weights.dimshuffle(0,1,'x')
#         return weighted_input.sum(axis=1)

#     def get_output_shape_for(self, input_shape):
#         return (input_shape[0], input_shape[-1])


class WeightedSum(Layer):
    def __init__(self, name="weighted_sum",  **kwargs):
        self.supports_masking = True
        self.name=name
        super(WeightedSum, self).__init__(**kwargs)
        
    def build(self, input_shape):
        super(WeightedSum, self).build(input_shape)  # Be sure to call this at the end


    def call(self, input_tensor, mask=None):
        
        x = input_tensor[0]
        #print("input",K.int_shape(x))
        a = input_tensor[1]
        print("weights", K.int_shape(a))

        a = K.expand_dims(a)
        weighted_input = K.sum(x * a, axis=1)
        #print("weighted sum",K.int_shape(weighted_input))
        
        return weighted_input

    def compute_output_shape(self, input_shape):
        
        a, b = input_shape
        
        return (a[0], a[-1])

    def compute_mask(self, x, mask=None):
        return None



def basicModel(embedding_matrix,MAX_NB_WORDS,MAX_PARA_LENGTH,MAX_PARAS):
    embedding_layer = Embedding( MAX_NB_WORDS+ 1,
                        EMBEDDING_DIM,
                        #weights = [embedding_matrix],
                        embeddings_initializer = Constant(embedding_matrix),
                        mask_zero=True,
                        input_length=MAX_PARA_LENGTH,
                        trainable=False)    
    
    
    para_input = Input(shape=(MAX_PARA_LENGTH, ), dtype='int32')
    embedded_sequences = embedding_layer(para_input)
    #norm_sequence = BatchNormalization()(embedded_sequences)
    l_lstm_sen = Bidirectional(GRU(70, return_sequences=True, implementation=2))(embedded_sequences)
    #l_dense_sen = TimeDistributed(Dense(140))(l_lstm_sen)
    #drop_out = Dropout(0.2)(l_lstm_sen)
    l_att = AttLayer()(l_lstm_sen)
    weighted_sum = WeightedSum()([l_lstm_sen,l_att])
    paraEncoder =Model(para_input,weighted_sum)
    paraEncoder.summary()
    
    doc_input = Input(shape=(MAX_PARAS, MAX_PARA_LENGTH), dtype='int32')
    doc_encoder = TimeDistributed(paraEncoder)(doc_input)
    #mask_doc = Masking(mask_value=0.0)(doc_encoder)
    #norm_doc = BatchNormalization()(mask_doc)
    l_lstm_para = Bidirectional(GRU(70, return_sequences=True, implementation=2))(doc_encoder)
    #l_dense_para = TimeDistributed(Dense(140))(l_lstm_para)
    #norm_doc_1 = BatchNormalization()(l_lstm_para)
    #drop_out = Dropout(0.2)(l_lstm_para) 
    l_att_para = AttLayer()(l_lstm_para)
    weighted_sum_doc = WeightedSum()([l_lstm_para, l_att_para])
    #batch_norm = BatchNormalization()(weighted_sum_doc)
    #drop_out = Dropout(0.2)(batch_norm)

    preds = Dense(1, activation='sigmoid',kernel_regularizer=l12_reg)(weighted_sum_doc) 

    model = Model(doc_input, preds)
    model.summary()
    
    return model


def trainModel(x_train, y_train, Model_Filepath, model,epochs_num, x_val, y_val, val_sample_weights, class_weights):
    
    
    model.compile(loss='binary_crossentropy',
              optimizer= opt,
              metrics=['acc'])

    
    auc_ap_eval = Evaluation(validation_data=(x_val, y_val), interval=1)
    #precision_eval = PrecisionEvaluation(validation_data=(x_train, y_train), interval=1)
    
    earlyStopping = EarlyStopping(monitor='auc',patience = 5, verbose =2, mode ='max')
    #checkpoint = ModelCheckpoint(Model_Filepath,save_weights_only=True, period=5)
    checkpoint = ModelCheckpoint(Model_Filepath,save_weights_only=True, monitor='auc', verbose=2, save_best_only=True, mode ='max')
                                 
    print("training start...")
    training=model.fit(x_train,y_train,
                    epochs=epochs_num,batch_size=BATCH_SIZE,callbacks=[auc_ap_eval, earlyStopping, checkpoint],
                    class_weight = class_weights,verbose=2,validation_data=[x_val,y_val,val_sample_weights])

    
    print('training end...')
    
    return training


basicmodel = basicModel(embedding_matrix,MAX_NB_WORDS,MAX_PARA_LENGTH,MAX_PARAS)
training = trainModel(X_train, Y_train, 'han_smr', basicmodel,epochs_num, X_val, Y_val, val_sample_weights, class_weights)


df=pd.DataFrame.from_dict(training.history)
df.columns=[ "Avg_Prec","acc","auc","loss", "val_acc", "val_loss"]
df.index.name='epoch'
print(df)

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(8,3))
df[["acc", "val_acc"]].plot(ax=axes[0])
df[["loss", "val_loss"]].plot(ax=axes[1])
df[["auc"]].plot(ax=axes[2])
plt.savefig('plot_smr'+date+ '.pdf')


basicmodel.load_weights('han_smr')
pred = basicmodel.predict(X_test)
ap_test = average_precision_score(Y_test, pred)
print("AP: ", ap_test)
auc_test = roc_auc_score(Y_test, pred)
print("AUC: ", auc_test)




