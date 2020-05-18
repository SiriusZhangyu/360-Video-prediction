from __future__ import print_function
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Add, Softmax
from tensorflow.keras.layers import Lambda,Concatenate,Flatten,ConvLSTM2D
from tensorflow.keras.layers import Permute,Conv2D
 
from tensorflow.keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
import sys,glob,io,random

    
from random import shuffle
# import matplotlib.pyplot as plt
# import cPickle as pickle
import numpy as np
import pdb
from tensorflow.keras.layers import Conv1D
from tensorflow.keras import optimizers

import matplotlib.pyplot as plt 
import pickle as pickle 
import numpy as np 
import pdb 

## utility layers

get_dim_layer = Lambda(lambda x: x[:,0,0,:,:])
flatten_layer = Flatten()

Concatenatelayer = Concatenate(axis=2)
Concatenatelayer1 = Concatenate(axis=-1)
from tensorflow.keras.layers import BatchNormalization



all_outputs= []
kernel_size=5 
batch_size =1 
epochs2 = 100 
latent_dim = 16
fps=30 
num_encoder_tokens = 10 
num_decoder_tokens = 10
max_decoder_seq_length=10

encoder_inputs=Input(batch_shape=(1,5,18,36,1))
decoder_inputs=Input(batch_shape=(1,1,18,36,1))
decoder_target=Input(batch_shape=(1,5,18,36,1))

expand_dim_layer = Lambda(lambda x: K.expand_dims(x,0))
get_dim_layer = Lambda(lambda x: x[:,0,0,:,:])
get_dim_layer1 = Lambda(lambda x: x[0,:,:,:,:])
flatten_layer = Flatten()
# get_dim1_layer = Lambda(lambda x: x[:,0,:])
Concatenatelayer = Concatenate(axis=2)
Concatenatelayer1 = Concatenate(axis=-1)

encoder_inputs=Input(batch_shape=(1,5,18,36,1))
decoder_inputs=Input(batch_shape=(1,1,18,36,1))
decoder_target=Input(batch_shape=(1,5,18,36,1))

convlstm_encoder = ConvLSTM2D(filters=latent_dim*2, kernel_size=(kernel_size, kernel_size),input_shape=(1,18,36,1), dropout=0.3,recurrent_dropout=0.0, padding='same', return_sequences=True, return_state=True)
encode_outputs0, state_h0, state_c0 = convlstm_encoder(encoder_inputs) 
states0 = [state_h0, state_c0]
convlstm_encoder1 = ConvLSTM2D(filters=latent_dim, kernel_size=(kernel_size, kernel_size),input_shape=(1,18,36,32), dropout=0.3,recurrent_dropout=0.0, padding='same', return_sequences=True, return_state=True)
encode_outputs1, state_h1, state_c1 = convlstm_encoder1(encoder_inputs) 
states1 = [state_h1, state_c1]
convlstm_encoder2 = ConvLSTM2D(filters=latent_dim//2, kernel_size=(kernel_size, kernel_size),input_shape=(1,18,36,16), dropout=0.3,recurrent_dropout=0.0, padding='same', return_sequences=True, return_state=True)
encode_outputs2, state_h2, state_c2 = convlstm_encoder2(encoder_inputs) 
states2 = [state_h2, tate_c2]

convlstm_decoder  = ConvLSTM2D(filters=latent_dim*2, kernel_size=(kernel_size, kernel_size),input_shape=(1,18,36,1), dropout=0.3,recurrent_dropout=0.0, padding='same', return_sequences=True, return_state=True)

convlstm_decoder1 = ConvLSTM2D(filters=latent_dim, kernel_size=(kernel_size, kernel_size),input_shape=(1,18,36,32), dropout=0.3,recurrent_dropout=0.0, padding='same', return_sequences=True, return_state=True)

convlstm_decoder2 = ConvLSTM2D(filters=latent_dim//2, kernel_size=(kernel_size, kernel_size),input_shape=(1,18,36,16), dropout=0.3,recurrent_dropout=0.0, padding='same', return_sequences=True, return_state=True)
bnlayer0 = BatchNormalization(axis=-1,center=True, scale=True) 
bnlayer1 = BatchNormalization(axis=-1,center=True, scale=True) 
bnlayer2 = BatchNormalization(axis=-1,center=True, scale=True)

all_outputs= [] 
inputs = decoder_inputs

Conv= Conv2D(1, kernel_size=(kernel_size,kernel_size), padding='same',data_format='channels_last',
        use_bias=True, kernel_initializer='glorot_uniform')
get_dim_layer1 = Lambda(lambda x: x[0,:,:,:,:])
expand_dim_layer = Lambda(lambda x: K.expand_dims(x,0))

for t in range(5):
    inputs = decoder_inputs
    outputs0, state_h0, state_c0 = convlstm_decoder([inputs]+states0)
    states0 = [state_h0, state_c0]
    outputs1, tate_h1, state_c1 = convlstm_decoder1([outputs0]+states1)
    states1 = [state_h1, tate_c1]
    outputs2, state_h2, state_c2 = convlstm_decoder2([outputs1]+states2)
    states2 = [state_h2, state_c2]
    outputs = Concatenatelayer1([outputs0,outputs1,outputs2])
    out=get_dim_layer1(outputs)
    outputs=Conv(out)
    outputs = expand_dim_layer(outputs)
    all_outputs.append(outputs)
    
 decoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs)
 model = Model([encoder_inputs, decoder_inputs],decoder_outputs) 
 
 model.compile(optimizer='RMSprop', loss='mean_squared_error')
 history=model.fit([encoder_inputs1,decoder_inputs1], decoder_target1, 
          batch_size=1, 
          epochs=200)
          
fig = plt.figure()#新建一张图
 
 
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')


plt.plot(history.history['loss'])

preinput=heatmap[15:20]
preinput=np.array(preinput) 
preinput=preinput.reshape(1,5,18,36,1) 
preinput= tf.convert_to_tensor(preinput, tf.float32, name='t') 

preinput1=heatmap[19]
preinput1=np.array(preinput1) 
preinput1=preinput1.reshape(1,1,18,36,1) 
preinput1= tf.convert_to_tensor(preinput1, tf.float32, name='t') 

k=model.predict([preinput,preinput1])
k=k.reshape(5,18,36)

plt.figure()

plt.subplot(5,2,1)
plt.imshow(k[0])

plt.subplot(5,2,2)
plt.imshow(heatmap[20])

plt.subplot(5,2,3)
plt.imshow(k[1])

plt.subplot(5,2,4)
plt.imshow(heatmap[21])

plt.subplot(5,2,5)
plt.imshow(k[2])

plt.subplot(5,2,6)
plt.imshow(heatmap[22])

plt.subplot(5,2,7)
plt.imshow(k[3])

plt.subplot(5,2,8)
plt.imshow(heatmap[23])

plt.subplot(5,2,9)
plt.imshow(k[4])

plt.subplot(5,2,10)
plt.imshow(heatmap[24])


MSE1=np.sum((k-heatmap[15:20])**2)



