import tensorflow as tf
import os
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.python.keras.layers.wrappers import Bidirectional
from tensorflow.keras import layers
#import h5py

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 


class LSTM:
    
  def __init__(self,DaysIn=24*7, DaysOut=1):
        # Input layerB

      input_ = layers.Input(shape=(DaysIn,1)) #, batch_size= 1
      x= layers.LSTM(50, return_sequences = True, stateful = False, dropout=0.01)(input_)
      #x= layers.BatchNormalization(axis=-1)(x)
      x= layers.LSTM(50, return_sequences = True, stateful = False, dropout=0.01)(x)
      x= layers.LSTM(50, return_sequences = True, stateful = False, dropout=0.01)(x)
      #x= layers.BatchNormalization(axis=-1)(x)
      x= layers.LSTM(50, return_sequences = False, stateful = False, dropout=0.01)(x)
      #x= layers.BatchNormalization(axis=-1)(x)
      #xFC = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(DaysOut, activation='elu'),name="FC")(x)
      xFC = layers.Dense(DaysOut, activation='linear',name="FC")(x)

      self.model= Model(inputs= input_, outputs=xFC)
    
      
  def training(self,trainGen,valGen,Nepoch=15):
    
    self.model.summary()
    loss_fn = tf.keras.losses.MeanSquaredError()
    opt = tf.keras.optimizers.RMSprop(learning_rate=0.01) #0.005
    callBack= tf.keras.callbacks.EarlyStopping(
                                      monitor="val_loss",
                                      min_delta=0.001,
                                      patience=50,
                                      verbose=0,
                                      mode="min",
                                      baseline=None,
                                      restore_best_weights=True,
                                  )
    rlrop = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', mode='min', patience= 3, factor= 0.01, min_lr= 1e-9, verbose=1,min_delta=0.001)
    self.model.compile(optimizer=opt, loss={"FC":loss_fn}, metrics={"FC":loss_fn})
    #self.model.fit_generator(trainGen,validation_data =valGen,epochs=15) 
    self.pre_training_history = self.model.fit(trainGen,validation_data =valGen,epochs=Nepoch, callbacks=[callBack, rlrop])
      
      
  def save_model(self,path,modelname="Model"):
    self.model.save(os.path.join(path,modelname)+".h5")

  def save_weights(self,path,modelname="Model_weights"):

    self.model.save_weights(os.path.join(path,modelname)+".h5")

  def load_weights(self,path,checkpoint="Model_weights"):
    self.model.load_weights(os.path.join(path,checkpoint)+".h5")


class GRU:
    
  def __init__(self,DaysIn=24*7, DaysOut=1):
        # Input layerB

      input_ = layers.Input(shape=(DaysIn,1)) #, batch_size= 1
      x= layers.GRU(50, return_sequences = True, stateful = False, dropout=0.2)(input_)
      #x= layers.BatchNormalization(axis=-1)(x)
      x= layers.GRU(50, return_sequences = True, stateful = False, dropout=0.2)(x)
      
      #x= layers.BatchNormalization(axis=-1)(x)
      x= layers.GRU(50, return_sequences = False, stateful = False, dropout=0.2)(x)
      #x= layers.BatchNormalization(axis=-1)(x)
      #xFC = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(DaysOut, activation='elu'),name="FC")(x)
      xFC = layers.Dense(DaysOut, activation='linear',name="FC")(x)

      self.model= Model(inputs= input_, outputs=xFC)
    
      
  def training(self,trainGen,valGen,Nepoch=15):
    
    self.model.summary()
    loss_fn = tf.keras.losses.MeanSquaredError()
    opt = tf.keras.optimizers.Adam(learning_rate=0.01) #0.005
    callBack= tf.keras.callbacks.EarlyStopping(
                                      monitor="val_loss",
                                      min_delta=0.001,
                                      patience=10,
                                      verbose=0,
                                      mode="min",
                                      baseline=None,
                                      restore_best_weights=True,
                                  )
    rlrop = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', mode='min', patience= 3, factor= 0.01, min_lr= 1e-9, verbose=1,min_delta=0.001)
    self.model.compile(optimizer=opt, loss={"FC":loss_fn}, metrics={"FC":loss_fn})
    #self.model.fit_generator(trainGen,validation_data =valGen,epochs=15) 
    self.pre_training_history = self.model.fit(trainGen,validation_data =valGen,epochs=Nepoch, callbacks=[callBack, rlrop])
      
      
  def save_model(self,path,modelname="Model"):
    self.model.save(os.path.join(path,modelname)+".h5")

  def save_weights(self,path,modelname="Model_weights"):

    self.model.save_weights(os.path.join(path,modelname)+".h5")

  def load_weights(self,path,checkpoint="Model_weights"):
    self.model.load_weights(os.path.join(path,checkpoint)+".h5")