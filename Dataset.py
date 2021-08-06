from tensorflow.keras.utils import Sequence
import pandas as pd
import numpy as np
import tsaug
import random
from tsaug.visualization import plot

class StandardScaler():
    def __init__(self, MinMax = False):
        self.mean = 0.
        self.std = 1.
        self.min = 0.
        self.max = 0.
        self.MinMax = MinMax
    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)
        self.min = data.min(0)
        self.max = data.max(0)

    def transform(self, data):
        std = self.std
        mean = self.mean
        minn = self.min
        maxx = self.max        
        #return (data - mean) / std
        return (data - minn) / (maxx-minn) if self.MinMax else (data - mean) / std

    def inverse_transform(self, data):
        std = self.std
        mean = self.mean
        minn = self.min
        maxx = self.max
        return (data * (maxx-minn)) + minn if self.MinMax else (data * std) + mean

class Dataset(Sequence):
      
  def __init__(self,path, target ,to_fit=True,batch_size=20, is_val=False,DaysIn=24*7, csv= True, is_test=False,DaysOut=1):
    self.to_fit = to_fit
    self.is_val = is_val
    self.target = target
    self.scaler = StandardScaler()
    self.DaysIn = DaysIn
    self.batch_size = batch_size
    self.is_test = is_test
    self.DaysOut= DaysOut
    if csv:
        df_raw = pd.read_csv(path)
    else:
        df_raw = pd.read_excel(path)

    
    df_data = df_raw[[self.target]]
    data = df_data.values

    if self.to_fit:
      
        num_val = int(len(df_data)*0.4)
        num_train = int(len(df_data) - num_val)

        self.X_data_train=data[:num_train-self.DaysOut]
        self.X_data_val=data[num_train-self.DaysIn:-self.DaysOut]

        self.Y_data_train=data[self.DaysIn:num_train]
        self.Y_data_val=data[num_train:]
    else: 
        
        self.X_data_train = data[:-1*self.DaysIn]
        self.Y_data_train=data[self.DaysIn:]

    self.idxList = [i for i in range(self.__len__()*self.batch_size)]
    self.scaler.fit(self.X_data_train)
    if self.to_fit:
        self.X_data_train = self.scaler.transform(self.X_data_train)
        self.X_data_val = self.scaler.transform(self.X_data_val)
        self.Y_data_train = self.scaler.transform(self.Y_data_train)
        self.Y_data_val = self.scaler.transform(self.Y_data_val)
    else:
        self.X_data_train = self.scaler.transform(self.X_data_train)

    if self.is_test:
        self.Y_data_train = self.scaler.transform(self.Y_data_train)

  def __getitem__(self, index):
    start=index*self.batch_size
    ending= index*self.batch_size+self.batch_size
    
    tempList=self.idxList[start:ending]

    if not self.is_val:
        X = [self.X_data_train[i:i+self.DaysIn] for i in tempList]
        if  self.to_fit or self.is_test:
            Y = [self.Y_data_train[i:i+self.DaysOut] for i in tempList]

         
    else:
        X = [self.X_data_val[i:i+self.DaysIn] for i in tempList]
        Y = [self.Y_data_val[i:i+self.DaysOut] for i in tempList]


    if self.to_fit and not self.is_val:
        return np.array(X), np.array(Y)
    elif self.is_val:
        return np.array(X), np.array(Y)
    else:
        return np.array(X)
    

  def on_epoch_end(self):
    
    np.random.seed(20)
    if self.is_val == False:
        np.random.shuffle(self.idxList)
        print("shuffle done!")


  def __len__(self):
    if self.is_val:
        return int (( len(self.X_data_val) - self.DaysIn +1 )/ self.batch_size)

    else:
        return int (( len(self.X_data_train) - self.DaysIn +1 )/ self.batch_size)

  def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

