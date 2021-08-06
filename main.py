#%%
from model import GRU, LSTM
from Dataset import Dataset
import matplotlib.pyplot as plt
import numpy as np
import random


daysIn= 30*4           # number of input point
daysOut = 1            # number of points predicted
Use_LSTM = False         #true if use lstm false for GRU

            ## the current script splits the input dataset into a 0.8 for training and 0.2 for validation
TrainDataset = Dataset('Google_Stock_Price_Train.csv', 'Volume',batch_size=48,DaysIn=daysIn,DaysOut=daysOut)
ValDataset = Dataset('Google_Stock_Price_Train.csv', 'Volume', is_val=True, batch_size=48,DaysIn=daysIn,DaysOut=daysOut)

#%%

model = LSTM(DaysIn=daysIn, DaysOut=daysOut)  if Use_LSTM else GRU(DaysIn=daysIn, DaysOut=daysOut)

model.training(TrainDataset, ValDataset, Nepoch=300)

ValDataset = Dataset('Google_Stock_Price_Train.csv', 'Volume', is_val=True, batch_size=1,DaysIn=daysIn,DaysOut=daysOut) #re-initiize to ensure not shuffled dataset

model.model.evaluate(ValDataset)

predictions = model.model.predict(ValDataset)

inverse_predicitons = ValDataset.inverse_transform(predictions.flatten())

inverse_True = np.array([i[1] for i in ValDataset])
inverse_True = ValDataset.inverse_transform(inverse_True.flatten())

st = random.randint(0,len(inverse_predicitons-48))
sl = slice(st,st+48)
plt.plot(inverse_predicitons[sl], label="predictions")
plt.plot(inverse_True[sl], label="True Values")
plt.legend()
plt.show()
print("Done")