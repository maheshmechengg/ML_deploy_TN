# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

dataset = pd.read_csv('JL_JT_ VPTDuty_CycleData_vehdataTN.csv')

dataset.columns = ['Configuration', 'Torque', '1ST Gear', 'FDR',
       'LAP','GVW','GCW','Tire','RearAxleTN', 'FrontAxleTN']

dataset['FrontAxleTN'].fillna(0, inplace=True)

#dataset['RearAxleTN'].replace(0,dataset['RearAxleTN'].median(), inplace=True)

dataset['FrontAxleTN'].replace(0,dataset['FrontAxleTN'].mean(), inplace=True)

X = dataset[['Torque', '1ST Gear', 'FDR','LAP','GVW','GCW','Tire']]

#X = dataset.iloc[:, :3]

#Converting words to integer values
#def convert_to_int(word):
#    word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8,
#                'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, 0: 0}
#    return word_dict[word]

#X['experience'] = X['experience'].apply(lambda x : convert_to_int(x))

#y = dataset.iloc[:, -1]
y = dataset["RearAxleTN"]

#Splitting Training and Test Set
#Since we have a very small dataset, we will train our model with all availabe data.

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(X, y)

# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[484, 4.71, 4.1,5105, 6250, 12450, 0.411]]))