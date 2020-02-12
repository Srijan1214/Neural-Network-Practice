import pandas as pd
import os
from pathlib import Path
df=pd.read_csv("D:/College/Hackru/one_hot_encoded.csv")

print(df.shape)
df.head(5)

df.dropna(inplace=True)

#Finding columns that have string values.

for col_name in df.columns:
    #print(col_name)
    #print(df[col_name])
    if df[col_name].dtypes=='object':
        a=df[col_name].unique()
        a=len(a)
        print(col_name + " has "+str(a)+" unique values.")


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df_scaled=pd.DataFrame(scaler.fit_transform(df),columns=df.columns)
df_scaled.head(12)

column_names=list(df) 
print(len(column_names))

all_features=df_scaled[column_names].values
all_classes=df_scaled['price_per_100g_ml_dollars'].values


df=df.sort_values(by=['price_per_100g_ml_dollars'])
X = df.drop('price_per_100g_ml_dollars',1)
y = df['price_per_100g_ml_dollars']



df=df.drop(columns='price_per_100g_ml_dollars')
df.head(12)

print(len(df_scaled.columns)-1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
#X_train,X_train_lr, y_train, y_train_lr = train_test_split(X_train,y_train,test_size=0.5)

from tensorflow.python.keras.layers import Dense,Dropout,Conv1D,MaxPooling1D
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras import regularizers
from sklearn.model_selection import cross_val_score
#import keras_metrics
print("Import Worked")
def create_model():
    model=Sequential()
    model.add(Conv1D(kernel_size=80,filters=70,padding='valid', activation='tanh',strides=1))
    model.add(MaxPooling1D(pool_size=5110))
    model.add(Dense(50, kernel_initializer='normal', activation='tanh', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(50, kernel_initializer='normal', activation='tanh', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(50,activation='linear'))
    model.add(Dense(1))
    
    #COMPILE MODE
    print("Output layer")
    model.compile(loss='mse', optimizer='nadam',metrics=["mae"])
    return model
 
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier
print("About to start estimator")   

keras_model = create_model()

import numpy as np
history = keras_model.fit((	X_train.to_numpy()).reshape(len(X_train),5197,1),y_train.to_numpy(),epochs=200, batch_size=50,verbose=1)