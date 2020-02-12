import pandas as pd
import os
from pathlib import Path
#data_path=Path('D:/research')
#print(data_path)
#filepath=D:\IntrusionDetection\research\KDD_Train.csv
#print(filepath)
df=pd.read_csv("D:/College/Hackru/one_hot_encoded.csv")

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
print(column_names)
print(len(column_names))

all_features=df_scaled[column_names].values
all_classes=df_scaled['price_per_100g_ml_dollars'].values

X = df.drop('price_per_100g_ml_dollars',1)
#print(X.head())
#print('-----')
y = df['price_per_100g_ml_dollars']
#print(y.head())
#X.shape
#y.shape


df=df.drop(columns='price_per_100g_ml_dollars')
df.head(12)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.5)
X_train,X_train_lr, y_train, y_train_lr = train_test_split(X_train,y_train,test_size=0.5)

print(X.shape)
import numpy as np

X["fill"]=np.zeros(len(X))
X["fillll"]=np.zeros(len(X))
X["filll"]=np.zeros(len(X))
print(X.shape)

temp=X.values
training_temp_data=np.reshape(temp,(len(X),80,65,1))

from tensorflow.python.keras.layers import Dense,Dropout,Conv2D,MaxPooling2D,Flatten
from tensorflow.python.keras.models import Sequential
from sklearn.model_selection import cross_val_score
#import keras_metrics
print("Import Worked")

def create_model():
    model=Sequential()
    model.add(Conv2D(400,(3,3),activation="relu",input_shape=(80,65,1)))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(400,(3,3),activation="relu"))
    model.add(Flatten())
    model.add(Dense(50,input_dim=len(df_scaled.columns)-1, kernel_initializer='normal', activation='tanh'))
    print('1st layer')
    #Hidden layer
    model.add(Dense(50,activation='linear'))
    #OUTPUT LAYER
    print("Hidden layer")
    model.add(Dense(1))
    
    #COMPILE MODE
    print("Output layer")
    model.compile(loss='mse', optimizer='nadam',metrics=["mae","acc"])
    return model
 
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier
print("About to start estimator")   

keras_model = create_model()
history = keras_model.fit(training_temp_data[:3685],y_train,epochs=10000, batch_size=50,verbose=1)
