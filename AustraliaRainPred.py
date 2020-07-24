#importing basic libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#importing tensorflow libraries for building the model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.callbacks import EarlyStopping

#splitting and scaling data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

#model start
model = Sequential()      #Sequential model is used

model.add(Dense(23,activation = 'relu'))     #First Layer with 23 Neurons,Rectified Linear Unit as activation functions
model.add(Dropout(0.5))                      #Dropout to avoid over-fitting

model.add(Dense(1,activation = 'sigmoid'))   #Binary Classification, hence last layer is sigmoid

model.compile(optimizer = 'adam',
             loss = 'binary_crossentropy',
             metrics = ['accuracy'])          #accuracy metric inserted for keeping track of train and validation data

stop = EarlyStopping(monitor = 'val_loss',
                    mode = 'min',
                    patience = 2)            #early stopping to avoid over-fiiting, model fluctuates a bit, hence patience has been set to 2

#model ends

#API Command : kaggle datasets download -d jsphyg/weather-dataset-rattle-package
data = pd.read_csv('weatherAUS.csv')        #cleansed data after exploritary data analysis 

X = data.drop('RainTomorrow',axis = 1).values   #X with dropping the target column
Y = data['RainTomorrow'].values                 #Y as the target column

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,random_state = 1,test_size = 0.2)    #splitting the data into train and test

#scaling with Min Max Scaler to bring data between 0 and 1
scale = MinMaxScaler()
X_train = scale.fit_transform(X_train)
X_test = scale.transform(X_test)

#fiiting the model with validation data as X_test and X_train. Epochs at around 25-35 and callback as the early stopping object
model.fit(X_train,Y_train,
         validation_data = (X_test,Y_test),
         epochs = 30,
         callbacks = [stop])

#predicting classes
y_pred_train = model.predict_classes(X_train)
y_pred_test = model.predict_classes(X_test)

#finding accuracy for each split
accuracy_train = accuracy_score(y_pred_train,Y_train)
accuracy_test = accuracy_score(y_pred_test,Y_test)

print(accuracy_train,accuracy_test)
#Output : (0.9958348103509393, 0.9948599787309464)
