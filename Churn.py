import pandas as pd
df = pd.read_csv("/home/meghan/Downloads/datasets/Churn_Modelling.csv")
df
df.isnull().sum()
df.dropna(inplace=True)
df.isnull().sum()
df.describe
Gender_data = pd.get_dummies(df["Gender"], drop_first=True)
Gender_data
Geo_data = pd.get_dummies(df["Geography"], drop_first=True)
Geo_data
df = df.drop(columns=["RowNumber","CustomerId","Surname","Gender","Geography"], axis=1)
df
df = pd.concat([df, Geo_data, Gender_data], axis=1)
df
x= df.drop(["Exited"], axis=1)
y= df["Exited"]
x
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=0)
x_train
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train= scaler.fit_transform(x_train) 
x_test= scaler.fit_transform(x_test)
x_train
x_test
import keras
from keras.models import Sequential
from keras.layers import Dense
classifier = Sequential()
classifier.add(Dense(activation="relu", input_dim=11, units=6,kernel_initializer="uniform"))
classifier.add(Dense(activation="relu", units=6,kernel_initializer="uniform"))
classifier.add(Dense(activation="sigmoid", units=1,kernel_initializer="uniform"))
classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])
classifier.summary()
classifier.fit(x_train,y_train, batch_size=10, epochs=50)
F_pred = classifier.predict(x_test)
F_pred
y_pred = (F_pred>0.5)
y_pred
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_pred, y_test)
acc
from sklearn.metrics import confusion_matrix
conf = confusion_matrix(y_pred, y_test)
conf
