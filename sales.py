import pandas as pd
df = pd.read_csv('./sales_data_sample.csv',encoding='latin1')
df.head()
df.describe().T
df.isnull().sum()
df.drop(columns = ['ADDRESSLINE1','ADDRESSLINE2','CITY','STATE','POSTALCODE','PHONE','TERRITORY','CONTACTLASTNAME','CONTACTFIRSTNAME','ORDERNUMBER','STATUS','STATE','ORDERDATE'],axis =1, inplace=True)
df.head()
df.shape
df.drop(columns = ['CUSTOMERNAME'],axis =1, inplace=True)
df.shape
df.dtypes
dealSize = pd.get_dummies(df['DEALSIZE'])
dealSize
COUNTRY = pd.get_dummies(df['COUNTRY'])
COUNTRY
productLine = pd.get_dummies(df['PRODUCTLINE'])
productLine
df.shape
df = pd.concat([df,dealSize,COUNTRY,productLine],axis=1)
df.drop(columns =['PRODUCTLINE','COUNTRY','DEALSIZE'],axis =1,inplace =True)
df.shape
df['PRODUCTCODE'] = pd.Categorical(df['PRODUCTCODE']).codes
df.describe().T
df.head()
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
distortion = []
k =range(1,10)
for n in k:
    km = KMeans(n_clusters = n)
    km.fit(df)
    distortion.append(km.inertia_)


plt.plot(k,distortion,'-bx')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('Elbow curve')
plt.show()

x_train = df.values
model = KMeans(n_clusters = 4,random_state=2)
model.fit(x_train)
pred = model.predict(x_train)
print(pred)
import numpy as np
unique,count = np.unique(pred,return_counts = True)
print(unique)
print(count)
pred_df = pd.DataFrame(pred)
df = pd.concat([df,pred_df],axis=1)
df.head()
df.shape
print(model.cluster_centers_)
df2 = df.drop(columns = [0],axis =1)
cc = pd.DataFrame(data = model.cluster_centers_, columns = [df2.columns])
cc
