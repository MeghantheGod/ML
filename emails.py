import pandas as pd
df = pd.read_csv("/home/meghan/Downloads/datasets/emails.csv")
df
x= df.drop("Email No.", axis=1)
y = df['Prediction']
x
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25, random_state=0)
x_train
y_train
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(x_train,y_train)
y_pred = knn.predict(x_test)
acc = accuracy_score(y_pred,y_test)
acc
err = (1-acc)
err
def knn(x_train,y_train,x_test,y_test,n):
    n_range = range(1,n)
    results = []
    for n in n_range:
        knn = KNeighborsClassifier(n_neighbors=n)
        knn.fit(x_train,y_train)
        y_pred = knn.predict(x_test)
        acc = accuracy_score(y_pred,y_test)
        results.append(acc)
    return results
import matplotlib.pyplot as plt
n=400
n_range = range(1,n)
output = knn(x_train,y_train,x_test,y_test,n)
plt.plot(n_range,output)
from sklearn.svm import SVC, LinearSVC
import math, time
start = time.time()
model = SVC(kernel='poly', C=2)
model.fit(x_train,y_train)
pred = model.predict(x_test)
acc= accuracy_score(y_test,pred)
print(round(acc*100,1), "%")
end = time.time()

print(f"{end-start:.5f}sec")
start = time.time()
model = LinearSVC(C=3)
model.fit(x_train,y_train)
pred = model.predict(x_test)
acc= accuracy_score(y_test,pred)
print(round(acc*100,1), "%")
end = time.time()

print(f"{end-start:.5f}sec")
start = time.time()
model = SVC(kernel='sigmoid', C=2)
model.fit(x_train,y_train)
pred = model.predict(x_test)
acc= accuracy_score(y_test,pred)
print(round(acc*100,1), "%")
end = time.time()

print(f"{end-start:.5f}sec")
start = time.time()
model = SVC(kernel='rbf', C=2)
model.fit(x_train,y_train)
pred = model.predict(x_test)
acc= accuracy_score(y_test,pred)
print(round(acc*100,1), "%")
end = time.time()

print(f"{end-start:.5f}sec")
