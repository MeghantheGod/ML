import pandas as pd
df = pd.read_csv("/home/meghan/Downloads/datasets/diabetes.csv")
df
from sklearn.model_selection import train_test_split
X= df.drop("Outcome", axis=1)
Y= df['Outcome']
X
Y
X_train,X_test,Y_train,Y_test= train_test_split(X,Y, test_size=0.25, random_state=0)
X_train
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train,Y_train)
y_pred = knn.predict(X_test)
y_pred
acc = accuracy_score(y_pred,Y_test)
acc
err = 1-acc
err
def knn(X_train,Y_train,X_test,Y_test,n):
    n_range = range(1,n)
    results = []
    for n in n_range:
        knn = KNeighborsClassifier(n_neighbors=n)
        knn.fit(X_train,Y_train)
        y_pred = knn.predict(X_test)
        acc = accuracy_score(y_pred,Y_test)
        results.append(acc)
    return results
import matplotlib.pyplot as plt
n=400
output = knn(X_train,Y_train,X_test,Y_test,n)
n_range = range(1,n)
plt.plot(n_range,output)
