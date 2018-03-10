from __future__ import print_function
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target

# print(iris_X[:2, :])
# print(iris_y)

X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=0.3)

print("X_train:", X_train)
print("y_train:", y_train)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
print("conclusion:",knn.predict(X_test))
print("conclusion:", y_test)

import pickle
knn2 = pickle.dumps(knn)
knn2 = pickle.loads(knn2)
print("conclusion2:", knn2.predict(X_test))