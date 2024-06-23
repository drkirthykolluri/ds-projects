from sklearn import datasets

#load dataset
cancer = datasets.load_breast_cancer()
print("Features:", cancer.feature_names)
print("Labels:", cancer.target_names)
print(cancer.data.shape)
print(cancer.data[0:5])
print(cancer.target)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.3, random_state=109)

from sklearn import svm
clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

from sklearn import metrics
print("Accuracy with rbf:", metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred))
print("Recall:", metrics.recall_score(y_test, y_pred))
