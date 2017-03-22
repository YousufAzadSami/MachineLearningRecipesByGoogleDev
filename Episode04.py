# imports to compare preditected result with actual data
from sklearn.metrics import accuracy_score
# importing dataset
from sklearn import datasets
iris_dataset = datasets.load_iris()

# imagine x as the input of a function and y as the output
# e.g. f(x) = y
x = iris_dataset.data
y = iris_dataset.target

'''
from sklearn.cross_validation Using this throws a warning,
cause cross_validation is deprecated and is replaces with model_selection
'''
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.5)

# using DecisionTree classifier
from sklearn import  tree
decision_tree_classifier = tree.DecisionTreeClassifier()
decision_tree_classifier.fit(x_train, y_train)
decision_tree_predictions = decision_tree_classifier.predict(x_test)
print("Using Decision Tree classifer the accuracy is : ", accuracy_score(y_test, decision_tree_predictions))

# using KNeighbors Classifier
from sklearn.neighbors import  KNeighborsClassifier
kn_classifier = KNeighborsClassifier()
kn_classifier.fit(x_train, y_train)
kn_predictions = kn_classifier.predict(x_test)
print("Using KNeighbour classifier the accuracy is : ", accuracy_score(y_test, kn_predictions))