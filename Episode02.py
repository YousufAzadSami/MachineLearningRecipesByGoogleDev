import numpy as np
from sklearn import tree
from sklearn.datasets import load_iris

iris = load_iris()

test_index = [0, 50, 100]

training_target = np.delete(iris.target, test_index)
training_data = np.delete(iris.data, test_index, axis=0)

testing_target = iris.target[test_index]
testing_data = iris.data[test_index]

# print(len(training_data), " : ", len(training_target))

classifier = tree.DecisionTreeClassifier()
classifier.fit(training_data, training_target)

# print(testing_data, " : ", testing_target)

# visualization code
from sklearn.externals.six import StringIO
import pydotplus

dot_data = StringIO()
tree.export_graphviz(
    classifier,
    out_file=dot_data,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True, rounded=True,
    impurity=False)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("classifier_visualization.pdf")
