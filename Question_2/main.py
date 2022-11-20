import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets, svm, metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from Q2_M20AIE217 import test_case_random, test_case_not_random
import argparse

digits = datasets.load_digits()
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

parser = argparse.ArgumentParser(
                    prog = 'test',
                    description = 'train a classifier',
                    epilog = 'dummy')

# parser.add_argument('filename')          
parser.add_argument('-c', '--clf_name')      
parser.add_argument('-v', '--random_state')  

args = parser.parse_args()
clf = args.clf_name
random_state = int(args.random_state)



# Spliting data into 50% train and 50% test subsets
# Q1 Ensure exact same splitting of the dataset; correspondingly update your code
# Answer.  The argument Random_state in train_test_split ensure exact same plit of the dataset.
print("Data set splitted with random state ")
X_train1, X_test1, y_train1, y_test1 = train_test_split(
    data, digits.target, test_size=0.5, random_state = random_state
)

if clf == "svm":
    model = svm.SVC()
else:
    model = DecisionTreeClassifier()

model.fit(X_train1,y_train1)

pred = model.predict(X_test1)

print(f"Accuracy is : {metrics.accuracy_score(y_test1,pred)}")
print(f"classification report is: {metrics.classification_report(y_test1,pred)}")