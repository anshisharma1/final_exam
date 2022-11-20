import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets
from Q1_M20AIE217 import test_case_random, test_case_not_random

digits = datasets.load_digits()
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))



# Split data into 50% train and 50% test subsets
# Q1 Ensure exact same splitting of the dataset; correspondingly update your code
# Answer. Random_state is the argument in train_test_split which ensure exact same plit of the dataset.
print("Data set got splitted with random state 42")
X_train1, X_test1, y_train1, y_test1 = train_test_split(
    data, digits.target, test_size=0.5, random_state = 42
)

print("Data set got splitted with random state 42")
X_train2, X_test2, y_train2, y_test2 = train_test_split(
    data, digits.target, test_size=0.5, random_state = 42
)

print("Data set got splitted with random state 101")
X_train3, X_test3, y_train3, y_test3 = train_test_split(
    data, digits.target, test_size=0.5, random_state = 101
)

print("Splited dataset compared with random state 42 and 42")
# Test case for comparison for same random state
test_case_random(X_train1, X_test1, X_train2, X_test2)

# Test case for comparison for different random state
print("Comparing the splited dataset with random state 42 and 101")
test_case_not_random(X_train1, X_test1, X_train3, X_test3)

print("Splited dataset compared with random state 42 and 101 for checking Field case")
# Test case for comparison for same random state
test_case_random(X_train1, X_test1, X_train3, X_test3)



