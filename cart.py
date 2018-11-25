import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from evaluations import Evaluation

df_training = pd.read_csv("training_data.txt")
train_data, train_label = df_training.loc[:,df_training.columns != 'label'], df_training['label']
df_testing = pd.read_csv("testing_data.txt")
test_data, test_label = df_testing.loc[:,df_testing.columns != 'label'], df_testing['label']

clf = DecisionTreeClassifier()
y_pred = clf.fit(train_data, train_label)
predictions = clf.predict(test_data)

print(predictions)
