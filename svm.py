import pandas as pd
from sklearn import svm
from evaluations import Evaluation

df_training = pd.read_csv("training_data.txt")
train_data, train_label = df_training.loc[:, df_training.columns != 'label'], df_training['label']
# eventually we wont need a testing_data file, we can just stream data in from app.py
# but for now I made one just so we can see if the model works
df_testing = pd.read_csv("testing_data.txt")
test_data, test_label = df_testing.loc[:,df_testing.columns != 'label'], df_testing['label']

clf = svm.SVC()
y_pred = clf.fit(train_data, train_label)
predictions = clf.predict(test_data)

evaluationdata = Evaluation(predictions, test_label)
print(predictions)
print(evaluationdata.getF1())