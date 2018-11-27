import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from evaluations import Evaluation

df_training = pd.read_csv("training_data.txt")
train_data, train_label = df_training.loc[:,df_training.columns != 'label'], df_training['label']
df_testing = pd.read_csv("testing_data.txt")
test_data, test_label = df_testing.loc[:,df_testing.columns != 'label'], df_testing['label']

knn = KNeighborsClassifier(n_neighbors=6).fit(train_data, train_label)
knn_predictions = knn.predict(test_data)
print(knn_predictions)
