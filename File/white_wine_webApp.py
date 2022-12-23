"""
Name: Anisjon Berdiev 
Student ID: 09893014
"""

import pandas as pd 
import tensorflow as tf
import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler


df = pd.read_csv(r"C:\Users\USER\OneDrive - Ming Chuan University\uni\ML_fall\hw files\second\winequality-white.csv", delimiter=';')
print(df.head(5))

#checking if there NULLs if there we need to clean it
print(df.info())
print("Total nulls: ", df.isna().sum().sum())
df['quality'].unique()

encoder = LabelEncoder()
df['quality'] = encoder.fit_transform(df['quality'])
{index: label for index, label in enumerate(encoder.classes_)}


################################################################
#VISUALZATION 
################################################################
corr = df.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, vmin=-1.0, vmax = 1.0)
plt.show()



################################################################
#Model training: spliting and training the model 
#Quantile splitting 
################################################################

pd.qcut(df['quality'], q=2, labels = [0, 1])

y = pd.qcut(df['quality'], q=2, labels = [0, 1])
X = df.drop('quality', axis =1)



scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=34)




num_feat = X.shape[1]
print(num_feat)

num_class = len(y.unique())
print(num_class)


inputs = tf.keras.Input(shape=(num_feat))
x = tf.keras.layers.Dense(64, activation='relu')(inputs)
x = tf.keras.layers.Dense(num_feat, activation='relu')(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs = inputs, outputs = outputs)

model.compile(
    optimizer = 'adam', 
    loss = 'binary_crossentropy', 
    metrics = ['accuracy'])

batch_size = 32
epochs = 100

history = model.fit(
    X_train,
    y_train, 
    validation_split=0.2, 
    batch_size = batch_size, 
    epochs = epochs
)

model.evaluate(X_test, y_test)

# Get the predicted labels
y_pred = model.predict(X_test)

# Round the predicted values to the nearest integer
y_pred = np.round(y_pred)

# Get the classification report
report = classification_report(y_test, y_pred)
print(report)

########################################################################
