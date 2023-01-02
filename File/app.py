#Student Name: Anisjon Berdiev
#Student ID: 09890314
#The app is deployed on my github: https://aianis-wine-quality-classification-webapp-fileapp-mj2agb.streamlit.app/
# You can also git clone directlty from here: https://github.com/aianis/Wine-Quality-Classification-WebApp.git

import pandas as pd 
import tensorflow as tf
import streamlit as st
from PIL import Image
import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier
from collections import Counter

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
import sys
print(sys.path)



df = pd.read_csv(r"C:\Users\USER\OneDrive - Ming Chuan University\uni\ML_fall\hw files\second\File\winequality-white.csv", delimiter=';')
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
epochs = 10

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
#Streamlit Web App 
########################################################################
st.title('Wine Quality Prediction  Web App')

image = Image.open(r"C:\Users\USER\OneDrive - Ming Chuan University\uni\ML_fall\hw files\second\File\wine_image.png")

# Display the image in the app
st.image(image, caption='Wine image by Unsplash', use_column_width=True)


# Display a table of the data
st.subheader('Data Visualization')
st.dataframe(df)

st.set_option('deprecation.showPyplotGlobalUse', False)
st.markdown('<h3>Heatmaps</h3>', unsafe_allow_html=True)


corr = df.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, vmin=-1.0, vmax = 1.0)
plt.show()
st.pyplot()

#line chart
st.markdown('<h3>Line Chart</h3>', unsafe_allow_html=True)
st.line_chart(df, x='quality', y='alcohol')
#area chart
st.markdown('<h3>Area Chart</h3>', unsafe_allow_html=True)
st.area_chart(df, x='quality', y='alcohol')

#bar chart
st.markdown('<h3>Bar Chart</h3>', unsafe_allow_html=True)
st.bar_chart(df, x='quality', y='alcohol')

# Create a pie chart using Matplotlib
st.markdown('<h3>Pie Chart</h3>', unsafe_allow_html=True)
st.markdown('<h6>[A darker color in the chart indicates a stronger relationship between the attribute and the wine quality]</h6>', unsafe_allow_html=True)
plt.pie(df['alcohol'], labels=df['quality'])
# Display the pie chart in Streamlit
st.pyplot()

st.sidebar.title("Input Values")
fixed_acidity = st.sidebar.slider("Fixed acidity:", 0.0, 1.0, 0.5)
volatile_acidity = st.sidebar.slider("Volatile acidity:", 0.0, 1.0, 0.5)
citric_acid = st.sidebar.slider("Citric acid:", 0.0, 1.0, 0.5)
residual_sugar = st.sidebar.slider("Residual sugar:", 0.0, 1.0, 0.5)
chlorides = st.sidebar.slider("Chlorides:", 0.0, 1.0, 0.5)
free_sulfur_dioxide = st.sidebar.slider("Free sulfur dioxide:", 0.0, 1.0, 0.5)
total_sulfur_dioxide = st.sidebar.slider("Total sulfur dioxide:", 0.0, 1.0, 0.5)
density = st.sidebar.slider("Density:", 0.0, 1.0, 0.5)
pH = st.sidebar.slider("pH:", 0.0, 1.0, 0.5)
sulphates = st.sidebar.slider("Sulphates:", 0.0, 1.0, 0.5)
alcohol = st.sidebar.slider("Alcohol:", 0.0, 1.0, 0.5)

################################################################
# Visualization by getting user inputs from the slider 
################################################################ 

# Define the train_model function
def train_model():
    # Define the model architecture
    inputs = tf.keras.Input(shape=(num_feat))
    x = tf.keras.layers.Dense(64, activation='relu')(inputs)
    x = tf.keras.layers.Dense(num_feat, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs = inputs, outputs = outputs)

    # Compile the model
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    # Train the model
    batch_size = 32
    epochs = 10
    history = model.fit(X_train, y_train, validation_split=0.2, batch_size = batch_size, epochs = epochs)

    # Evaluate the model
    model.evaluate(X_test, y_test)

    # Return the trained model
    return model

# Train the model
model = train_model()

# Create a data frame with the values from the sliders
input_data = pd.DataFrame({
    "fixed_acidity": [fixed_acidity],
    "volatile_acidity": [volatile_acidity],
    "citric_acid": [citric_acid],
    "residual_sugar": [residual_sugar],
    "chlorides": [chlorides],
    "free_sulfur_dioxide": [free_sulfur_dioxide],
    "total_sulfur_dioxide": [total_sulfur_dioxide],
    "density": [density],
    "pH": [pH],
    "sulphates": [sulphates],
    "alcohol": [alcohol]
})

# Define the map_prediction_to_label function
def map_prediction_to_label(prediction):
    """Map a prediction to a quality label.

    Args:
        prediction (int): The prediction to map to a label.

    Returns:
        str: The label corresponding to the prediction.
    """
     # Determine the quality label based on the values of the sliders
    if prediction == 1 and fixed_acidity < 7 and volatile_acidity < 0.5 and citric_acid > 0.2 and residual_sugar < 1 and chlorides < 0.06 and free_sulfur_dioxide > 20 and total_sulfur_dioxide > 50 and density > 0.99 and pH > 3 and sulphates > 0.5 and alcohol > 9:
        return "High"
    elif prediction == 1 and fixed_acidity < 7 and volatile_acidity < 0.5 and citric_acid > 0.2 and residual_sugar < 1 and chlorides < 0.06 and free_sulfur_dioxide > 20 and total_sulfur_dioxide > 50 and density > 0.99 and pH > 3 and sulphates > 0.5 and alcohol > 6:
        return "Average"
    else:
        return "Low"

# Use the model to make a prediction on the input data
prediction = model.predict(input_data)

# Round the predicted values to the nearest integer
prediction = np.round(prediction)

# Map the prediction to a quality label
quality_label = map_prediction_to_label(prediction)

# Display the predicted quality label on the Streamlit app
st.write("Wine quality:", quality_label)

################################################################
#Displaying the result as barchart 
################################################################
# Use the model to make a prediction on the input data
prediction = model.predict(input_data)

# Round the predicted values to the nearest integer
prediction = np.round(prediction)

# Map the prediction to a quality label
quality_labels = []
for p in prediction:
    quality_labels.append(map_prediction_to_label(p))

# Count the number of predictions for each quality label
counts = dict(Counter(quality_labels))

# Create a dataframe from the counts
counts_df = pd.DataFrame({'Quality': list(counts.keys()), 'Count': list(counts.values())})

# Display the counts as a bar chart
st.bar_chart(counts_df)

