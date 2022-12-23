import pandas as pd
import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report

# Load the dataset
df = pd.read_csv(r"C:\Users\USER\OneDrive - Ming Chuan University\uni\ML_fall\hw files\second\winequality-white.csv", delimiter=';')

# Check for null values
print(df.info())
print("Total nulls: ", df.isna().sum().sum())

# Encode the target variable
encoder = LabelEncoder()
df['quality'] = encoder.fit_transform(df['quality'])

# Visualize the correlations between features
corr = df.corr()
sns.heatmap(corr, annot=True, vmin=-1.0, vmax = 1.0)
plt.show()

# Split the data into features and target
y = df['quality']
X = df.drop('quality', axis =1)

# Standardize the features
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8, random_state=34)

# Define the model
num_feat = X.shape[1]
num_class = len(y.unique())

inputs = tf.keras.Input(shape=(num_feat))
x = tf.keras.layers.Dense(64, activation='relu')(inputs)
x = tf.keras.layers.Dense(num_feat, activation='relu')(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs = inputs, outputs = outputs)

# Compile the model
model.compile(
    optimizer = 'adam', 
    loss = 'binary_crossentropy', 
    metrics = ['accuracy'])

# Train the model
batch_size = 32
epochs = 100
history = model.fit(
    X_train,
    y_train, 
    validation_data=(X_val, y_val), 
    batch_size = batch_size, 
    epochs = epochs
)

# Evaluate the model on the validation set
val_loss, val_acc = model.evaluate(X_val, y_val)
print(f'Validation Loss: {val_loss:.4f} Validation Acc: {val_acc:.4f}')

# Generate a classification report
y_pred = model.predict(X_val).round().astype(int)
print(classification_report(y_val, y_pred))

# Use k-fold cross-validation to evaluate the model
# Set the number of folds
n_folds = 5

# Define the KFold object
kfold = KFold(n_splits=n_folds, shuffle=True, random_state=34)

# Initialize a list to store the evaluation metrics for each fold
eval_metrics = []

# Loop over the folds
for train_index, val_index in kfold.split(X):
    # Split the data into training and validation sets
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    # Train the model on the training set
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    # Evaluate the model on the validation set
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    
    # Store the evaluation metrics
    eval_metrics.append((val_loss, val_acc))

# Calculate the mean and standard deviation of the evaluation metrics
mean_loss = np.mean([x[0] for x in eval_metrics])
mean_acc = np.mean([x[1] for x in eval_metrics])
std_loss = np.std([x[0] for x in eval_metrics])
std_acc = np.std([x[1] for x in eval_metrics])

print(f'Mean Validation Loss: {mean_loss:.4f} (+/- {std_loss:.4f})')
print(f'Mean Validation Acc: {mean_acc:.4f} (+/- {std_acc:.4f})')


# Define the models to be evaluated
models = [
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    SVC(),
    MLPClassifier()
]

# Initialize a list to store the evaluation metrics for each model
model_metrics = []

# Loop over the models
for model in models:
    # Initialize a list to store the evaluation metrics for each fold
    eval_metrics = []
    
    # Loop over the folds
    for train_index, val_index in kfold.split(X):
        # Split the data into training and validation sets
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        # Train the model on the training set
        model.fit(X_train, y_train)

        # Evaluate the model on the validation set
        y_pred = model.predict(X_val)
        val_acc = accuracy_score(y_val, y_pred)
        
        # Store the evaluation metric
        eval_metrics.append(val_acc)
    
    # Calculate the mean and standard deviation of the evaluation metrics
    mean_acc = np.mean(eval_metrics)
    std_acc = np.std(eval_metrics)
    
    # Store the evaluation metrics for the model
    model_metrics.append((model, mean_acc, std_acc))

# Sort the models by mean accuracy
model_metrics.sort(key=lambda x: x[1], reverse=True)

# Print the evaluation metrics for each model
for model, mean_acc, std_acc in model_metrics:
    print(f'Model: {model.__class__.__name__}')
    print(f'Mean Validation Acc: {mean_acc:.4f} (+/- {std_acc:.4f})')
    print("          ")
val_loss, val_acc = model.evaluate(X_val, y_val)
