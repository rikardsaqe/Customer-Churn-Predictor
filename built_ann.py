# Building an ANN

# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')

# Input matrix of features
X = dataset.iloc[:, 3:13].values

# Output data
y = dataset.iloc[:, 13].values

# Encoding categorical data for both geography (X_1) and gender (X_2)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])

# Deal with Dummy Variable Trap
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Part 2: Making the ANN

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential 
from keras.layers import Dense
from keras.layers import Dropout

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer 
classifier.add(Dense(output_dim = 6, init = 'uniform' , activation = 'relu', input_dim = 11))
# Applies dropout to reduce overfitting 
classifier.add(Dropout(p = 0.1))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform' , activation = 'relu'))
# Applies dropout to reduce overfitting
classifier.add(Dropout(p = 0.1))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform' , activation = 'sigmoid'))

#Compiling the ANN (adam is an implementation of stochastic gradient descent)
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']) 

# Fitting the ANN to the training set 
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

# Part 3: Making predictions and evaluting the model

# Predicting the test set results
y_pred = classifier.predict(X_test)
# Evalutes whether the prediction is true or false (will the customer leave the bank or not?)
y_pred = (y_pred > 0.5)

# Predicting a single new customer's outcome
# Use a double pair of brackets in np.array to create a singular horizontal vector of the data by putting
# it in the first item of a 2D array. Convereted categorical data into its dummy variable. Used
# sc.transform to scale data to be the same as the training set. 
new_prediction = classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = (new_prediction > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Part 4: Evalutating, Improving, and Tuning the ANN

# Evaluating the ANN (implementing K-Fold Cross Valdiation)

# Need to use function from scikit learn
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential 
from keras.layers import Dense

# Function that builds the neural network architecture we created above
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(output_dim = 6, init = 'uniform' , activation = 'relu', input_dim = 11))
    classifier.add(Dense(output_dim = 6, init = 'uniform' , activation = 'relu'))
    classifier.add(Dense(output_dim = 1, init = 'uniform' , activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']) 
    return classifier 

# Creating an object, classifier, of class KerasClassifier, with the same structure as the previous 
# classifier but different training data given that it will use K-Fold Cross Valdiation below
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, nb_epoch = 100)
    
# Will output the 10 accuracies of the different test sets applied to classifier  
# using K-Fold Cross Valdiation
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)

# Calculate average accuracy of all 10 training sets
mean = accuracies.mean()

# Calculate variance of all 10 training sets
variance = accuracies.std()
    

# Improving the ANN
# Went back and implemented dropout Regularization to reduce overfitting (only if needed)


# Tuning the ANN (attempting to optimize our hyperparameters)

# Need to use function from scikit learn
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential 
from keras.layers import Dense

# Function that builds the neural network architecture we created above
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(output_dim = 6, init = 'uniform' , activation = 'relu', input_dim = 11))
    classifier.add(Dense(output_dim = 6, init = 'uniform' , activation = 'relu'))
    classifier.add(Dense(output_dim = 1, init = 'uniform' , activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy']) 
    return classifier 

# Creating an object, classifier, of class KerasClassifier 
classifier = KerasClassifier(build_fn = build_classifier)

# Dictionary (key-value store: hyperparemters-differing values)of hyperparamters 
# we are attepmting to optimize 
parameters = {'batch_size': [1], 
              'nb_epoch': [100, 500],
              'optimizer': ['adam', 'rmsprop']}

# Create an object, x, of class, GridSearchCV, that contains parameters we want to tune 
# with differing values and also utilizes K-Fold Cross Validation
grid_search = GridSearchCV(estimator = classifier, 
                           param_grid = parameters, 
                           scoring = 'accuracy', 
                           cv = 10) 

# Fits our grid_search object to our training data set
grid_search = grid_search.fit(X_train, y_train)

# Best selction of paremters from grid search (computed using built in function from GridSearchCV)
best_parameters = grid_search.best_params_
 
# Best accuracy given the best set of parameters (computed using built in function from GridSearchCV)
best_accuracy = grid_search.best_score_



