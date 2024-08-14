# Demonstrating-PCA
Demonstrating the Computational Efficiency of Principal Component Analysis (PCA) in Classification, Regression, Computer Vision, and Time-Series Analysis Tasks. 

# Classification and Regression

1. Importing Libraries
Data Processing Libraries: pandas, numpy
Encoding and Scaling: StandardScaler, OneHotEncoder, LeaveOneOutEncoder, CatBoostEncoder
Machine Learning Libraries: LogisticRegression, LinearRegression, DecisionTreeClassifier, DecisionTreeRegressor, RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor, SVC, SVR, XGBClassifier, XGBRegressor
Metrics and Evaluation: mean_squared_error, r2_score, mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score
Visualization: matplotlib
Utilities: time, os

2. Data Loading Function (load_data)
This function loads a dataset from either a CSV or JSON file based on the file extension provided. It uses pandas to read the data and returns a DataFrame.

3. Data Preprocessing Function (preprocess_data)
Missing Value Handling: Replaces missing values with the median of each column.

Feature Type Identification:

Numerical Columns: Identifies columns with numerical data types.
Categorical Columns: Identifies columns with object or category data types.
Encoding:

OneHotEncoder: Used if all categorical columns have fewer than 10 unique values.
LeaveOneOutEncoder: Used if all categorical columns have between 10 and 100 unique values.
CatBoostEncoder: Used if any categorical column has more than 100 unique values.
Handling Encoded Data:

If OneHotEncoder is used, the encoded data is concatenated back to the original numerical columns.
If LeaveOneOutEncoder or CatBoostEncoder is used, the original categorical columns are replaced with the encoded values.

4. Principal Component Analysis (PCA) Function (perform_pca)
PCA is performed on the preprocessed features to reduce dimensionality and capture the variance in the data.
Explained Variance Ratio: The ratio of variance captured by each principal component is calculated.

5. Scree Plot Function (plot_scree)
Plots the cumulative explained variance ratio against the number of principal components.
This plot helps visualize how much of the variance in the data is captured by the first few principal components.

6. Model Evaluation for Classification (evaluate_classification_models)
Trains and evaluates several classification models (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, Support Vector Classifier, XGBoost).
Metrics Evaluated: Accuracy, Precision, Recall, F1 Score, and Training Time.

7. Model Evaluation for Regression (evaluate_regression_models)
Trains and evaluates several regression models (Linear Regression, Decision Tree, Random Forest, Gradient Boosting, Support Vector Regressor, XGBoost).
Metrics Evaluated: Mean Squared Error (MSE), R^2 Score, Mean Absolute Error (MAE), and Training Time.

8. Main Script
Load Dataset: The dataset is loaded using the load_data function.
Preprocess Data: The dataset is preprocessed using the preprocess_data function.
PCA Analysis: PCA is performed, and the cumulative variance explained by all components is plotted.
Split Data: The dataset is split into training and testing sets.
Model Evaluation: Depending on whether the task is classification or regression, the relevant models are trained, evaluated, and their performance metrics are printed.

Purpose: The code is a comprehensive pipeline that takes in a dataset, preprocesses it (including handling categorical data), performs PCA to reduce dimensionality, and evaluates different machine learning models based on the task type (classification or regression).

# Computer Vision

Data Preprocessing
Loading and Preprocessing Image Data:

The load_image_data function uses TensorFlow's ImageDataGenerator to load images from a directory structure, rescale the pixel values, and generate batches of image data. It processes images into a numpy array format suitable for further analysis.
The function returns the images and their corresponding labels as numpy arrays.
Principal Component Analysis (PCA):

The perform_pca function reshapes the images (which are 3D arrays) into 2D arrays where each row corresponds to a flattened image. PCA is then performed to reduce the dimensionality of the image data, capturing the most significant variance in the data.
The explained variance ratio of each principal component is returned, along with the PCA model itself.
Plotting the Scree Plot:

The plot_scree function visualizes the explained variance ratio for each principal component in a bar chart. This plot helps in understanding how many components are necessary to capture a significant amount of variance in the data.
Model Development and Evaluation
Building and Evaluating the VGG16-Based Model:
The evaluate_image_models function initializes a VGG16 model pre-trained on ImageNet but excludes the top (classification) layer.
A fully connected (dense) layer and an output layer tailored to the number of classes in the dataset are added on top of the base model.
The base model layers are frozen (non-trainable) to leverage the pre-trained features.
The model is compiled with the Adam optimizer and trained on the training dataset. The training time and accuracy on the test dataset are recorded.
Example Usage
Loading Data:

The example shows how to load image data from a directory, where images are organized into subfolders by class.
PCA Analysis:

PCA is applied to the flattened image data, and a scree plot is generated to visualize the explained variance by each component. The total variance captured by PCA is also printed.
Model Evaluation:

The data is split into training and test sets using train_test_split.
The VGG16-based model is trained and evaluated on the test set, with the results (training time and accuracy) printed to the console.

# Time-Series Analysis

Data Preprocessing
Loading Data:

The load_data function reads data from CSV, JSON, or Excel files into a pandas DataFrame. The file path and format are provided as inputs.
Preprocessing Time-Series Data:

The preprocess_timeseries_data function prepares the data for analysis:
It drops the target variable from the features.
It identifies and converts datetime columns to the correct dtype and then drops them from the features.
Missing values in numeric columns are filled with the median value.
Categorical variables are one-hot encoded using OneHotEncoder.
Numeric features are standardized using StandardScaler.
Principal Component Analysis (PCA):

The perform_pca function applies PCA to the preprocessed data to reduce dimensionality and captures the variance explained by each principal component.
Plotting the Scree Plot:

The plot_scree function creates a bar chart to visualize the explained variance ratio for each principal component, helping to determine the number of components needed.
Model Development and Evaluation
Evaluating the ARIMA Model:
The evaluate_arima_model function fits an ARIMA model to the training data, forecasts the test data, and evaluates the model's performance using metrics like Mean Squared Error (MSE), R-squared, and Mean Absolute Error (MAE).
The model order (p, d, q) is provided as input to the ARIMA function, which determines the autoregressive, differencing, and moving average parameters.
