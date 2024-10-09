# Sales-prediction
Advertising Data Analysis with Linear Regression
This project demonstrates the use of linear regression to predict sales based on advertising expenditure data for TV, Radio, and Newspaper advertisements. It uses Python libraries such as Pandas, NumPy, Matplotlib, Seaborn, and Scikit-learn to perform data exploration, model training, and evaluation.

Table of Contents
Project Overview
Dataset
Libraries Used
Exploratory Data Analysis (EDA)
Model Training
Model Evaluation
Results
How to Run
Project Overview
The goal of this project is to predict sales based on the amount spent on TV, Radio, and Newspaper advertising. A linear regression model is trained and evaluated on the dataset, which contains information about advertising budgets and corresponding sales.

Dataset
The dataset used in this project is advertising.csv. It contains 200 records with four columns:

TV: Budget spent on TV advertising
Radio: Budget spent on Radio advertising
Newspaper: Budget spent on Newspaper advertising
Sales: Sales generated as a result of the advertising campaigns
Libraries Used
The following Python libraries are used in this project:

Pandas: For data manipulation and analysis.
NumPy: For numerical operations.
Matplotlib and Seaborn: For data visualization.
Scikit-learn: For model training and evaluation.
Exploratory Data Analysis (EDA)
We begin by loading the dataset and performing some basic data exploration:

Check the first few rows with sales.head().
Examine the structure of the dataset using sales.info().
Display summary statistics with sales.describe().
Pairplot and Correlation Heatmap
A pairplot is created to visualize relationships between different features using sns.pairplot(sales).
A heatmap is plotted to examine the correlation between features using sns.heatmap(sales.corr(), annot=True).
Sales Distribution
The distribution of sales is plotted with a histogram and KDE (Kernel Density Estimation) using sns.histplot().
Model Training
The features used for training are TV, Radio, and Newspaper, and the target variable is Sales.

Data Splitting: The data is split into training and testing sets using train_test_split from Scikit-learn.

python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)
Linear Regression Model: A linear regression model is created and trained using Scikit-learn's LinearRegression.

python
lm = LinearRegression()
lm.fit(X_train, y_train)
Model Coefficients: The coefficients for the features are stored in a DataFrame.

python
cdf = pd.DataFrame(lm.coef_, X.columns, columns=['Coeff'])
Model Evaluation
The trained model's predictions on the test set are compared with the actual sales values. A scatter plot is created to visualize the accuracy of predictions, and a histogram of the prediction errors is plotted.

Evaluation Metrics
The following evaluation metrics are used:

Mean Absolute Error (MAE)
Mean Squared Error (MSE)
Root Mean Squared Error (RMSE)
python
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
Results
MAE: 1.4009
MSE: 3.0406
RMSE: 1.7437
The scatter plot of actual vs. predicted sales indicates that our model performs well, and the distribution of the errors shows a near-normal distribution, which is a good sign for linear regression.

How to Run
Install the required libraries:
bash
pip install pandas numpy matplotlib seaborn scikit-learn
Download the dataset advertising.csv and place it in the project directory.
Run the Jupyter notebook or Python script to execute the analysis.
