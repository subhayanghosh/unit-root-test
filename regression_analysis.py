import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

# Read the CSV file into a pandas DataFrame
data = pd.read_csv('aggr_dataset.csv')

# Preprocess the data
# Convert trade balance from string to float
data['TRADE BALANCE'] = data['TRADE BALANCE'].replace(
    {',': ''}, regex=True).astype(float)

# Define independent and dependent variables
X = data[['EXCHANGE RATE', 'TRADE BALANCE', 'UNEMPLOYMENT RATE']]
y = data['GDP']

# Add constant to the independent variables (for intercept)
X = sm.add_constant(X)

# Fit the regression model
model = sm.OLS(y, X).fit()

# Print the regression results
print(model.summary())
