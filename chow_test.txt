import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
from scipy.stats import f

# Load data from CSV
df = pd.read_csv('gdp_data.csv')

# Handle formatting of 'TRADE_BALANCE' column
df['TRADE BALANCE'] = df['TRADE BALANCE'].str.replace(',', '').astype(float)

# Create a dummy variable for the break period
df['dummy'] = (df['YEAR'] > 2018).astype(int)

# Separate data for periods before and after 2018
data_before = df[df['YEAR'] <= 2018]
data_after = df[df['YEAR'] > 2018]

# Regression before 2018
X_before = data_before[['EXCHANGE RATE',
                        'TRADE BALANCE', 'UNEMPLOYMENT RATE', 'dummy']]
y_before = data_before['GDP']
X_before = sm.add_constant(X_before)
model_before = sm.OLS(y_before, X_before).fit()

# Regression after 2018
X_after = data_after[['EXCHANGE RATE',
                      'TRADE BALANCE', 'UNEMPLOYMENT RATE', 'dummy']]
y_after = data_after['GDP']
X_after = sm.add_constant(X_after)
model_after = sm.OLS(y_after, X_after).fit()

# Chow test
SSR_pooled = model_before.ssr + model_after.ssr
SSR_before = model_before.ssr
SSR_after = model_after.ssr
df_before = len(y_before) - (len(model_before.params) - 1)
df_after = len(y_after) - (len(model_after.params) - 1)
df_pooled = df_before + df_after
F = ((SSR_pooled - (SSR_before + SSR_after)) / (SSR_before + SSR_after)) / \
    (df_pooled / (SSR_before + SSR_after))
p_value = 1 - stats.f.cdf(F, df_before, df_after)

print("Chow Test Statistic:", F)
print("P-value:", p_value)
