import pandas as pd
from statsmodels.tsa.stattools import adfuller

df = pd.read_csv('student_exam_data.csv')
print(df.head())

# Function to perform ADF test


def adf_test(data, variable_name):
    result = adfuller(data)
    print(f"ADF Statistic for {variable_name}: {result[0]}")
    print(f"p-value for {variable_name}: {result[1]}")
    print(f"Critical Values for {variable_name}:")
    for key, value in result[4].items():
        print(f"   {key}: {value}")


# Perform ADF test for 'Study Hours'
print("\nADF test for 'Study Hours':")
adf_test(df['Study Hours'], 'Study Hours')
print()

# Perform ADF test for 'Previous Exam Score'
print("\nADF test for 'Previous Exam Score':")
adf_test(df['Previous Exam Score'], 'Previous Exam Score')
