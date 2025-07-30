# Table of Contents

- [Table of Contents](#table-of-contents)
  - [A typical ML workﬂow/pipeline](#a-typical-ml-workﬂowpipeline)
  - [Typical Tasks](#typical-tasks)
  - [Terminology](#terminology)
  - [Data Exploration](#data-exploration)
    - [Variables](#variables)
    - [Univariate Analysis](#univariate-analysis)
    - [Bi-variate Analysis](#bi-variate-analysis)
  - [Feature Cleaning](#feature-cleaning)
    - [Missing Values](#missing-values)
      - [Why Missing](#why-missing)
      - [Missing Mechanisms](#missing-mechanisms)
      - [Identifying Missing Values in which mechanism](#identifying-missing-values-in-which-mechanism)
      - [How to Assume a Missing Mechanism](#how-to-assume-a-missing-mechanism)
      - [How to Handle Missing Data](#how-to-handle-missing-data)
    - [Outliers](#outliers)
    - [Rare Values](#rare-values)
    - [High Cardinality](#high-cardinality)
  - [Feature Engineering](#feature-engineering)
    - [Scaling](#scaling)
    - [Discretize](#discretize)
    - [Encoding](#encoding)
    - [Transformation](#transformation)
    - [Generation](#generation)
  - [Feature Selection](#feature-selection)
    - [Filter Method](#filter-method)
    - [Wrapper Method](#wrapper-method)
    - [Embedded Method](#embedded-method)
    - [Feature Shuﬄing](#feature-shuﬄing)
    - [Hybrid Method](#hybrid-method)
    - [Dimensionality Reduction](#dimensionality-reduction)

## A typical ML workﬂow/pipeline

![alt text](image.png)

## Typical Tasks

![alt text](image-1.png)

## Terminology

![alt text](image-2.png)

## Data Exploration

### Variables

![alt text](image-3.png)

### Univariate Analysis

![alt text](image-4.png)

### Bi-variate Analysis

![alt text](image-5.png)

## Feature Cleaning

### Missing Values

#### Why Missing

- no value is stored in a certain observation within a variable.
- certain algorithms cannot work when missing value are present
- even for algorithm that handle missing data, without treatment the model can lead to inaccurate conclusion

#### Missing Mechanisms

- Depending on the mechanism, we may choose to process the missing values diﬀerently.

1. Missing Completely at Random (MCAR):
   - Missing data has no relationship with any values in the dataset
   - Randomly missing across all observations
   - Excluding these cases doesn't bias results

2. Missing at Random (MAR):
   - Missing data is related to other observed variables
   - Example: Men more likely to reveal weight than women
   - Can be controlled by including related variables in analysis

3. Missing Not at Random (MNAR):
   a. Depends on Unobserved Predictors:
      - Missing data related to unmeasured factors
      - Example: Treatment discomfort causing study dropouts
   
   b. Depends on Missing Value Itself:
      - Probability of missing data depends on the value itself
      - Example: Higher earners less likely to reveal income

#### Identifying Missing Values in which mechanism

1. Analyze patterns:
   - Look for relationships between missing data and other variables
   - Use visualization tools or statistical tests to identify patterns

2. Domain knowledge:
   - Understand the context of your data
   - Consider reasons why data might be missing

3. Logical deduction:
   - MCAR is rare in real-world scenarios
   - MAR is more common and often assumed
   - MNAR requires careful consideration of the variable itself

4. Little's MCAR test:
   - A statistical test to check if data is MCAR
   - If the test fails, data is likely MAR or MNAR

5. Sensitivity analysis:
   - Compare results using different missing data handling methods
   - Large differences may indicate MNAR

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# Create a sample dataset
np.random.seed(42)
n = 1000

df = pd.DataFrame({
    'age': np.random.randint(18, 80, n),
    'gender': np.random.choice(['M', 'F'], n),
    'income': np.random.normal(50000, 15000, n)
})

# Introduce missing values
df.loc[df['gender'] == 'F', 'income'] = np.where(np.random.random(n) < 0.3, np.nan, df.loc[df['gender'] == 'F', 'income'])
df.loc[df['age'] > 60, 'income'] = np.where(np.random.random(n) < 0.4, np.nan, df.loc[df['age'] > 60, 'income'])

# Function to check missing data
def analyze_missing_data(df):
    print("Missing data summary:")
    print(df.isnull().sum())
    print("\nPercentage of missing data:")
    print(df.isnull().sum() / len(df) * 100)
    
    # Visualize missing data patterns
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis')
    plt.title('Missing Data Heatmap')
    plt.show()
    
    # Check relationship between missing income and gender
    gender_missing = pd.crosstab(df['gender'], df['income'].isnull())
    chi2, p_value, _, _ = chi2_contingency(gender_missing)
    print("\nRelationship between missing income and gender:")
    print(f"Chi-square statistic: {chi2}")
    print(f"p-value: {p_value}")
    
    # Check relationship between missing income and age
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df['income'].isnull(), y=df['age'])
    plt.title('Age Distribution for Missing and Non-Missing Income')
    plt.show()

# Analyze the missing data
analyze_missing_data(df)
```

#### How to Assume a Missing Mechanism

- By business understanding. In many situations we can assume the mechanism by probing into the
business logic behind that variable.

- By statistical test. Divide the dataset into ones with/without missing and perform t-test to see if
there's signiﬁcant diﬀerences. If there is, we can assume that missing is not completed at random.

#### How to Handle Missing Data 

![alt text](image-6.png)

- popular way is to adopt:
  - Mean/Median/Mode Imputation (depend on the distribution)
  - End of distribution Imputation
  - Add a variable to denote NA
  - Some algorithms like XGboost incorporate missing data treatment into its model building process

### Outliers

### Rare Values

### High Cardinality

## Feature Engineering

### Scaling
### Discretize
### Encoding
### Transformation
### Generation

## Feature Selection

### Filter Method
### Wrapper Method
### Embedded Method
### Feature Shuﬄing
### Hybrid Method
### Dimensionality Reduction