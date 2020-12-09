# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
%matplotlib inline
df = pd.read_csv('../data/train_edited.csv')

# %%

    # Plot the model
    plt.figure(figsize=(12,9))

    # Generate a scatterplot of predicted values versus actual values.
    plt.scatter(preds, y, s=5, color='teal', alpha = 0.3)
    #sns.lmplot(x='Predicted', y="SalePrice", data=test_data, ci=False, order=2);
    # Plot a line.
    plt.plot([0, np.max(y)],
             [0, np.max(y)],
             color = 'black')

    # Tweak title and axis labels.
    plt.xlabel("Predicted Values: $\hat{y}$", fontsize = 20)
    plt.ylabel("Actual Values: $y$", fontsize = 20)
    plt.title('Predicted Values vs. Actual Values', fontsize = 24);

# %%
residuals = y - preds

for col in X:
    print(col)
    plt.figure(figsize=(8,6))
    plt.scatter(X[col], residuals)
    plt.xlabel(col)
    plt.ylabel('Residuals')
    plt.title(f'{col} vs. Residuals')
    plt.show()

plt.figure(figsize=(10,6))
plt.scatter(preds, residuals)
plt.axhline(c='red', ls='dashed')
plt.title('Scatterplot of Residuals vs. Predictions');

# %%
sns.distplot(residuals)
