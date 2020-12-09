# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import statsmodels.api as sm 
%matplotlib inline
df = pd.read_csv('../data/train_edited.csv')

# %%
mdl = LinearRegression()
y =  df.SalePrice
features = ['Overall_Qual','Total_SQFT','Year_Built','Garage_Area']
X = df[features]
Paved = pd.get_dummies(df['Street'])
Paved = Paved['Pave']
X = X.join(Paved)

feature_names = ['Overall_Qual','Total_SQFT','Year_Built','Garage_Area', 'Pave']

# Instantiate PolynomialFeatures
poly = PolynomialFeatures(include_bias=False)

# Create X_poly
X_poly = poly.fit_transform(X)

# View X_poly in a DataFrame
X_poly_df = pd.DataFrame(X_poly, columns=poly.get_feature_names(feature_names))

# %%
X_train, X_test, y_train, y_test = train_test_split(X_poly_df, y)
# %%
ss=StandardScaler()
ss.fit(X_train)
X_train_ss= ss.transform(X_train)
X_test_ss = ss.transform(X_test)
# %%
mdl = LinearRegression()
mdl.fit(X_train_ss, y_train)
# preds = mdl.predict(X_poly_)
cross_val_score(mdl, X_train_ss, y_train, cv = 5).mean()

# %%
cross_val_score(mdl, X_test_ss, y_test, cv = 5).mean()

# %%
