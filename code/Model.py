# %% Imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
%matplotlib inline

try:
    df = pd.read_csv('../data/train_edited.csv')
    test = pd.read_csv('../data/test_edited.csv') ## 
except:
    df = pd.read_csv('./data/train_edited.csv')
    test = pd.read_csv('./data/test_edited.csv')
# %% Check Dataset
df.info()

# %% Select Features
#numeric_features = list(df.drop(columns='SalePrice').drop(columns='MS_SubClass')._get_numeric_data().columns)
numeric_features = ['Overall_Qual','Total_SQFT','Year_Built','Garage_Area', 'Overall_Cond' ]
categorical_features =  ['Neighborhood', 'Street','MS_SubClass', 'MS_Zoning', 'Bldg_Type', 'Utilities']

#categorical_features = list(df.drop(columns=numeric_features).drop(columns='SalePrice'))
#categorical_features.append('MS_SubClass')
# all_else = list(df.drop(columns=numeric_features).drop(columns='SalePrice').drop(columns= categorical_features))

# %% Build pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='NA')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

PolynomialStandardScaler = Pipeline(steps=[
    ('polynomial', PolynomialFeatures()),
    ('scalar', StandardScaler(with_mean=False))])


# df = pd.get_dummies(data = df, columns= categorical_features, drop_first=True)


# %% Ridge Pipeline
ridge_clf = Pipeline(steps=[
                     ('preprocessor', preprocessor),
                     ('poly_ss', PolynomialStandardScaler),
                     ('classifier', RidgeCV())])
# %% Lasso Pipeline
lasso_clf = Pipeline(steps=[
                     ('preprocessor', preprocessor),
                     ('poly_ss', PolynomialStandardScaler),
                     ('classifier', LassoCV())])
# %% Train Test Split
X = df.drop(columns='SalePrice').fillna(0) # No numerical values were filled with NA only categorical
y = df['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# %% Fit the Model (Lasso)
lasso_clf.fit(X_train, y_train); 

# %% Test and Train Model Scores (Lasso)
print(f"model score: {lasso_clf.score(X_train, y_train)}")

print(f"model score: {lasso_clf.score(X_test, y_test)}")

# %% Make Predictions (Lasso)

lasso_preds = lasso_clf.predict(test)

# %% Save to csv
ids = test['Id']
lasso_df = pd.DataFrame(lasso_preds)
lasso_df = lasso_df.join(ids)
lasso_df = lasso_df.rename({0:'SalePrice'}, axis=1)
lasso_df.to_csv('../data/submission_lasso.csv')

# %% Fit the Model (Ridge)
ridge_clf.fit(X_train, y_train)

# %% Test and Train Model Scores (Ridge)
print(f"model score: {ridge_clf.score(X_train, y_train)}")

print(f"model score: {ridge_clf.score(X_test, y_test)}")

# %% Make Predictions (Ridge)

ridge_preds = ridge_clf.predict(test)

# %% Save to csv
ids = test['Id']
ridge_df = pd.DataFrame(ridge_preds)
ridge_df = ridge_df.join(ids)
ridge_df = ridge_df.rename({0:'SalePrice'}, axis=1)
ridge_df.to_csv('../data/submission_ridge.csv')


# %% Lasso Predictions plot

    # Plot the model
    plt.figure(figsize=(12,9))

    # Generate a scatterplot of predicted values versus actual values.
    plt.scatter(lasso_clf.predict(X_train), y_train, s=5, color='teal', alpha = 0.3)
    #sns.lmplot(x='Predicted', y="SalePrice", data=test_data, ci=False, order=2);
    # Plot a line.
    plt.plot([0, np.max(y)],
             [0, np.max(y)],
             color = 'black')

    # Tweak title and axis labels.
    plt.xlabel("Predicted Values: $\hat{y}$", fontsize = 20)
    plt.ylabel("Actual Values: $y$", fontsize = 20)
    plt.title('Predicted Values vs. Actual Values for Lasso', fontsize = 24);
    
    plt.savefig('../images/lasso_predictions.png')

# %% Ridge Predictions plot

    # Plot the model
    plt.figure(figsize=(12,9))

    # Generate a scatterplot of predicted values versus actual values.
    plt.scatter(ridge_clf.predict(X_train), y_train, s=5, color='teal', alpha = 0.3)
    #sns.lmplot(x='Predicted', y="SalePrice", data=test_data, ci=False, order=2);
    # Plot a line.
    plt.plot([0, np.max(y)],
             [0, np.max(y)],
             color = 'black')

    # Tweak title and axis labels.
    plt.xlabel("Predicted Values: $\hat{y}$", fontsize = 20)
    plt.ylabel("Actual Values: $y$", fontsize = 20)
    plt.title('Predicted Values vs. Actual Values for Ridge', fontsize = 24);
    
    plt.savefig('../images/ridge_predictions.png')

# %%
