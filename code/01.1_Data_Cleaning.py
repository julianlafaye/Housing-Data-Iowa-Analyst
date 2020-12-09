# %%
import numpy as np
import pandas as pd
import seaborn as sns
import missingno as msno

try:
    train = pd.read_csv('./datasets/train.csv') ## 
    test = pd.read_csv('./datasets/test.csv') ## 
except:
    train = pd.read_csv('../datasets/train.csv') ## 
    test = pd.read_csv('../datasets/test.csv') ## 

pd.set_option('display.max_columns', None)

# %%
###
train.columns = train.columns.str.replace(' ', '_')
test.columns = test.columns.str.replace(' ', '_')
### 
sqf_cols = [col for col in train.columns if 'SF' in col]
train['Total_SQFT'] = train[sqf_cols].sum(axis=1)
sqf_cols = [col for col in test.columns if 'SF' in col]
test['Total_SQFT'] = test[sqf_cols].sum(axis=1)

train.corr()['SalePrice'].sort_values()
train.at[1712, 'Garage_Area'] = 0
sqft_outliers = train[train['Total_SQFT']> 15000].index
list(sqft_outliers)
train = train.drop(sqft_outliers)

############################ Finding High Correlations ####################################
###########################################################################################
## # Create correlation matrix                                                           ##
## corr_matrix = train.corr().abs()                                                      ##
## # Select upper triangle of correlation matrix                                         ##
## upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))   ##
## # Find index of feature columns with correlation greater than 0.8                     ##
## Higher_Corr = [column for column in upper.columns if any(upper[column] > 0.8)]        ##
###########################################################################################

Higher_Corr = ["Full_Bath",      
"Year_Remod/Add",
"Year_Built",     
"1st_Flr_SF",     
"Total_Bsmt_SF",  
"Garage_Cars",    
"Garage_Area",    
"Gr_Liv_Area",    
"Total_SQFT",      
"Overall_Qual",   
"SalePrice"]

train.to_csv('../data/train_edited.csv', index=False)
test.to_csv('../data/test_edited.csv',index=False)
print("#### Data Cleaned! ####")


# %%


# %%
