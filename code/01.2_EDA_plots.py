#%%|
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set(style="whitegrid")
test = pd.read_csv('../data/test_edited.csv')
train = pd.read_csv('../data/train_edited.csv')
default_cmap = sns.diverging_palette(220, 20, n=13)

# %%
#### All Variables Except Sale Price ####
all_but = ['Id', 'PID', 'MS_SubClass', 'MS_Zoning', 'Lot_Frontage',
       'Lot_Area', 'Street', 'Alley', 'Lot_Shape', 'Land_Contour', 'Utilities',
       'Lot_Config', 'Land_Slope', 'Neighborhood', 'Condition_1',
       'Condition_2', 'Bldg_Type', 'House_Style', 'Overall_Qual',
       'Overall_Cond', 'Year_Built', 'Year_Remod/Add', 'Roof_Style',
       'Roof_Matl', 'Exterior_1st', 'Exterior_2nd', 'Mas_Vnr_Type',
       'Mas_Vnr_Area', 'Exter_Qual', 'Exter_Cond', 'Foundation', 'Bsmt_Qual',
       'Bsmt_Cond', 'Bsmt_Exposure', 'BsmtFin_Type_1', 'BsmtFin_SF_1',
       'BsmtFin_Type_2', 'BsmtFin_SF_2', 'Bsmt_Unf_SF', 'Total_Bsmt_SF',
       'Heating', 'Heating_QC', 'Central_Air', 'Electrical', '1st_Flr_SF',
       '2nd_Flr_SF', 'Low_Qual_Fin_SF', 'Gr_Liv_Area', 'Bsmt_Full_Bath',
       'Bsmt_Half_Bath', 'Full_Bath', 'Half_Bath', 'Bedroom_AbvGr',
       'Kitchen_AbvGr', 'Kitchen_Qual', 'TotRms_AbvGrd', 'Functional',
       'Fireplaces', 'Fireplace_Qu', 'Garage_Type', 'Garage_Yr_Blt',
       'Garage_Finish', 'Garage_Cars', 'Garage_Area', 'Garage_Qual',
       'Garage_Cond', 'Paved_Drive', 'Wood_Deck_SF', 'Open_Porch_SF',
       'Enclosed_Porch', '3Ssn_Porch', 'Screen_Porch', 'Pool_Area', 'Pool_QC',
       'Fence', 'Misc_Feature', 'Misc_Val', 'Mo_Sold', 'Yr_Sold', 'Sale_Type', 'Total_SQFT']

# %%
############## PAIRPLOT OF EVERY VARIABLE ##############
####################### DNR ############################
# sns.pairplot(df, x_vars= all_but, y_vars='SalePrice')#

# %%
###HEATMAP#####
default_cmap = sns.diverging_palette(220, 20, n=13)
plt.figure(figsize=(2,12))
sns.heatmap(train.corr()[['SalePrice']].sort_values('SalePrice'), 
            cmap=default_cmap, 
            annot=True,
            vmax=1,
            vmin=-1)
# Thanks SalMac86! 
# https://github.com/mwaskom/seaborn/issues/1773
b, t = plt.ylim() # discover the values for bottom and top 
b += 0.5 # Add 0.5 to the bottom
t -= 0.5 # Subtract 0.5 from the top
plt.ylim(b, t) # update the ylim(bottom, top) values
plt.title("Correlation to Sale Price")
plt.savefig('../images/Correlation_to_Sale_Price', bbox_inches="tight")
# %%
### Sub Class Boxplot ###
SubClass_order = train.groupby('MS_SubClass').mean()['SalePrice'].sort_values().index

sns.set(style="whitegrid")
ax = sns.boxenplot(x="MS_SubClass", y="SalePrice",
              color="b", order= SubClass_order,
              scale="linear", data=train)
plt.title('Prices By Sub Class')
plt.savefig('../images/Prices_By_Sub_Class', bbox_inches="tight")
print("# Reference:___________________________________________________________________\n"
"# |                                                                           |\n"
"# |   20..............1-STORY 1946 & NEWER ALL STYLES                         |\n"
"# |   30..............1-STORY 1945 & OLDER                                    |\n"
"# |   40..............1-STORY W/FINISHED ATTIC ALL AGES                       |\n"
"# |   45..............1-1/2 STORY - UNFINISHED ALL AGES                       |\n"
"# |   50..............1-1/2 STORY FINISHED ALL AGES                           |\n"
"# |   60..............2-STORY 1946 & NEWER                                    |\n"
"# |   70..............2-STORY 1945 & OLDER                                    |\n"
"# |   75..............2-1/2 STORY ALL AGES                                    |\n"
"# |   80..............SPLIT OR MULTI-LEVEL                                    |\n"
"# |   85..............SPLIT FOYER                                             |\n"
"# |   90..............DUPLEX - ALL STYLES AND AGES                            |\n"
"# |   120.............1-STORY PUD (Planned Unit Development) - 1946 & NEWER   |\n"
"# |   150.............1-1/2 STORY PUD - ALL AGES                              |\n"
"# |   160.............2-STORY PUD - 1946 & NEWER                              |\n"
"# |   180.............PUD - MULTILEVEL - INCL SPLIT LEV/FOYER                 |\n"
"# |   190.............2 FAMILY CONVERSION - ALL STYLES AND AGES               |\n"
"# |___________________________________________________________________________|\n")

## %%
### Lot Slope ###
# sns.set(style="whitegrid")
# sns.boxenplot(x="Land_Contour", y="SalePrice",
#               color="teal", order=["Lvl", "Bnk", "HLS", "Low"],
#               scale="linear", data=train)
# plt.title('Prices by Lot Slope')

# %%
### Street Type ###
sns.set(style="whitegrid")
sns.boxenplot(x="Street", y="SalePrice",
              color="grey", scale="linear", data=train)
plt.title('Prices by Street Material')
plt.savefig('../images/Streets', bbox_inches="tight")

# %%
### Decade Built Boxplot ###
decades=[i*10+1870 for i in range(15)]
decades_Labels=[i*10+1870 for i in range(14)]
year_bins = pd.cut(train['Year_Built'],                          
                            right=True, 
                            include_lowest=True,
                            bins=decades,
                            labels= decades_Labels)
train['Decade_Built'] = year_bins
sns.set(style="whitegrid")
sns.boxenplot(x="Decade_Built", y="SalePrice", order= decades_Labels,
              color="purple", scale="linear", data=train)
plt.xticks(rotation=45)
plt.title("Prices by Decade Built")
plt.savefig('../images/Sale_Price_by_Decade', bbox_inches="tight")

# # %%
# ### Year Built vs Sale Price ###
# sns.set(style="white")
# g = sns.jointplot(train['Year_Built'], train['SalePrice'], 
#                         kind= 'scatter', color="darkslategrey", alpha=0.2)
# plt.title(' Year Built vs Sale Price ')

# # %%
# ### Fireplaces ###
# sns.set(style="whitegrid")
# sns.boxenplot(x="Fireplaces", y="SalePrice",
#               color="r", scale="linear", data=train)
# plt.title("Price by # of Fireplaces")
# %%
### Neighborhood Boxplot ###
train.groupby('Neighborhood').mean()["Year_Built"]
neighborhood_order = train.groupby('Neighborhood').mean()['SalePrice'].sort_values().index
neighborhood_labels = [item[:3] for item in neighborhood_order]
sns.set(style="whitegrid")
plt.figure(figsize=(10,4))
ax = sns.boxenplot(x="Neighborhood", y="SalePrice", orient="v",
              color="limegreen", order= neighborhood_order,
              scale="linear", data=train)
#ax.set_xticklabels(neighborhood_labels)
ax.xaxis.grid(True)
plt.xticks(rotation=45,)
plt.title("Price by Neighborhood")
plt.savefig('../images/Price_by_Neighborhood', bbox_inches="tight")

# %%
### Quality Boxplot ###
sns.set(style="whitegrid")
sns.boxenplot(x="Overall_Qual", y="SalePrice",
              color="g", order= [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              scale="linear", data=train)
plt.title("Price by Quality")
plt.savefig('../images/Price_by_Quality', bbox_inches="tight")
# %%
# ### Sale Type Boxplot ###
# sns.set(style="whitegrid")
# sns.boxenplot(x="Sale_Type", y="SalePrice",
#               color="g",
#               scale="linear", data=train)
# plt.title('Price by Sale Type')
# %%
### Sale Price distplot ###
sns.set(style="whitegrid")
print( f"Mean Sale Price: ${round(train['SalePrice'].mean())}")
sns.distplot(train['SalePrice'], hist=False, rug=True, color="g")
plt.title("Sale Prices")
plt.savefig('../images/Sale_Prices', bbox_inches="tight")

## %%
# ### Sale Price by Lot Area ###
# g = (sns.jointplot(train['Lot_Area'], train['SalePrice'], kind= 'scatter', color="darkslategrey",alpha= .2))
# g.ax_marg_x.set_xlim(0,20_000 )
# plt.title('Slae Price by Lot Area')


# %%
### sqft. distpllot ###
sns.set(style="whitegrid")
print( f"Mean SQF: {round(train['Total_SQFT'].mean())} sqft.")
sns.distplot(train['Total_SQFT'], hist=False, rug=True, color="r")
plt.title('Total SqFt.')
plt.savefig('../images/SqFts', bbox_inches="tight")

# %%
###  Total sqft vs Sale Price ###
sns.set_style('dark')
g = (sns.jointplot(train['Total_SQFT'], train['SalePrice'], kind= 'reg', color="darkslategrey", joint_kws = {'scatter_kws':dict(alpha=0.1)}))
g.ax_marg_x.set_xlim(0, 11_000)
plt.savefig('../images/Sale_Price_by_Total_SqFt', bbox_inches="tight")

# %%
### Overall Quality Distplot ###
sns.set(style="white", palette="muted", color_codes=True)
ax = sns.distplot(train['Overall_Qual'], kde=False, hist=True, color="r").set(xticks= [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
plt.title("Overall Quality")
plt.savefig('../images/Overall_Quality', bbox_inches="tight")

#####Plot Templates######
# ###BOXENPLOT###
# sns.boxenplot(x="MS SubClass", y="SalePrice",
#               color="b", 
#               scale="linear", data=train_dataset)
# ###SCATTER###
# sns.set_style('dark')
# g = (sns.jointplot(train['Total_SQFT'], train['SalePrice'], kind= 'reg', color="darkslategrey", joint_kws = {'scatter_kws':dict(alpha=0.1)}))
# g.ax_marg_x.set_xlim(0, 11_000)
# ###DISTPLOT###
# sns.set(style="whitegrid")
# print( f"Mean SQF: {round(train['Total_SQFT'].mean())} sqft.")
# sns.distplot(train['Total_SQFT'], hist=False, rug=True, color="r")
# ###HEATMAP#####
# default_cmap = sns.diverging_palette(220, 20, n=13)
# plt.figure(figsize=(12, 12))
# sns.heatmap(train_dataset.corr(), 
#             cmap=default_cmap, 
#             annot=True,
#             vmax=1,
#             vmin=-1)
# # Thanks SalMac86! 
# # https://github.com/mwaskom/seaborn/issues/1773
# b, t = plt.ylim() # discover the values for bottom and top 
# b += 0.5 # Add 0.5 to the bottom
# t -= 0.5 # Subtract 0.5 from the top
# plt.ylim(b, t) # update the ylim(bottom, top) values
