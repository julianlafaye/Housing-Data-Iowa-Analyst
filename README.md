
# Ames Iowa Housing Data

This Project explores modeling opportuities in a housing dataset obtained from Ames Iowa

## EDA
Looking For Correlations we found that the main predictors ended up being sqft and Overall quality variables.

![](images\Correlation_to_Sale_Price.png)



However there were also good predictors that were categorical and thus could not be account for by correaltion alone

![](images\Price_by_Neighborhood.png)

![](images\Sale_Price_by_Decade.png)

## Preprocessing and Model Fitting

For my Model I decided to use 'Overall_Qual','Total_SQFT','Year_Built','Garage_Area', 'Pave', and 'Neighborhood' as main features. I transformed these into Polynomial Features and scaled them as well.

![](images\ridge_predictions.png)

![](images\lasso_predictions.png)