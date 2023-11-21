# Practical-Application-Assignment_11_1

## What Drives the Price of a Car?

## Context

## Business Understanding

## Data Understanding

### Data Descriptions

#### Here are some Pie Charts that show proportions in the features selected.
<img src="images/condition_pie_chart.png"/>
<img src="images/cylinders_pie_chart.png"/>
<img src="images/fuel_pie_chart.png"/>
<img src="images/title_pie_chart.png"/>
<img src="images/transmission_pie_chart.png"/>
<img src="images/drive_pie_chart.png"/>
<img src="images/size_pie_chart.png"/>
<img src="images/type_pie_chart.png"/>
<img src="images/manufacturer_pie_chart.png"/>
<img src="images/paint_color_pie_chart.png"/>

#### Here are some Kernel Density Estimate plots visualizing the distribution of observations for a few features.
<img src="images/condition_kdeplot.png"/>
<img src="images/cylinders_kdeplot.png"/>
<img src="images/drive_kdeplot.png"/>
<img src="images/size_kdeplot.png"/>
<img src="images/type_kdeplot.png"/>


#### Here are some Heatmaps visualizing the magnitude of individual values.
##### We can see that "odometer" has a negative correlation with price (-0.4) and model year (-0.47).
##### We can see that "drive" has a negative correlation with price (-0.18) and model year (-0.12).
##### We can see that "size" has a negative correlation with price (-0.088) and model year (-0.075).
##### This feature will be dropped.
<img src="images/heatmap.png"/>

##### Here is the Heat Map that will drive the selection of features for our modeling.
<img src="images/drop_size_heatmap.png"/>


### Modeling
#### Using "train_test_split" on the data we derived from the Heatmaps, we end up with 
21,733 records for training
 9,314 records for testing

#### Applying Linear Regression
Coefficients: 
[5.75105537e-01 3.00495003e+03 3.26941663e+02 5.24749893e+02]
Intercept: 0.00
Mean squared error: 149,420,617.16
Mean absolute error: 9,153.10
R2 Score: 0.12

#### Applying Lasso
Coefficients:
[1487.78153216 3784.29067093  237.87201826  407.27009612]
Intercept: -2,994,567.65
Mean squared error: 95,599,902.58
Mean absolute error: 6,871.82
R2 Score: 0.44

#### Applying Ridge with alpha=0.10
Coefficients:
[1487.78662594 3784.33990351  237.87455949  407.33792712]
Intercept: -2,994,578.21
Mean squared error: 95,599,902.57
Mean absolute error: 6,871.82
R2 Score: 0.44


#### With Intercept: 0, Plain Linear Regression is the best option
#### Here is the Box Plot for that model
<img src="/images/Box_Plot_0.png"/>

#### Something unsusal appears when we plot the Test Prices vs Predicted Prices
##### The Price Predictions in orange are much shorter than the Test Prices.
<img src="/images/Plain_Predict_Plot_1.png"/>

##### But there NO NEGATIVE price predctions!
<img src="/images/Plain_Predict_Plot_0.png"/>


#### Applying Polynomial Feature Selection with Degree = 4
Intercept: -7,498,424,291.24
Mean squared error: 64,791,763.31
Mean absolute error: 5,413.84
R2 Score: 0.62

#### Applying Ridge with alpha=0.10
Intercept: -7,039,740,390.37
Mean squared error: 64,871,764.62
Mean absolute error: 5,419.70
R2 Score: 0.62

#### Applying Ridge with alpha=1.0
Intercept: -6,930,278,059.67
Mean squared error: 64,917,351.23
Mean absolute error: 5,421.55
R2 Score: 0.62

#### Applying Ridge with alpha=10.0
Intercept: -6,093,683,908.14
Mean squared error: 65,237,594.85
Mean absolute error: 5,444.64
R2 Score: 0.62

#### Building a DataFrame to organize our Model Predictions we can produce a Box Plot
<img src="images/Box_Plot_4.png"/>

#### When we run Simple Cross Validation, the results indicate that the Polynomial Degree 4 is the best option.
#### The Complexity that Minimized Test Error is also 4.
<img src="/images/Degree_Complexity.png">

#### However, something unsusal appears when we plot the Test Prices vs Predicted Prices for Degree 2.
##### The Price Predictions in orange are a little bit shorter than the Test Prices, but better than the predictions from Plain Linear Regression.
<img src="images/Poly_Predict_Plot_1.png">


##### However, there NO NEGATIVE price predctions!
<img src="/images/Poly_Predict_Plot_0.png"/>

### Evaluation
#### Standard Feature selection produces
#####  best R2: 0.12
#####  worst Mean squared error: 149,420,617.16
#####  worst Mean absolute error: 9,153.10

### Insights
##### The only insights I can think of is to reference the experts.
##### Car Dealership Guy on Twitter has his finger on the pulse of car sales.

#### In his latest tweets, 
##### Used car prices are declining 
<img src="images/CarDealershipGuy_Down.png" width="50%"/>

##### Ford has the least trusted branch
<img src="images/CarDealershipGuy_Ford.png" width="50%"/>

##### Chevy is offering BIG incentives to move trucks
<img src="images/CarDealershipGuy_Chevy.png" width="50%"/>

##### Telsa is planning to start selling cars on Carsdotcom.
<img src="images/CarDealershipGuy_Tesla.png" width="50%"/>

##### Hyundai is planning to start selling cars on Amazon.
<img src="images/CarDealershipGuy_Hyundai.png" width="50%"/>

##### Hybrids are out selling EVs
<img src="images/CarDealershipGuy_Hybrids.png" width="50%"/>

##### And the worst news for last...
#### Interest Rates on Inventory Floorplans are through the roof!
<img src="images/CarDealershipGuy_IFP.png" width="50%"/>


### Next steps and recommendations
#### I recommend a cautious approach to build your inventory.
#### Your sales people can tell you what shoppers are saying and how they feel about what's for sale.
#### Find tools that help identify vehicles that suit your local demographics.
#### Price those vehicles strategically to get the most profit.
#### Leverage your network of financers to get the reasonable interest rates to maximize your cash floor.

## Link to notebook
### https://github.com/jiml-mlai-bootcamp-ucberkeley/Practical-Application-Assignment_11_1