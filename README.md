# Practical-Application-Assignment_11_1

## What Drives the Price of a Car?

## Context

## Business Understanding

## Data Understanding

### Data Descriptions

DataFrame.info()
RangeIndex: 426880 entries, 0 to 426879
Data columns (total 18 columns):
 #   Column        Non-Null Count   Dtype  
---  ------        --------------   -----  
 0   id            426880 non-null  int64  
 1   region        426880 non-null  object 
 2   price         426880 non-null  int64  
 3   year          425675 non-null  float64
 4   manufacturer  409234 non-null  object 
 5   model         421603 non-null  object 
 6   condition     252776 non-null  object 
 7   cylinders     249202 non-null  object 
 8   fuel          423867 non-null  object 
 9   odometer      422480 non-null  float64
 10  title_status  418638 non-null  object 
 11  transmission  424324 non-null  object 
 12  VIN           265838 non-null  object 
 13  drive         296313 non-null  object 
 14  size          120519 non-null  object 
 15  type          334022 non-null  object 
 16  paint_color   296677 non-null  object 
 17  state         426880 non-null  object 
dtypes: float64(2), int64(2), object(14)
memory usage: 58.6+ MB

DataFrame.describe()
#	id	price	year	odometer
count	4.268800e+05	4.268800e+05	425675.000000	4.224800e+05
mean	7.311487e+09	7.519903e+04	2011.235191	9.804333e+04
std	4.473170e+06	1.218228e+07	9.452120	2.138815e+05
min	7.207408e+09	0.000000e+00	1900.000000	0.000000e+00
25%	7.308143e+09	5.900000e+03	2008.000000	3.770400e+04
50%	7.312621e+09	1.395000e+04	2013.000000	8.554800e+04
75%	7.315254e+09	2.648575e+04	2017.000000	1.335425e+05
max	7.317101e+09	3.736929e+09	2022.000000	1.000000e+07

DataFrame.isnull().sum()
id                   0
region               0
price                0
year              1205
manufacturer     17646
model             5277
condition       174104
cylinders       177678
fuel              3013
odometer          4400
title_status      8242
transmission      2556
VIN             161042
drive           130567
size            306361
type             92858
paint_color     130203
state                0

# Feature         Unique Values  
--------------    --------------
  id              426,880
  region          404
  price           15,655
  year            114
  manufacturer    42
  model           29649
  condition       6
  cylinders       8
  fuel            5
  odometer        104,870
  title_status    6
  transmission    3
  VIN             118,246
  drive           3
  size            4
  type            13
  paint_color     12
  state           51
  model_year      

### Data Preparation
#### "id" and "VIN" have no value in predicting vehicle prices.  These will be discarded.
#### "model" has too many values to be useful, so it will be discarded.
#### "year" has 114 unique values that could be trimmed.
##### 2022 has 1, so we can discard that row.
##### The difference between 1995(62) and 1996(97) is 35 which is significant.
##### I will discard all the rows before 1996.
##### $0 prices shouldn't be considered an actual sale.  Those will be discarded.
##### $1 is typically a nominal amount that is 'paid' and not a real value.  Those will be discarded.

#### This is what we end up with going forward.
Int64Index: 31048 entries, 215 to 426833
Data columns (total 18 columns):
 #   Column        Non-Null Count  Dtype  
---  ------        --------------  -----  
 0   id            31048 non-null  int64  
 1   region        31048 non-null  object 
 2   price         31048 non-null  int64  
 3   manufacturer  31048 non-null  object 
 4   model         31048 non-null  object 
 5   condition     31048 non-null  object 
 6   cylinders     31048 non-null  object 
 7   fuel          31048 non-null  object 
 8   odometer      31048 non-null  float64
 9   title_status  31048 non-null  object 
 10  transmission  31048 non-null  object 
 11  VIN           31048 non-null  object 
 12  drive         31048 non-null  object 
 13  size          31048 non-null  object 
 14  type          31048 non-null  object 
 15  paint_color   31048 non-null  object 
 16  state         31048 non-null  object 
 17  model_year    31048 non-null  int64  
dtypes: float64(1), int64(3), object(14)
memory usage: 4.5+ MB

#### Here are some Pie Charts that show proportions in the features selected.
<img src="images/condition_pie_chart.png"/>
<img src="images/cylinders_pie_chart.png"/>
<img src="images/drive_pie_chart.png"/>
<img src="images/fuel_pie_chart.png"/>
<img src="images/manufacturer_pie_chart.png"/>
<img src="images/size_pie_chart.png"/>
<img src="images/title_pie_chart.png"/>
<img src="images/transmission_pie_chart.png"/>
<img src="images/type_pie_chart.png"/>


#### Here are some Kernel Density Estimate plots visualizing the distribution of observations for the features selected.

<img src="images/condition_kdeplot.png"/>
<img src="images/cylinders_kdeplot.png"/>
<img src="images/drive_kdeplot.png"/>
<img src="images/size_kdeplot.png"/>
<img src="images/type_kdeplot.png"/>



#### Here are some Heatmaps visualizing the magnitude of individual values.
##### We can see that "odometer" has a negative correlation with price (-0.4) and model year (-0.47).
##### This feature will be dropped.
<img src="images/heatmap.png"/>

##### We can see that "drive" has a negative correlation with price (-0.18) and model year (-0.12).
##### This feature will be dropped.
<img src="images/drop_odometer_heatmap.png"/>

##### We can see that "size" has a negative correlation with price (-0.088) and model year (-0.075).
##### This feature will be dropped.
<img src="images/drop_drive_heatmap.png"/>

##### Here is the Heat Map that will drive the selection of features for our modeling.
<img src="images/drop_size_heatmap.png"/>


<class 'pandas.core.frame.DataFrame'>
Int64Index: 31048 entries, 215 to 426833
Data columns (total 5 columns):

 #   Column      Non-Null Count  Dtype
---  ------      --------------  -----
 0   cylinders   31048 non-null  int32
 1   model_year  31048 non-null  int64
 2   condition   31048 non-null  int32
 3   type        31048 non-null  int32
 4   price       31048 non-null  int64
dtypes: int32(3), int64(2)
memory usage: 2.1 MB


### Modeling
#### Using "train_test_split" on the data we derived from the Heatmaps, we end up with 
21,733 records for training
 9,314 records for testing

#### Applying Linear Regression we get the following:
Coefficients: 
[5.75105537e-01 3.00495003e+03 3.26941663e+02 5.24749893e+02]
Intercept: 0.00
Mean squared error: 149,420,617.16

#### Applying Lasso we get 
Coefficients:
[1487.78153216 3784.29067093  237.87201826  407.27009612]
Intercept: -2,994,567.65
Mean squared error: 95,599,902.58

#### Applying Ridge with alpha=0.10 we get
Coefficients:
[1487.78662594 3784.33990351  237.87455949  407.33792712]
Intercept: -2,994,578.21
Mean squared error: 95,599,902.57


#### With Intercept: 0, Plain Linear Regression is the best option
#### Here is the Box Plot for that model
<img src="/images/Box_Plot_0.png"/>

#### Something unsusal appears when we plot the Test Prices vs Predicted Prices
##### The Price Predictions in orange are much shorter than the Test Prices.
<img src="/images/Plain_Predict_Plot_1.png"/>

##### But there NO NEGATIVE price predctions!
<img src="/images/Plain_Predict_Plot_0.png"/>


#### Applying Polynomial Feature Selection with Degree = 4, we get
Coefficients: 
 [ 7.84112497e+00 -3.47891249e+03 -3.03536397e+03  1.60673866e+03
  1.11457356e+04  5.03186076e+04  2.16585148e+04 -5.55615328e+03
 -4.07424693e+04 -5.51081285e+06  5.75789912e+04 -2.64690203e+06
  8.25930345e+05 -5.54172283e+05 -7.39707440e+00 -4.95691327e+01
 -2.14212708e+01  5.33799476e+00 -3.11326107e+02  5.44451714e+03
  6.32326726e+01  2.58595227e+03 -8.27307704e+02  7.01405698e+02
  5.81592190e+04  2.22857359e+03 -9.76445352e+03  4.64762071e+03
  7.13896715e+03 -5.49348521e+04  5.69904184e+03 -5.50184519e+03
  1.69245350e+04 -3.76596390e+04  1.38075353e-03  1.22103341e-02
  5.29837696e-03 -1.28128807e-03  1.59971021e-01 -1.34525532e+00
 -4.60414812e-02 -6.32644893e-01  2.07006960e-01 -2.10044891e-01
 -2.67121434e+01 -8.84727138e-01  4.80418544e+00 -2.19400974e+00
 -3.48749617e+00  2.75566433e+01 -2.51340110e+00  2.74329659e+00
 -8.27870220e+00  1.76352853e+01 -3.08894307e+02 -2.22549549e+01
  1.23044101e+01 -1.64898783e+01 -1.21519814e+01  4.80662174e+01
 -1.13594887e+01  3.74660819e+00 -3.08028453e+01 -8.91411890e+01
 -2.49632710e+01  1.85266296e+00 -2.02722559e+01 -2.13836314e-01
  2.08180606e+02]
Intercept: -7,498,424,291.24
Mean squared error: 64,791,763.31

#### Applying Ridge with alpha=0.10 we get
Coefficients: 
 [ 6.51357465e+00  4.67108686e+01 -1.05251855e+00 -5.80376037e+00
  1.05076570e+04  3.24926611e+04 -4.32201463e+03 -4.00889638e+03
  5.22314485e+01 -2.96850320e+03 -8.35169621e+02 -3.12539800e+03
 -9.54107553e+01  3.83646709e+02 -6.98809043e+00 -3.18448332e+01
  4.41258779e+00  3.86883340e+00 -3.48206487e+02 -2.73122190e+01
  1.00091845e+02 -4.38672590e+01 -5.81194632e+00  8.44615415e+01
  5.74526388e+04  1.87338578e+03 -9.20307840e+03  3.44239905e+03
  8.01897986e+03 -4.23072392e+04  5.93455051e+03 -5.17186250e+03
  1.43268156e+04 -2.40877478e+04  1.30711386e-03  7.80418259e-03
 -1.12393670e-03 -9.33064736e-04  1.68169761e-01  1.37170580e-02
 -4.99785107e-02  2.13337791e-02  2.72340288e-03 -4.03949088e-02
 -2.63412002e+01 -6.99094387e-01  4.53188889e+00 -1.59087328e+00
 -3.91524041e+00  2.13143021e+01 -2.63126258e+00  2.58494500e+00
 -6.96472035e+00  1.08491994e+01 -3.11508270e+02 -2.25178810e+01
  1.25085955e+01 -1.78189402e+01 -1.24146785e+01  4.39825671e+01
 -1.11649234e+01  2.92466666e+00 -3.49579564e+01 -9.02522101e+01
 -2.49164166e+01  1.70915263e+00 -2.28687149e+01  7.09439280e-02
  2.16019852e+02]
Intercept: -7,039,740,390.37
Mean squared error: 64,871,764.62

#### Applying Ridge with alpha=1.0 we get
Coefficients: 
 [ 7.63802923e+00  4.73053499e+01 -6.05381691e+00 -5.90906640e+00
  1.03461894e+04  3.18297331e+04 -4.38743062e+03 -3.96659061e+03
  1.27003160e+02 -2.30503515e+02 -7.09210323e+01 -3.03950945e+02
 -3.38887600e+01  3.08460091e+01 -6.88128968e+00 -3.13022511e+01
  4.47027858e+00  3.94338036e+00 -2.78404731e+02 -2.10013015e+01
  4.96132152e+01 -4.74062095e+01 -9.83496394e-01 -2.05727042e+01
  4.48982056e+04 -2.29777294e+02 -5.64524069e+03  3.71504628e+03
  7.54184665e+03 -1.44253464e+04  5.89181760e+03 -4.88301146e+03
  1.01160819e+04 -6.27474941e+03  1.28723894e-03  7.69838961e-03
 -1.13640420e-03 -9.80508258e-04  1.33395439e-01  9.86164610e-03
 -2.51651373e-02  2.23832494e-02  3.16401751e-04  1.18651124e-02
 -2.00887063e+01  3.60077420e-01  2.78626567e+00 -1.72095014e+00
 -3.68058283e+00  7.53376425e+00 -2.60827994e+00  2.44087205e+00
 -4.88034707e+00  1.96844230e+00 -3.12118585e+02 -2.40559058e+01
  1.13410545e+01 -1.86046885e+01 -1.23598155e+01  3.01929305e+01
 -1.13949667e+01  2.89779856e+00 -3.29842829e+01 -9.81848741e+01
 -2.50152216e+01  1.76655638e+00 -2.27739904e+01  1.73470872e+00
  2.24467564e+02]
Intercept: -6,930,278,059.67
Mean squared error: 64,917,351.23

#### Applying Ridge with alpha=10.0 we get
Coefficients: 
 [ 6.81210612e+00  2.88028138e+01 -5.69045227e+00 -2.92367106e+00
  9.12912805e+03  1.93163809e+04 -3.82810572e+03 -1.95712811e+03
  7.95631554e+01  1.74009182e+01 -1.82693002e+00 -3.91669631e+01
 -1.19944070e+01  1.00969764e+00 -6.08224636e+00 -1.91095933e+01
  3.88715932e+00  1.95343114e+00 -1.17739143e+02 -4.17431846e+00
  2.79374602e+01 -4.42714250e+01  9.13323192e+00 -4.08663078e+01
  1.46089910e+04 -2.25458221e+03 -1.07377801e+03  2.89027795e+03
  3.81745524e+03 -2.63444042e+03  5.53370042e+03 -3.51147505e+03
  3.04683518e+03 -7.83846207e+02  1.13969592e-03  4.72983432e-03
 -9.84659222e-04 -4.87513455e-04  5.34095673e-02  1.35210067e-03
 -1.47196214e-02  2.07353641e-02 -4.70670406e-03  2.16315335e-02
 -5.00450323e+00  1.39380122e+00  5.90898572e-01 -1.30012952e+00
 -1.82757844e+00  1.91552675e+00 -2.42717255e+00  1.75680753e+00
 -1.37647346e+00 -7.16130530e-01 -3.13605712e+02 -2.69895061e+01
  8.32102418e+00 -2.03683681e+01 -1.35936960e+01 -1.71849000e+01
 -1.17396851e+01  3.08535248e+00 -3.07895306e+01 -1.10104927e+02
 -2.51854513e+01  1.80669947e+00 -2.16692450e+01  1.37156606e+00
  2.22953555e+02]
Intercept: -6,093,683,908.14
Mean squared error: 65,237,594.85

#### A Grid Search produces the following
itting 4 folds for each of 4 candidates, totalling 16 fits
********** Linear Regression Train Score: 0.6203823296833513 **********
{'mean_fit_time': array([0.01246637, 0.01262593, 0.01762003, 0.0309791 ]),
 'std_fit_time': array([0.00241989, 0.00355831, 0.0015485 , 0.00775756]),
 'mean_score_time': array([0.00316393, 0.00389838, 0.0018391 , 0.00534457]),
 'std_score_time': array([0.00220175, 0.00281682, 0.00184225, 0.00156117]),
 'param_polynomial_features__degree': masked_array(data=[1, 2, 3, 4],
              mask=[False, False, False, False],
        fill_value='?',
             dtype=object),
 'params': [{'polynomial_features__degree': 1},
  {'polynomial_features__degree': 2},
  {'polynomial_features__degree': 3},
  {'polynomial_features__degree': 4}],
 'split0_test_score': array([0.46032978, 0.52816713, 0.6034722 , 0.63130387]),
 'split1_test_score': array([0.43426475, 0.50795397, 0.58431857, 0.61644193]),
 'split2_test_score': array([0.43156495, 0.50296471, 0.57964632, 0.60512888]),
 'split3_test_score': array([0.43570193, 0.50304881, 0.58374892, 0.61389708]),
 'mean_test_score': array([0.44046535, 0.51053365, 0.5877965 , 0.61669294]),
 'std_test_score': array([0.0115645 , 0.01037914, 0.00922811, 0.00942187]),
 'rank_test_score': array([4, 3, 2, 1])}

#### Building a DataFrame to organize our Model Predictions we can produce a Box Plot
<img src="images/Box_Plot_4.png"/>

#### When we run Simple Cross Validation, the results indicate that the Polynomial Degree 4 is the best option.
{'memory': None,
 'steps': [('poly_features', PolynomialFeatures(degree=4, include_bias=False)),
  ('lin_reg', LinearRegression())],
 'verbose': False,
 'poly_features': PolynomialFeatures(degree=4, include_bias=False),
 'lin_reg': LinearRegression(),
 'poly_features__degree': 4,
 'poly_features__include_bias': False,
 'poly_features__interaction_only': False,
 'poly_features__order': 'C',
 'lin_reg__copy_X': True,
 'lin_reg__fit_intercept': True,
 'lin_reg__n_jobs': None,
 'lin_reg__positive': False}


#### The Complexity that Minimized Test Error is also 4.
<img src="/images/Degree_Complexity.png">

#### However, something unsusal appears when we plot the Test Prices vs Predicted Prices for Degree 2.
##### The Price Predictions in orange are a little bit shorter than the Test Prices, but better than the predictions from Plain Linear Regression.
<img src="images/Poly_Predict_Plot_1.png">


##### However, there NO NEGATIVE price predctions!
<img src="/images/Poly_Predict_Plot_0.png"/>

### Evaluation
#### 

### Deployment
#### 


## Link to notebook
### https://github.com/jiml-mlai-bootcamp-ucberkeley/Practical-Application-Assignment_11_1