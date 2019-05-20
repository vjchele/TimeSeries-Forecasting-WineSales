
## Timeseries Forecasting Model Monthly Sales of French Champagne 
#### The problem is to predict the number of monthly sales of champagne for the Perrin Freres label (named for a region in France).

#### The dataset provides the number of monthly sales of champagne from January 1964 to September 1972, or just under 10 years of data.

#### The values are a count of millions of sales and there are 105 observations.

#### The dataset is credited to Makridakis and Wheelwright, 1989.

#### The following steps will taken to accomplish the project
#### 1.Load and explore the dataset.
#### 2.Visualize the dataset.
#### 3.Develop a persistence model.
#### 4.Develop an autoregressive model.
#### 5.Develop an ARIMA model.
#### 6.Visualize forecasts and summarize forecast error.


```python
import pandas as pd
from pandas import Series
from pandas import TimeGrouper
from pandas.plotting import autocorrelation_plot
from pandas import read_csv
from pandas import datetime
from pandas import DataFrame
from pandas import concat
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from pandas.plotting import lag_plot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARIMAResults
import math
import numpy
```

### Load Dataset (champagne.csv)
### Storing the last 12 records for validation


```python
#Load the Dataset
df = pd.read_csv('champagne.csv',header=0, parse_dates=[0],index_col=0, squeeze=True)
validation = df[df.size-13:]
validation.to_csv('champagne-validation.csv')
df = df[0:df.size-13]

```


```python
# Let's take a peek at the data
df.head()
#df.tail()
```




    Month
    1964-01-01    2815
    1964-02-01    2672
    1964-03-01    2755
    1964-04-01    2721
    1964-05-01    2946
    Name: Sales, dtype: int64




```python
#Histogram of the data
df.hist(bins=10)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1d2d8416ac8>




![png](Wine%20Sales%20Forecasting_files/Wine%20Sales%20Forecasting_5_1.png)



```python
#describing the data
df.describe()
# There are 105 records
# min sales is 1413
# max sales is 13916
```




    count       92.000000
    mean      4626.880435
    std       2496.213250
    min       1573.000000
    25%       3034.750000
    50%       4001.000000
    75%       5019.500000
    max      13916.000000
    Name: Sales, dtype: float64




```python
#Number of Observations
df.size
```




    92




```python
#Extracting Sales Data for the month of January 1950
df['1964']

```




    Month
    1964-01-01    2815
    1964-02-01    2672
    1964-03-01    2755
    1964-04-01    2721
    1964-05-01    2946
    1964-06-01    3036
    1964-07-01    2282
    1964-08-01    2212
    1964-09-01    2922
    1964-10-01    4301
    1964-11-01    5764
    1964-12-01    7312
    Name: Sales, dtype: int64



# Visualizing Data


```python
#timeseries plot
df.plot(figsize=(15,6))
plt.show()

```


![png](Wine%20Sales%20Forecasting_files/Wine%20Sales%20Forecasting_10_0.png)



```python
# Dot plot of the data
df.plot(style='k.')
plt.show()
```


![png](Wine%20Sales%20Forecasting_files/Wine%20Sales%20Forecasting_11_0.png)



```python
df.resample('Y').plot()
plt.show()

```


![png](Wine%20Sales%20Forecasting_files/Wine%20Sales%20Forecasting_12_0.png)



```python
#Denisty plot
df.plot(kind='kde')
plt.show()
```


![png](Wine%20Sales%20Forecasting_files/Wine%20Sales%20Forecasting_13_0.png)


# Baseline Forecasting Model
#### Baseline in forecast provides a point of comparison. Persistence Algorithm is a common Baseline algorithm.
#### We will use the following steps to perform baseline
#### 1.Transform the univariate dataset into a supervised learning problem.
#### 2. Establish the train and test datasets for the test harness.
#### 3. Define the persistence model.
#### 4. Make a forecast and establish a baseline performance.
#### 5. Review the complete example and plot the output.


```python
values = pd.DataFrame(df.values)
lagged_ds = pd.concat([values.shift(1),values],axis=1)
lagged_ds.columns = ['t-1','t']
lagged_ds
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>t-1</th>
      <th>t</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>2815</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2815.0</td>
      <td>2672</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2672.0</td>
      <td>2755</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2755.0</td>
      <td>2721</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2721.0</td>
      <td>2946</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2946.0</td>
      <td>3036</td>
    </tr>
    <tr>
      <th>6</th>
      <td>3036.0</td>
      <td>2282</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2282.0</td>
      <td>2212</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2212.0</td>
      <td>2922</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2922.0</td>
      <td>4301</td>
    </tr>
    <tr>
      <th>10</th>
      <td>4301.0</td>
      <td>5764</td>
    </tr>
    <tr>
      <th>11</th>
      <td>5764.0</td>
      <td>7312</td>
    </tr>
    <tr>
      <th>12</th>
      <td>7312.0</td>
      <td>2541</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2541.0</td>
      <td>2475</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2475.0</td>
      <td>3031</td>
    </tr>
    <tr>
      <th>15</th>
      <td>3031.0</td>
      <td>3266</td>
    </tr>
    <tr>
      <th>16</th>
      <td>3266.0</td>
      <td>3776</td>
    </tr>
    <tr>
      <th>17</th>
      <td>3776.0</td>
      <td>3230</td>
    </tr>
    <tr>
      <th>18</th>
      <td>3230.0</td>
      <td>3028</td>
    </tr>
    <tr>
      <th>19</th>
      <td>3028.0</td>
      <td>1759</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1759.0</td>
      <td>3595</td>
    </tr>
    <tr>
      <th>21</th>
      <td>3595.0</td>
      <td>4474</td>
    </tr>
    <tr>
      <th>22</th>
      <td>4474.0</td>
      <td>6838</td>
    </tr>
    <tr>
      <th>23</th>
      <td>6838.0</td>
      <td>8357</td>
    </tr>
    <tr>
      <th>24</th>
      <td>8357.0</td>
      <td>3113</td>
    </tr>
    <tr>
      <th>25</th>
      <td>3113.0</td>
      <td>3006</td>
    </tr>
    <tr>
      <th>26</th>
      <td>3006.0</td>
      <td>4047</td>
    </tr>
    <tr>
      <th>27</th>
      <td>4047.0</td>
      <td>3523</td>
    </tr>
    <tr>
      <th>28</th>
      <td>3523.0</td>
      <td>3937</td>
    </tr>
    <tr>
      <th>29</th>
      <td>3937.0</td>
      <td>3986</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>62</th>
      <td>3957.0</td>
      <td>4510</td>
    </tr>
    <tr>
      <th>63</th>
      <td>4510.0</td>
      <td>4276</td>
    </tr>
    <tr>
      <th>64</th>
      <td>4276.0</td>
      <td>4968</td>
    </tr>
    <tr>
      <th>65</th>
      <td>4968.0</td>
      <td>4677</td>
    </tr>
    <tr>
      <th>66</th>
      <td>4677.0</td>
      <td>3523</td>
    </tr>
    <tr>
      <th>67</th>
      <td>3523.0</td>
      <td>1821</td>
    </tr>
    <tr>
      <th>68</th>
      <td>1821.0</td>
      <td>5222</td>
    </tr>
    <tr>
      <th>69</th>
      <td>5222.0</td>
      <td>6872</td>
    </tr>
    <tr>
      <th>70</th>
      <td>6872.0</td>
      <td>10803</td>
    </tr>
    <tr>
      <th>71</th>
      <td>10803.0</td>
      <td>13916</td>
    </tr>
    <tr>
      <th>72</th>
      <td>13916.0</td>
      <td>2639</td>
    </tr>
    <tr>
      <th>73</th>
      <td>2639.0</td>
      <td>2899</td>
    </tr>
    <tr>
      <th>74</th>
      <td>2899.0</td>
      <td>3370</td>
    </tr>
    <tr>
      <th>75</th>
      <td>3370.0</td>
      <td>3740</td>
    </tr>
    <tr>
      <th>76</th>
      <td>3740.0</td>
      <td>2927</td>
    </tr>
    <tr>
      <th>77</th>
      <td>2927.0</td>
      <td>3986</td>
    </tr>
    <tr>
      <th>78</th>
      <td>3986.0</td>
      <td>4217</td>
    </tr>
    <tr>
      <th>79</th>
      <td>4217.0</td>
      <td>1738</td>
    </tr>
    <tr>
      <th>80</th>
      <td>1738.0</td>
      <td>5221</td>
    </tr>
    <tr>
      <th>81</th>
      <td>5221.0</td>
      <td>6424</td>
    </tr>
    <tr>
      <th>82</th>
      <td>6424.0</td>
      <td>9842</td>
    </tr>
    <tr>
      <th>83</th>
      <td>9842.0</td>
      <td>13076</td>
    </tr>
    <tr>
      <th>84</th>
      <td>13076.0</td>
      <td>3934</td>
    </tr>
    <tr>
      <th>85</th>
      <td>3934.0</td>
      <td>3162</td>
    </tr>
    <tr>
      <th>86</th>
      <td>3162.0</td>
      <td>4286</td>
    </tr>
    <tr>
      <th>87</th>
      <td>4286.0</td>
      <td>4676</td>
    </tr>
    <tr>
      <th>88</th>
      <td>4676.0</td>
      <td>5010</td>
    </tr>
    <tr>
      <th>89</th>
      <td>5010.0</td>
      <td>4874</td>
    </tr>
    <tr>
      <th>90</th>
      <td>4874.0</td>
      <td>4633</td>
    </tr>
    <tr>
      <th>91</th>
      <td>4633.0</td>
      <td>1659</td>
    </tr>
  </tbody>
</table>
<p>92 rows × 2 columns</p>
</div>



### Train and Test Dataset


```python
# split into train and test sets
X = lagged_ds.values
train_size = int(len(X) * 0.66)
train, test = X[1:train_size], X[train_size:]
train_X, train_y = train[:,0], train[:,1]
test_X, test_y = test[:,0], test[:,1]
print(test_X)
print(test_y)
```

    [11331.  4016.  3957.  4510.  4276.  4968.  4677.  3523.  1821.  5222.
      6872. 10803. 13916.  2639.  2899.  3370.  3740.  2927.  3986.  4217.
      1738.  5221.  6424.  9842. 13076.  3934.  3162.  4286.  4676.  5010.
      4874.  4633.]
    [ 4016.  3957.  4510.  4276.  4968.  4677.  3523.  1821.  5222.  6872.
     10803. 13916.  2639.  2899.  3370.  3740.  2927.  3986.  4217.  1738.
      5221.  6424.  9842. 13076.  3934.  3162.  4286.  4676.  5010.  4874.
      4633.  1659.]
    

## Persistence Algorithm


```python
# persistence model
def model_persistence(x):
    return x
# walk-forward validation
predictions = list()
for x in test_X:
    yhat = model_persistence(x)
    predictions.append(yhat)
test_score = mean_squared_error(test_y, predictions)
rmse = math.sqrt(test_score)
print('RMSE: %.3f' % rmse)
```

    RMSE: 3372.763
    


```python
# prepare data
X = df.values
X = X.astype('float32')
train_size = int(len(X) * 0.50)
train, test = X[0:train_size], X[train_size:]
# walk-forward validation
history = [x for x in train]
predictions = list()
for i in range(len(test)):
    # predict
    yhat = history[-1]
    predictions.append(yhat)
    # observation
    obs = test[i]
    history.append(obs)
    print('>Predicted=%.3f, Expected=%3.f' % (yhat, obs))
# report performance
mse = mean_squared_error(test, predictions)
rmse = math.sqrt(mse)
print('RMSE: %.3f' % rmse)
```

    >Predicted=5428.000, Expected=8314
    >Predicted=8314.000, Expected=10651
    >Predicted=10651.000, Expected=3633
    >Predicted=3633.000, Expected=4292
    >Predicted=4292.000, Expected=4154
    >Predicted=4154.000, Expected=4121
    >Predicted=4121.000, Expected=4647
    >Predicted=4647.000, Expected=4753
    >Predicted=4753.000, Expected=3965
    >Predicted=3965.000, Expected=1723
    >Predicted=1723.000, Expected=5048
    >Predicted=5048.000, Expected=6922
    >Predicted=6922.000, Expected=9858
    >Predicted=9858.000, Expected=11331
    >Predicted=11331.000, Expected=4016
    >Predicted=4016.000, Expected=3957
    >Predicted=3957.000, Expected=4510
    >Predicted=4510.000, Expected=4276
    >Predicted=4276.000, Expected=4968
    >Predicted=4968.000, Expected=4677
    >Predicted=4677.000, Expected=3523
    >Predicted=3523.000, Expected=1821
    >Predicted=1821.000, Expected=5222
    >Predicted=5222.000, Expected=6872
    >Predicted=6872.000, Expected=10803
    >Predicted=10803.000, Expected=13916
    >Predicted=13916.000, Expected=2639
    >Predicted=2639.000, Expected=2899
    >Predicted=2899.000, Expected=3370
    >Predicted=3370.000, Expected=3740
    >Predicted=3740.000, Expected=2927
    >Predicted=2927.000, Expected=3986
    >Predicted=3986.000, Expected=4217
    >Predicted=4217.000, Expected=1738
    >Predicted=1738.000, Expected=5221
    >Predicted=5221.000, Expected=6424
    >Predicted=6424.000, Expected=9842
    >Predicted=9842.000, Expected=13076
    >Predicted=13076.000, Expected=3934
    >Predicted=3934.000, Expected=3162
    >Predicted=3162.000, Expected=4286
    >Predicted=4286.000, Expected=4676
    >Predicted=4676.000, Expected=5010
    >Predicted=5010.000, Expected=4874
    >Predicted=4874.000, Expected=4633
    >Predicted=4633.000, Expected=1659
    RMSE: 3158.174
    


```python
# Fit regression model
train_X = train_X.reshape(-1,1)
train_y = train_y.reshape(-1,1)
test_X = test_X.reshape(-1,1)

regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor(max_depth=5)
regr_1.fit(train_X, train_y)
regr_2.fit(train_X, train_y)
y_1 = regr_1.predict(test_X)
y_2 = regr_2.predict(test_X)
test_score = mean_squared_error(y_1, test_y)
print('Test Mean Squared Error with max_depth 2: %.3f' % test_score)
test_score = mean_squared_error(y_2, test_y)
print('Test Mean Squared Error with max_depth 5: %.3f' % test_score)


```

    Test Mean Squared Error with max_depth 2: 6523797.735
    Test Mean Squared Error with max_depth 5: 5744130.467
    

## Plot the Baseline Prediction


```python
plt.plot(train_y)
plt.plot([None for i in train_y] + [x for x in test_y])

plt.plot([None for i in train_y] + [x for x in predictions])
plt.plot([None for i in train_y] + [x for x in y_2])

plt.show()
```


![png](Wine%20Sales%20Forecasting_files/Wine%20Sales%20Forecasting_23_0.png)


# Autocorrelation of Data
## We can check to see if there is an autocorrelation in the data


```python
lag_plot(df)
plt.show()
# We see strong correlation only in the left and then deteriorate
```


![png](Wine%20Sales%20Forecasting_files/Wine%20Sales%20Forecasting_25_0.png)



```python
autocorrelation_plot(df)
plt.show()
```


![png](Wine%20Sales%20Forecasting_files/Wine%20Sales%20Forecasting_26_0.png)



```python
plot_acf(df)
plt.show()
# Autocorrelation plot shows a significant lag for 1 month.
```


![png](Wine%20Sales%20Forecasting_files/Wine%20Sales%20Forecasting_27_0.png)



```python
plot_pacf(df,lags=31)
plt.show()
# Partial Autocorrelation plot shows a significant lag for 1 month.
```

    C:\ProgramData\Anaconda3\lib\site-packages\statsmodels\regression\linear_model.py:1283: RuntimeWarning: invalid value encountered in sqrt
      return rho, np.sqrt(sigmasq)
    


![png](Wine%20Sales%20Forecasting_files/Wine%20Sales%20Forecasting_28_1.png)


# ARIMA Forecasting


```python
model = ARIMA(df, order=(5,1,0))
model_fit = model.fit(disp=0)
print(model_fit.summary())
# plot residual errors
residuals = DataFrame(model_fit.resid)
residuals.plot()
plt.show()
residuals.plot(kind='kde')
plt.show()
print(residuals.describe())
```

    C:\ProgramData\Anaconda3\lib\site-packages\statsmodels\tsa\base\tsa_model.py:171: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
      % freq, ValueWarning)
    C:\ProgramData\Anaconda3\lib\site-packages\statsmodels\tsa\base\tsa_model.py:171: ValueWarning: No frequency information was provided, so inferred frequency MS will be used.
      % freq, ValueWarning)
    

                                 ARIMA Model Results                              
    ==============================================================================
    Dep. Variable:                D.Sales   No. Observations:                   91
    Model:                 ARIMA(5, 1, 0)   Log Likelihood                -829.912
    Method:                       css-mle   S.D. of innovations           2199.099
    Date:                Sun, 19 May 2019   AIC                           1673.825
    Time:                        09:33:18   BIC                           1691.401
    Sample:                    02-01-1964   HQIC                          1680.915
                             - 08-01-1971                                         
    =================================================================================
                        coef    std err          z      P>|z|      [0.025      0.975]
    ---------------------------------------------------------------------------------
    const             6.6713    101.410      0.066      0.948    -192.089     205.432
    ar.L1.D.Sales    -0.2221      0.105     -2.107      0.038      -0.429      -0.016
    ar.L2.D.Sales    -0.3707      0.097     -3.803      0.000      -0.562      -0.180
    ar.L3.D.Sales    -0.2393      0.101     -2.371      0.020      -0.437      -0.042
    ar.L4.D.Sales    -0.4175      0.095     -4.384      0.000      -0.604      -0.231
    ar.L5.D.Sales    -0.0648      0.103     -0.630      0.530      -0.266       0.137
                                        Roots                                    
    =============================================================================
                      Real          Imaginary           Modulus         Frequency
    -----------------------------------------------------------------------------
    AR.1            0.6522           -0.9979j            1.1922           -0.1579
    AR.2            0.6522           +0.9979j            1.1922            0.1579
    AR.3           -0.8830           -1.0176j            1.3472           -0.3637
    AR.4           -0.8830           +1.0176j            1.3472            0.3637
    AR.5           -5.9857           -0.0000j            5.9857           -0.5000
    -----------------------------------------------------------------------------
    


![png](Wine%20Sales%20Forecasting_files/Wine%20Sales%20Forecasting_30_2.png)



![png](Wine%20Sales%20Forecasting_files/Wine%20Sales%20Forecasting_30_3.png)


                     0
    count    91.000000
    mean      0.673735
    std    2211.319407
    min   -7438.843119
    25%   -1122.741709
    50%     -73.546209
    75%    1386.647223
    max    4635.056248
    

# Rolling Forecast ARIMA model


```python
X = df.values
X = X.astype('float32')
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()

# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)
# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]

for i in range(len(test)):
    # difference data
    months_in_year = 12
    diff = difference(history, months_in_year)
    # predict
    #model = ARIMA(diff, order=(1,1,1))
    model = ARIMA(diff, order=(1,0,0))
    #model_fit = model.fit(trend='nc', disp=0)
    model_fit = model.fit(trend='nc',disp=0)
    yhat = model_fit.forecast()[0]
    yhat = inverse_difference(history, yhat, months_in_year)
    predictions.append(yhat)
    # observation
    obs = test[i]
    history.append(obs)
    print('>Predicted=%.3f, Expected=%3.f' % (yhat, obs))
# report performance
mse = mean_squared_error(test, predictions)
rmse = math.sqrt(mse)
print('RMSE: %.3f' % rmse)
plt.plot(test)
plt.plot(predictions, color='red')
plt.show()
```

    >Predicted=3838.223, Expected=4016
    >Predicted=4409.135, Expected=3957
    >Predicted=4053.391, Expected=4510
    >Predicted=4226.190, Expected=4276
    >Predicted=4692.904, Expected=4968
    >Predicted=4848.533, Expected=4677
    >Predicted=3942.508, Expected=3523
    >Predicted=1591.713, Expected=1821
    >Predicted=5076.802, Expected=5222
    >Predicted=6973.232, Expected=6872
    >Predicted=9843.302, Expected=10803
    >Predicted=11607.283, Expected=13916
    >Predicted=4942.988, Expected=2639
    >Predicted=3672.968, Expected=2899
    >Predicted=4264.314, Expected=3370
    >Predicted=3985.743, Expected=3740
    >Predicted=4828.049, Expected=2927
    >Predicted=4097.306, Expected=3986
    >Predicted=3323.429, Expected=4217
    >Predicted=2012.601, Expected=1738
    >Predicted=5199.404, Expected=5221
    >Predicted=6871.728, Expected=6424
    >Predicted=10681.003, Expected=9842
    >Predicted=13646.968, Expected=13076
    >Predicted=2394.624, Expected=3934
    >Predicted=3242.627, Expected=3162
    >Predicted=3439.276, Expected=4286
    >Predicted=3985.158, Expected=4676
    >Predicted=3188.611, Expected=5010
    >Predicted=4633.121, Expected=4874
    >Predicted=4500.502, Expected=4633
    >Predicted=1871.655, Expected=1659
    RMSE: 911.224
    


![png](Wine%20Sales%20Forecasting_files/Wine%20Sales%20Forecasting_32_1.png)



```python
residuals = [test[i]-predictions[i] for i in range(len(test))]
residuals = DataFrame(residuals)
print(residuals.describe())
# plot
plt.figure()
plt.subplot(211)
residuals.hist(ax=plt.gca())
plt.subplot(212)
residuals.plot(kind='kde', ax=plt.gca())
plt.show()
```

                     0
    count    32.000000
    mean     30.897271
    std     925.271816
    min   -2303.987835
    25%    -426.562950
    50%     -29.515087
    75%     320.474358
    max    2308.717116
    


![png](Wine%20Sales%20Forecasting_files/Wine%20Sales%20Forecasting_33_1.png)


# Grid Search for Hyperparameters


```python
# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)
 
# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]
 
# evaluate an ARIMA model for a given order (p,d,q) and return RMSE
def evaluate_arima_model(X, arima_order):
    # prepare training dataset
    X = X.astype('float32')
    train_size = int(len(X) * 0.50)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    # make predictions
    predictions = list()
    for t in range(len(test)):
        # difference data
        months_in_year = 12
        diff = difference(history, months_in_year)
        model = ARIMA(diff, order=arima_order)
        model_fit = model.fit(trend='nc', disp=0)
        yhat = model_fit.forecast()[0]
        yhat = inverse_difference(history, yhat, months_in_year)
        predictions.append(yhat)
        history.append(test[t])
    # calculate out of sample error
    mse = mean_squared_error(test, predictions)
    rmse = math.sqrt(mse)
    return rmse

# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
    dataset = dataset.astype('float32')
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                
                try:
                    mse = evaluate_arima_model(dataset, order)
                    print(mse)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                        print('ARIMA%s RMSE=%.3f' % (order,mse))
                except:
                    
                    continue
    print('Best ARIMA%s RMSE=%.3f' % (best_cfg, best_score))
 

# evaluate parameters
p_values = range(0, 2)
d_values = range(0, 2)
q_values = range(0, 2)
#warnings.filterwarnings("ignore")
evaluate_models(df.values, p_values, d_values, q_values)
```

    942.7436319500628
    ARIMA(0, 0, 1) RMSE=942.744
    967.0680349773364
    948.1422851648852
    

    C:\ProgramData\Anaconda3\lib\site-packages\statsmodels\base\model.py:488: HessianInversionWarning: Inverting hessian failed, no bse or cov_params available
      'available', HessianInversionWarning)
    C:\ProgramData\Anaconda3\lib\site-packages\statsmodels\tsa\kalmanf\kalmanfilter.py:649: RuntimeWarning: divide by zero encountered in true_divide
      R_mat, T_mat)
    C:\ProgramData\Anaconda3\lib\site-packages\statsmodels\tsa\tsatools.py:650: RuntimeWarning: invalid value encountered in true_divide
      newparams = ((1-np.exp(-params))/(1+np.exp(-params))).copy()
    C:\ProgramData\Anaconda3\lib\site-packages\statsmodels\tsa\tsatools.py:651: RuntimeWarning: invalid value encountered in true_divide
      tmp = ((1-np.exp(-params))/(1+np.exp(-params))).copy()
    C:\ProgramData\Anaconda3\lib\site-packages\statsmodels\base\model.py:488: HessianInversionWarning: Inverting hessian failed, no bse or cov_params available
      'available', HessianInversionWarning)
    

    1078.1168761654612
    963.9412356343594
    Best ARIMA(0, 0, 1) RMSE=942.744
    

# Saving the Model


```python
bias = 165.904728
# save model
model_fit.save('model.pkl')
numpy.save('model_bias.npy', [bias])
```

# Making the First prediction using the Validation dataset


```python
validation = pd.read_csv('champagne-validation.csv',header=0, parse_dates=[0],index_col=0, squeeze=True)
y = validation.values.astype('float32')

# load model

model_fit = ARIMAResults.load('model.pkl')
bias = numpy.load('model_bias.npy')
# make first prediction
yhat = float(model_fit.forecast()[0])
yhat = bias + inverse_difference(history, yhat, months_in_year)
predictions = list()
yhat = float(model_fit.forecast()[0])
yhat = bias + inverse_difference(history, yhat, months_in_year)
predictions.append(yhat)
history.append(y[0])
print('>Predicted=%.3f, Expected=%3.f' % (yhat, y[0]))

```

    >Predicted=6176.559, Expected=6981
    


```python
predictions = list()
# rolling forecasts
for i in range(len(y)):
    # difference data
    months_in_year = 12
    diff = difference(history, months_in_year)
    # predict
    model = ARIMA(diff, order=(0,0,1))
    model_fit = model.fit(trend='nc', disp=0)
    yhat = model_fit.forecast()[0]
    yhat = bias + inverse_difference(history, yhat, months_in_year)
    predictions.append(yhat)
    # observation
    obs = y[i]
    history.append(obs)
    print('>Predicted=%.3f, Expected=%3.f' % (yhat, obs))
# report performance

mse = mean_squared_error(y, predictions)
rmse = math.sqrt(mse)
print('RMSE: %.3f' % rmse)
plt.plot(y)
plt.plot(predictions, color='red')
plt.show()
```

    >Predicted=7146.904, Expected=6981
    >Predicted=10016.905, Expected=9851
    >Predicted=12835.905, Expected=12670
    >Predicted=4513.905, Expected=4348
    >Predicted=3729.905, Expected=3564
    >Predicted=4742.905, Expected=4577
    >Predicted=4953.905, Expected=4788
    >Predicted=4783.905, Expected=4618
    >Predicted=5477.905, Expected=5312
    >Predicted=4463.905, Expected=4298
    >Predicted=1578.905, Expected=1413
    >Predicted=6042.905, Expected=5877
    RMSE: 165.905
    


![png](Wine%20Sales%20Forecasting_files/Wine%20Sales%20Forecasting_40_1.png)


