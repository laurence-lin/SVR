# SVR
support vector regression practice


2018.07.16 update

Test SVR regressioin on time series energy dataset on UCI

Time step = 5, adjust to 7

Use 7 days of energy and weather data to predict 8th day's energy consume in a low energy house.

Result:

Final argument: C = 100, time step = 7

![img](https://github.com/laurence-lin/SVR/blob/master/SVR_time_series_forecasting.png)

Discussion:

We could set penalty C be large to 100, and SVR could still get a proper result on test set.
First use linear kernel, and found performance is poor, then adjust to RBF kernel.

SVR is availabel for time series forecasting.
