# Ex.No: 08     MOVINTG AVERAGE MODEL AND EXPONENTIAL SMOOTHING
### Date: 14-10-2025


### AIM:
To implement Moving Average Model and Exponential smoothing Using Python.
### ALGORITHM:
1. Import necessary libraries
2. Read the electricity time series data from a CSV file,Display the shape and the first 20 rows of
the dataset
3. Set the figure size for plots
4. Suppress warnings
5. Plot the first 50 values of the 'Value' column
6. Perform rolling average transformation with a window size of 5
7. Display the first 10 values of the rolling mean
8. Perform rolling average transformation with a window size of 10
9. Create a new figure for plotting,Plot the original data and fitted value
10. Show the plot
11. Also perform exponential smoothing and plot the graph
### PROGRAM:
```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error

data = pd.read_csv("Sunspots.csv")
data['Date'] = pd.to_datetime(data['Date'])
data = data.set_index('Date')

sun = data[['Monthly Mean Total Sunspot Number']]

print("Shape:", sun.shape)
print(sun.head(10))

plt.figure(figsize=(10,5))
plt.plot(sun)
plt.title('Original Sunspot Data'); plt.show()

roll5 = sun.rolling(5).mean()
roll10 = sun.rolling(10).mean()

print("\nMA(5):"); print(roll5.head(10))
print("\nMA(10):"); print(roll10.head(20))

plt.figure(figsize=(10,5))
plt.plot(sun, label='Original')
plt.plot(roll5, label='MA 5')
plt.plot(roll10, label='MA 10')
plt.title('Moving Averages'); plt.legend(); plt.show()

scaler = MinMaxScaler()
scaled = scaler.fit_transform(sun.values).flatten()
scaled = pd.Series(scaled + 1, index=sun.index)

train_size = int(len(scaled)*0.8)
train, test = scaled[:train_size], scaled[train_size:]

model = ExponentialSmoothing(train, trend='add', seasonal='mul', seasonal_periods=12).fit()
pred = model.forecast(len(test))

plt.figure(figsize=(10,5))
plt.plot(train, label='Train')
plt.plot(test, label='Test')
plt.plot(pred, label='Predicted')
plt.title('Holt-Winters Forecast'); plt.legend(); plt.show()

rmse = np.sqrt(mean_squared_error(test, pred))
print("\nRMSE:", rmse)
print("Variance:", scaled.var())
print("Mean:", scaled.mean())

final = ExponentialSmoothing(scaled, trend='add', seasonal='mul', seasonal_periods=12).fit()
future = final.forecast(int(len(scaled)/4))

plt.figure(figsize=(10,5))
plt.plot(scaled, label='Original')
plt.plot(future, label='Future', color='red')
plt.title('Future Forecast'); plt.legend(); plt.show()
```

### OUTPUT:

#### Moving Average:
```
First 10 values of rolling mean (window=5):
Date
1749-01-31       NaN
1749-02-28       NaN
1749-03-31       NaN
1749-04-30       NaN
1749-05-31    110.44
1749-06-30    118.94
1749-07-31    129.68
1749-08-31    128.44
1749-09-30    135.18
1749-10-31    132.00
Name: Monthly Mean Total Sunspot Number, dtype: float64

First 20 values of rolling mean (window=10):
Date
1749-01-31       NaN
1749-02-28       NaN
1749-03-31       NaN
1749-04-30       NaN
1749-05-31       NaN
1749-06-30       NaN
1749-07-31       NaN
1749-08-31       NaN
1749-09-30       NaN
1749-10-31    121.22
1749-11-30    137.98
1749-12-31    141.75
1750-01-31    142.30
1750-02-28    145.67
1750-03-31    146.37
1750-04-30    147.17
1750-05-31    146.37
1750-06-30    151.99
1750-07-31    153.57
1750-08-31    158.16
Name: Monthly Mean Total Sunspot Number, dtype: float64
```
#### Plot Transform Dataset:
<img width="1005" height="545" alt="image" src="https://github.com/user-attachments/assets/2b7cca88-27f0-43f2-9b5d-8c827b0d0277" />
<img width="992" height="545" alt="image" src="https://github.com/user-attachments/assets/5a92df90-f686-4af0-adf7-a4c382f2dfd8" />


```
Root Mean Squared Error (RMSE): 0.17906611954618418
Variance: 0.02906697583970718
Mean: 1.2053711071952424
```



#### Exponential Smoothing:
<img width="1001" height="545" alt="image" src="https://github.com/user-attachments/assets/654c491f-bfca-4cc4-80b3-1cc57a34606a" />



### RESULT:
Thus we have successfully implemented the Moving Average Model and Exponential smoothing using python.
