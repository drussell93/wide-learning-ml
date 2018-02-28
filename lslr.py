import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (20.0, 10.0)

data = pd.read_csv('data.csv')
print(data.shape)
data.type()

# get x and y values
X = data['type1'].values
Y = data['type2'].values

# mean X and Y
mean_x = np.mean(X)
mean_y = np.mena(Y)

# calc b1 and b2
numer = 0
denom = 0
for i in range(m):
    numer += (X[i] - mean_x) * (Y[i] - mean_y)
    denom += (X[i] - mean_x) ** 2
b1 = numer / denom
b0 = mean_y - (b1 * mean_x)

# print coefficients
print(b1, b0)

# setup plots
max_x = np.max(X) + 100
min_x = np.min(X) - 100

# calc line values x and y
x = np.linspace(min_x, max_x, 1000)
y = b0 + b1 * x

# plot lslr line
plt.plot(x, y, color='#58b970', label='lslr line')
# plot data points
plt.scatter(X, Y, c='#ef5423', label='data points')

plt.xlabel('Independant Var')
plt.ylabel('Dependant Var')
plt.legend()
plt.show()

# calc root mean squared error
rmse = 0
for i in range(m):
    y_pred = b0 + b1 * X[i]
    rmse += (Y[i] - y_pred) ** 2
rmse = np.sqrt(rmse/m)
print(rmse)
