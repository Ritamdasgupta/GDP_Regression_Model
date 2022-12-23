# GDP Regression Model
A model of the Indian economy over the last 40 years using Linear Regression.
# Introduction
The aim of this project is:
1. The demonstration of the validity of the Cobb-Douglas production function in the context of the Indian economy over the last 40 years.
2. Calculation of Total Factor Productivity $A$ and Capital Output Elasticity $\alpha$ using Linear Regression.
3. To be able to make predictions about the GDP, given capital and labour inputs.
# Theory
The Cobb–Douglas production function is a particular functional form of the production function, widely used to represent the technological relationship between the amounts of two or more inputs (particularly physical capital and labor) and the amount of output that can be produced by those inputs.
Specifically, it is given by $$Y=AK^\alpha L^\left(1-\alpha\right)$$ where $Y$ is the GDP or Total Production of the country, $A$ is the Total Factor Productivity (TFP), $K$ is the capital input, $L$ is the labour input, and $\alpha$ is the output elasticity of capital.
See more [here](https://en.wikipedia.org/wiki/Cobb%E2%80%93Douglas_production_function#:~:text=In%20economics%20and%20econometrics%2C%20the,that%20can%20be%20produced%20by).

Typically, values of $K$, $L$ and $Y$ are known, and using these, $\alpha$ and $A$ can be calculated, using linear regression, as demonstrated here.
Taking log both sides, we have:
$$\ln(Y)=\ln(A)+\alpha \ln(K)+\left(1-\alpha\right)\ln(L)$$
Rearranging, we obtain,
$$\ln\left(\frac{Y}{L}\right)=\alpha\ln\left(\frac{K}{L}\right)+\ln(A)$$
Thus, setting $\ln\left(\frac{K}{L}\right)$ as the *feature* and $\ln\left(\frac{Y}{L}\right)$ as the *target* value converts our problem into the one-dimensional linear regression form, $y=wx+b$, where $w$ represents $\alpha$ and $b$ represents $\ln(A)$. We now are ready to design a model to find these values.
# Description of the Model
Raw input is given in the form of two numpy arrays, X and Y, where X is the input array consisting of (L,K) pairs and Y is the output array containing the corresponding GDP. These would be used to train the regression model. In the code, data taken from an [official RBI source](https://www.rbi.org.in/Scripts/KLEMS.aspx) has been provided (commented out). X_train represents data from 1980-2020, Y_train is corresponding GDP. The data is also separately provided for each decade in 4 different arrays, along with corresponding GDPs. 

The `input_transformation()` function has been used to transform the X and Y arrays into $\ln(Y/L)$ and $\ln(K/L)$ arrays so that linear regression can be directly applied upon them. 

Upon these processed input and output arrays, *feature scaling* has been performed using Z-score normalization so that the gradient descent algorithm runs faster.
This has been done using the `feature_scaling()` function.

Next, the cost-function `J()` has been implemented. It is this mean-square cost function which we are looking to minimize. Note that a regularization parameter has been provided, `lambda_`, to prevent overfitting, however, in case of the datasets we have provided the best fit appears without regularization, since the optimal values of $w$ and $b$ are already quite small.

A helper function, `partial_diff()` has been defined to calculate the partial derivative values at given $w$ and $b$. This will be useful for the gradient descent algorithm.

Finally, `gradient_descent()` has been implemented to optimize the values of parameters $w$ and $b$ on the scaled data, so as to minimize the cost function. We are running 100000 iterations of gradient descent, which appears to be sufficient, as demonstrated by the learning curve on X_train and Y_train:
