# GDP Regression Model
A model of the Indian economy over the last 40 years using Linear Regression.
# Introduction
The Cobb–Douglas production function is a particular functional form of the production function, widely used to represent the technological relationship between the amounts of two or more inputs (particularly physical capital and labor) and the amount of output that can be produced by those inputs.
Specifically, it is given by $$Y=AK^\alpha L^\left(1-\alpha\right)$$ where $Y$ is the GDP or Total Production of the country, $A$ is the Total Factor Productivity (TFP), $K$ is the capital input, $L$ is the labour input, and $\alpha$ is the output elasticity of capital.
See more [here](https://en.wikipedia.org/wiki/Cobb%E2%80%93Douglas_production_function#:~:text=In%20economics%20and%20econometrics%2C%20the,that%20can%20be%20produced%20by).

Typically, values of $K$, $L$ and $Y$ are known, and using these, $\alpha$ and $A$ can be calculated, using linear regression, as demonstrated here.
Taking log both sides, we have:
$$\ln(Y)=\ln(A)+\alpha \ln(K)+\left(1-\alpha\right)\ln(L)$$
Rearranging, we obtain,
$$\ln\left(\frac{Y}{L}\right)=\alpha\ln\left(\frac{K}{L}\right)+\ln(A)$$
Thus, setting $\ln\left(\frac{K}{L}\right)$ as the *feature* and $\ln\left(\frac{Y}{L}\right)$ as the *target* value converts our problem into the one-dimensional linear regression form, $y=wx+b$, where $w$ represents $\alpha$ and $b$ represents $\ln(A)$.
