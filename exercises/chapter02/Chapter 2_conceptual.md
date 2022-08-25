## Chapter 2

### Exercises Conceptual

##### 1. Bias-Variance Trade-Off

<img src="/Users/maoyubohe618/Library/Application Support/typora-user-images/截屏2022-08-15 15.38.31.png" alt="截屏2022-08-15 15.38.31" style="zoom:50%;" />

The expected test MSE, for a given value x0, can decomposed  into sum of quantities: the variance of <img src="/Users/maoyubohe618/Desktop/截屏2022-08-15 15.35.41.png" alt="截屏2022-08-15 15.35.41" style="zoom:50%;" />, the squared bias of <img src="/Users/maoyubohe618/Desktop/截屏2022-08-15 15.35.41.png" alt="截屏2022-08-15 15.35.41" style="zoom:50%;" />,  and the variance of the error terms <img src="/Users/maoyubohe618/Library/Application Support/typora-user-images/截屏2022-08-15 15.37.15.png" alt="截屏2022-08-15 15.37.15" style="zoom:50%;" /> That is shown on equation 2.7. To minimise the expected test error, a ideal statistical learning model should simultaneously achieves low variance and low bias. As a general rule, as we use more flexible methods, the variance will increase and the bias will decrease. This is referred to as the **bias-variance trade-off** .

(a) **Extremely large sample, few predictors**

A flexible method is expected to be **better**.  

Since the sample size is extremely large and the number of predictors is small, a more flexible method would be able to better fit the data, while not fitting the noise due to the very large sample size. In other words, a more flexible model would have less bias without much risk of overfitting. 

(b) **Extremely large predictors, few observations**

A flexible method is expected to be **worse**.

It is very likely that when the numbe of predictors is extremely large and the number of observations is small, a flexible model would fit the noise. This means when giving another new data set to test the model, the quality of fit would likely be significantly different. It is possible that we would get high variances and the model would be overfitting.

(c) **Highly non-linear relationship**

A flexible method is expected to be **better**.

In this case, a flexible method will likely be better fit a highly non-linear relationship, otherwise the model would be with high bias and not capture the non-linearities characteristics. A flexible model with more freedom shapes can better estimate the true f.  

(d) **Extremely high variance**

A flexible method is expected to be **worse**.

Since the variance is extremely high, a more flexible model will fit the noise and thus very likely overfit. A less flexible model will be more likely to still capture the essential features of the model without picking up the noise or random fluctuation. 

##### 2.Regression VS Classification Problems, Inference or Prediction

(a) CEO salary

Regression, since the variable we are interested in, CEO salary, is quantitative. As we want to know the relationship between each  individual predictor and the response. Thus we are interested in inference. The number of predictors is 3, and the sample size is 500 (p=3, n=300).

(b) New product launch: success or failure

Classification, bacause we wish to know the outcome is categorical, success or failure. In this case, we would like to know whether launching a new product will be success or failure more than understand why. So we are interested in prediction. The number of predictors is 13, the sample size is 20 (p=13, n=20).

(c) Change in the USD/Euro exchange

Regression, since the variable we want to predict is quantitative, and we are more interested in prediction given weekly data of different markets. The number of predictors is 3, and the sample size is 52 (p=3, n=300).

##### 3. Bias-Variance Decomposition

 (a) a sketch for bias-variance decomposition

<img src="/Users/maoyubohe618/Library/Application Support/typora-user-images/截屏2022-08-17 18.22.13.png" alt="截屏2022-08-17 18.22.13" style="zoom:33%;" />

<img src="/Users/maoyubohe618/Library/Application Support/typora-user-images/image-20220819133633666.png" alt="image-20220819133633666" style="zoom:50%;" />

```python
# the functions chosen here were chosen just as a rough, quick way to sketch the functions in a plot
# they do not represent in any way an analytical formula for these quantities or anything of the sort
# these formulas would depend on the model and fitting procedure in any case
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0.0, 10.0, 0.02)

def squared_bias(x):
    return .002*(-x+10)**3
def variance(x):
    return .002*x**3
def training_error(x):
    return 2.38936 - 0.825077*x + 0.176655*x**2 - 0.0182319*x**3 + 0.00067091*x**4
def test_error(x):
    return 3 - 0.6*x + .06*x**2
def bayes_error(x):
    return x + 1 - x

plt.xkcd()
#frame = plt.gca()
#frame.axes.xaxis.set_ticklabels([])
plt.figure(figsize=(10, 8))
plt.plot(x,squared_bias(x), label='squared bias')
plt.plot(x, variance(x), label='variance')
plt.plot(x, training_error(x), label='training error')
plt.plot(x, test_error(x), label='test error')
plt.plot(x, bayes_error(x), label='Bayes error')
plt.legend(loc='upper center')
plt.xlabel('model flexibility')
plt.show()

# arbitrary units
```

(b) all five lines >= 0

As a general rule, as we use **more flexible methods, the variance will increase and the bias will decrease.** The relative rate of change of these two quantities determine whether the test MSE increases or decreases. As we increase the flexibility of a class of methods, the bias tends to initially decrease faster than the variance increases. Consequently, the expected test MSE declines. However, at some point increasing flexibility has little impact on the bias but starts to significantly increase the variance. When this happens the test MSE increases.

- Squared bias - decreases monotonically because a statistical method increases in flexibility yields a closer fit.
- Variance - increases monotonically because a statistical method increases in flexibility yields higher variance .
- Training error - decreases monotonically because a more flexible model can fit points better.
- Bayes (irreducible) error - defines the lowest limit, it is the smallest test MSE over all statistical methods. When the training error is lower than the irreducible error, overfitting has taken place.
- Test error - U shape curve because the test MSE is the sum of three quantities: he variance of <img src="/Users/maoyubohe618/Desktop/截屏2022-08-15 15.35.41.png" alt="截屏2022-08-15 15.35.41" style="zoom:50%;" />, the squared bias of <img src="/Users/maoyubohe618/Desktop/截屏2022-08-15 15.35.41.png" alt="截屏2022-08-15 15.35.41" style="zoom:50%;" />,  and the variance of the error terms <img src="/Users/maoyubohe618/Library/Application Support/typora-user-images/截屏2022-08-15 15.37.15.png" alt="截屏2022-08-15 15.37.15" style="zoom:50%;" /> Thus increasing in flexibility yields a closer fit before it overfits. 

##### 4. Some Real-Life  Applications for Statistical Learning

##### (a) Real-Life Applicaitons of Classification

- **Email spam filters.** There are only two categories of the response of a email spam filter: spam or not spam, and the predictors can include several variables: whether you have corresponded with the sender, whether you have their email address in some of your non-spam emails, whether similar emails have already been tagged as spam by other users, etc. The main goal is **prediction** bacause the main function is to accurately predict whether a future email message is spam or not-spam. <!--(An important aspect in this case is the rate of false positives and false negatives. In the case of email it is usually much more acceptable to have false negatives (spam that got through to your inbox) than false positives (important email that was classified as spam and you have never noticed).)-->
- **Face recognition**. In the case of face recognition, the input is an image, the classes are people to be recognised, the response has just two categories: recognised or unrecognised, and the learning program should learn to associate the face images to identities. Here the main goal is **prediction**, since the most important thing is to recognise a correct identity from an image. 

- **Stock price forecasting**. We are interested in **predicting** the price of a stock, our goal for this setting is that it predict weather the stock market increases or decreases. Here, predicting accurately is more important than obtaining a deep understanding of the relationships between each individual predictor and the response. 

##### (b) Real-Life Applicaitons of Regression

- **Estate price evaluating**. Is a house over-valued or under-valued (inference)?  How much extra will a house worth if it has a view of a river (prediction)? In this case, the response can be varied, and the predictors such as zoning, crime rate, distance from a river or a underground station, size of house, and so forth.
- **Wether forecasting**. The response can be temperature, wind speed, UV index, air pollution, etc. The predictors can be the same variables for previous times. The goal is both **prediction** (will it rain tomorrow?) and **inference** (what causes temperature increase?).
- **Covid-19 forecasts**. In this case, the response can be forecasts of new and total deaths in different locations, and the predictors can be the same quantities reported by different hospitals or organisations, infection rate, population of a city, etc. The important applicaition is both prediction (forecasts of new deaths?) and inference (what causes new deaths?).

##### (c) Real-Life Applicaitons of  Clustering

- **Recommender systems**. A film recommendation system like Netfilx uses things like 

  - viewing history and rated titles
  - other members with similar tastes and preferences on this system 
  - information about the titles, such as their genre, categories, actors, release year, etc

  all these pieces of data are used as inputs to do the personalised recommendation for users. 

- **Crime analysis**. Cluster analysis can be used to identify areas where there are greater incidences of particular types of crime. By identifying these distinct areas or "hot spots" where a similar crime has happened over a period of time, it is possible to manage law enforcement resources more effectively.
- **Market research**. Using cluster analysis, market researchers can group consumers into market segments according to their similarities. This can provide a better understanding on group characteristics, as well as it can provide useful insights about how different groups of consumers/potential customer behave and relate with different groups.

##### 5.Flexible Methods VS Less Flexible Methods

##### - Advantages

​	(1) fit for data set with larger sample size 

​	(2) less bias

#####  - Disadvantages

​	(1) risk of overfitting

​	(2) high variance

​	(3) less interpretability

​	(4) hard to train model

##### Which Method is preferable?

##### - More Flexible

​	(1) large sample size and small number of predictors.

​	(2) non-linear relationship between the predictors and response.

​	(3) the goal is prediction and less interpretability required.

##### - Less Flexible

​	(1) small sample size and large number of predictors.

​	(2) the goal is inference and easy to interpret the relationship between predictors and response.



##### 6. Non-Parametric VS Parametric Statistical Learning Approach

Parametric method makes explicit assumptions about the functional form of f given data set we are going to modelling. It reduces the problem of estimating f down to one of estimating a set of parameters. By comparison with non-parametric method, this approach does not impose any pre-specified method on f, it seeks an estimate of f that gets as close to the data points as possible without being too rough or wiggly. 

The advantages of a parametric approach are, in general, that it requires less observations, is faster and more computationally tractable, and less misguided by noise. The disadvantages are the method impose restrictions on the flexibility of model , and any parametric approach brings with it the possibility that the functional form used to eatimate f is very different from the true f, in which case the resulting model will not fit the data well. In contrast, non-parametric approaches avoid this risk, since essentially no assumptions about the form of f is made. But non-parametric approaches suffer from a major disadvantage: since they do not reduce the problem of estimating f to a small number of parameters, a very large number of observations is required in order to obtain an accurate estimate for f. 

##### 7.

```R
# exercise 07
X1 <- c(0, 2, 0, 0, -1, 1)
X2 <- c(3,0,1,1,0,1)
X3 <- c(0, 0, 3, 2, 1, 1)
Y <- c("Red", "Red", "Red", "Green", "Green", "Red")
obs <- data.frame(X1=X1, X2=X2, X3=X3, Y=Y)
test_point <- c(0,0,0,NULL)
# (a): Compute the Euclidean distance between each observation and the test point
obs_matrix <- obs[, 1:3]
euclidean <- function(x) sqrt(sum((x)^2))
euclidean_distance <- apply(obs_matrix, 1, euclidean)


```

##### (a) Euclidean distance 

Euclidean distance to test point (0,0,0) : 
Distance = 3.000000 2.000000 3.162278 2.236068 1.414214 1.732051

##### (b) What's the prediction with K=1? Why?

```R
obs$Distance <- euclidean_distance
obs[order(obs$Distance),]
```

Obs sorted by distance:

|      | X1   | X2   | X3   | Y     | Distance    |
| ---- | ---- | ---- | ---- | ----- | ----------- |
| 1    | 0    | 3    | 0    | Red   | 3           |
| 2    | 2    | 0    | 0    | Red   | 2           |
| 3    | 0    | 1    | 3    | Red   | 3.16227766  |
| 4    | 0    | 1    | 2    | Green | 2.236067977 |
| 5    | -1   | 0    | 1    | Green | 1.414213562 |
| 6    | 1    | 1    | 1    | Red   | 1.732050808 |

We can see the prediction with K=1 is Green by sorting data by distance. Observation 5 at distance 1.41 which is the nearest neighbour.

##### (c) What's the prediction with K=3? Why?

When K is 3, our prediction is Red, as the three nearest points are point 5, point 6 and point 2 with distance 1.41, 1.73, 2 respectively. 2/3 are Red, thus the prediction is Red.

##### (d) Highly non-linear Bayes decision boundary

We would expect better value with small K. When k is small, we can better capture local non-liearitities, because majority of neighbours can vary significantly from point to point since the there are fewer neighbours. By contrast, a large value of k leads to a smoother decision boundary, this happens because KNN uses majority voting and this means less emphasis on individual points. One of its nearest neighbors changing from one class to the other would still leave the majority voting the same. So in this problem, we would expect **the best value of K to be small**.



 































