### exercise 07
X1 <- c(0, 2, 0, 0, -1, 1)
X2 <- c(3,0,1,1,0,1)
X3 <- c(0, 0, 3, 2, 1, 1)
Y <- c("Red", "Red", "Red", "Green", "Green", "Red")
obs <- data.frame(X1=X1, X2=X2, X3=X3, Y=Y)
test_point <- c(0,0,0)
# Q: Compute the Euclidean distance between each observation and the test point
obs_matrix <- obs[, 1:3]
euclidean <- function(x) sqrt(sum((x)^2))
euclidean_distance <- apply(obs_matrix, 1, euclidean)
# Q: What's the prediction with K=1?
obs$Distance <- euclidean_distance
obs[order(obs$Distance),]
# A: We can see the prediction with K=1 is Green by sorting data by distance. Observation 5 at distance 1.41 which is the nearest neighbour.

# Q: What's the prediction with K=3?
# A: When K is 3, our prediction is Red, as the three nearest points are point 5, point 6 and point 2 with distance 1.41, 1.73, 2 respectively. 2/3 are Red, thus the prediction is Red.

### exercise 08
# (a) read College CSV dateset as college object
getwd() # get current working dictionary
college <- read.csv("/Users/maoyubohe618/Documents/R/allR/dataset_csv/College.csv", na.strings = "?",
                    stringsAsFactors = T)
dim(college)
names(college)

# (b) get access to the column where the names of each university are stored
rownames(college) <- college[,1]
View(college)
names(college)
college <- college[,-1] # eliminate the first column in the data
View(college)
names(college)

# (c)
# (i) summary of the college dataset
summary(college)
# (ii) pairs of variables are made into scatter plots
pairs(college[,2:10])
# (iii) produce side-by-side boxplots for Outstate VS Private
boxplot(college$Outstate~college$Private,data = college,
        main = "Tuition for Private/Public University", xlab = "Privte University",
        ylab = "Out-of-State Tuition", col = (c("gold","lightgreen")))
# (iv) a new variable called Elite, plot histogram for Outstate VS Private
Elite <- rep("No", nrow(college))
Elite[college$Top10perc > 50] <- "Yes"
Elite <- as.factor(Elite) # encode Elite as a vextor
college <- data.frame(college, Elite)
summary(college$Elite)
plot(college$Private, college$Outstate, main = "Catogory of University", xlab = "Private University",
     ylab = "Out-of-State Tuition", col = (c("gold","lightgreen")))
# (v) plot four histograms simultaneiusly
hist(college$Apps, breaks = 50, col = "gold", main = "Application Distribution")
hist(college$Accept, breaks = 50, col = "lightgreen", main = "Acceptance Distribution")
hist(college$Enroll, breaks = 50, col = "lightblue", main = "Enrollment Distribution")
hist(college$F.Undergrad, breaks = 50, col = "purple", main = "Full-Time Undergraduates Distribution")
par(mfrow = c(2,2))

### exercise 9
auto <- read.csv("/Users/maoyubohe618/Documents/R/allR/dataset_csv/Auto.csv")
dim(auto)
names(auto)
summary(auto)
# (a) Quantitative or qualitative predictors
# - variable name is qualitative, other predictors are quantitative.
# as.factor() will convert a quantitative variable into a qualitative one

# (b) Range of each quantitative preictor
range(auto$weight)
range(auto$cylinders)
range(auto$acceleration)
# (c) mean and standard deviation
mean(auto$weight)
sd(auto$weight)
# (d) remove observations and create a new subset dataframe
newsubset <- auto[-10:-85,1:8]
newsubset$horsepower <- as.numeric(as.factor(newsubset$horsepower))
# calculate mean, range, standard deviation at the same time by define wrap_cal function
wrap_cal <- function (x) c("Mean"=mean(x), "Min"=min(x), "Max"=max(x), "Standard Dev"=sd(x))
# apply defined function over a List or Vector
lapply(newsubset,wrap_cal)
# (e)
auto$cylinders <- as.factor(auto$cylinders)
plot(auto$cylinders, auto$mpg, xlab = "Number of Cylinders",
     ylab = "Miles/Gallon")
# Overall, the majority of cars having 4 or 5 cylinders in this dataset. In general, a car with four cylinders gets higher mpg(beyond 20 mpg), while a 8-cylinder car gets the lowest mpg (around or below 20 mpg).
# (f)
# find petential relationship between each variable.
plot(auto)
# mpg decreases over the increase of weight
plot(auto$weight, auto$mpg, xlab = "Weight",
     ylab = "Miles/Gallon")

### exercise 10
# (a) load dataset
library(ISLR2)
?Boston
glimpse(Boston)
# The Boston data frame has 506 rows and 13 columns, containing housing values in 506 suburbs of Boston.

# (b) pairwise scatterplots
pairs(Boston)
plot(Boston$crim,Boston$medv)
# It seems to exist a positve non-linear relationship between crim and medv.
plot(Boston$lstat,Boston$medv)
# It seems to exist a negative non-linear relationship between lstat and medv.
plot(Boston$dis,Boston$nox)
# dis and nox seem to have a negative non-linear relationship.It makes sense since the farther the distance, the less the nitric oxides concentration.

# (c) Explore relationship between crim and the other variables.
# Creating a correlation matrix to identify multicollinearity among numerical variables.
corr <- round(cor(Boston), 2)
ggcorrplot(corr, hc.order = TRUE, type = "lower", lab = TRUE, lab_size = 3, method="circle", outline.color = "gray", show.legend = TRUE, show.diag = FALSE, title="Correlogram of Variables")
plot(Boston$rad, Boston$crim)
plot(Boston$tax, Boston$crim)
# RAD - index of accessibility to radial highways (correlation 0.63).
# Tax - full-value property-tax rate per $10,000 (correlation 0.58).
# Above two variables are relatively associated with crim.

# (d) The 5 highest census tracts in crime rate, tax rate and pupil-teacher ratio
crim_sorted <- Boston[order(Boston$crim,decreasing = TRUE),]
head(crim_sorted, n=5)
# index{381,419,406,411,415} are the five 5 census tracts with higest crime rate.
tax_sorted <- Boston[order(Boston$tax,decreasing = TRUE),]
head(tax_sorted, n=5)
# index{489,490,491,492,493} are the five 5 census tracts with higest tax rate.
ptratio_sorted <- Boston[order(Boston$ptratio ,decreasing = TRUE),]
head(ptratio_sorted, n=5)
# index{355,356,128,129,130} are the five 5 census tracts with higest pupil teacher ratios.

# (e) How many of the census tracts bound the Charles river
table(Boston$chas)
# 35 out of 506 census tracts are bound for the Charles river.

# (f) Median pupil-teacher ratio
median(Boston$ptratio)
# 19.05

# (g) lowest value of median value of owner-occupied homes
min(Boston$medv) # 5
Boston[Boston$medv==5,]
summary(Boston)
# index{399,406} have the lowest median value of owner-occupied homes.
# Relative to the other towns, these two suburbs have high CRIM, INDUS in quantile 75%, does not bound the Charles river, above mean NOX, RM below quantile 25%, maximum AGE, DIS near to the minimum value, maximum RAD, TAX in quantile 75%, PTRATIO as well, LSTAT above quantile 75%.

# (h) average number of rooms per dwelling
room_number7 <- Boston[Boston$rm > 7,]
nrow(room_number)
# There are 64 of the census tracts average more 7 rooms per dwelling.
room_number8 <- Boston[Boston$rm > 8,]
summary(room_number8)
# CRIM is lower, INDUS proportion is lower, * % of lower status of the population (LSTAT) is lower.


