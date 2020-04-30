library(randomForest)
library(psych)
library(MLmetrics)
library(rsample)
library(forecast)
library(corrplot)
library(EnvStats)
library(timeSeries)
library(e1071)
library(DAAG) #Cross Validation(Linear Regression)
library(TTR)
library(caret)
library(ggplot2)
library(lubridate)
library(tidyr)


Transportation_df <- read.csv("transportation-fuels-spot-prices-beginning-2006.csv")

#Checking for Structure and Summarising the dataframe 
head(Transportation_df,5)
str(Transportation_df)
summary(Transportation_df)

#Checking for Null Values
sum(is.na(Transportation_df$NY.Jet.Fuel.Price....gal.))

#Using the Date Column to Add New Column Month And Year

Transportation_df$month <- month(Transportation_df$Date)
Transportation_df$year <- year(Transportation_df$Date)

#Selecting the Newly Added Column and Equating it to Variable  
temp <- Transportation_df %>%
        dplyr::select(month,year, NY.Jet.Fuel.Price....gal.) %>%
        na.omit() %>%
        dplyr::arrange(desc(-year,month))

#Removing the 6th,7th and 8th column from the dataframe
Transportation_df <- Transportation_df[,-c(6,7,8)]

######### GRAPHICAL ANALYSIS(Scatterplot, Density Plot, Correlation Plot)#########

#Scatterplots
scatter.smooth(x=Transportation_df$WTI.Crude.Oil.Spot.Price....barrel.,
               y=Transportation_df$NY.Conventional.Gasoline.Spot.Price....gal.
               ,xlab = "WTI Crude Oil $/Ga",ylab = "Conventional Gas $/Ga")

scatter.smooth(x=Transportation_df$Brent.Crude.Oil.Spot.Price....barrel.,
               y=Transportation_df$NY.Ultra.Low.Sulfur.Diesel.Spot.Price....gal.
               ,xlab = "Brent Crude Oil $/Ga",ylab = "Sulfur Diesel $/Ga")

#Density Plot to Check Normality of Response Variables
plot(density(Transportation_df$NY.Ultra.Low.Sulfur.Diesel.Spot.Price....gal.),
     main="Density Plot: Sulphur Diesel", ylab="Frequency", 
     sub=paste("Skewness:", round(e1071::skewness(Transportation_df$NY.Ultra.Low.Sulfur.Diesel.Spot.Price....gal.), 2)))  # density plot for 'speed'
polygon(density(Transportation_df$NY.Ultra.Low.Sulfur.Diesel.Spot.Price....gal.), col="red")

plot(density(Transportation_df$NY.Conventional.Gasoline.Spot.Price....gal.),
     main="Density Plot: Conventional Gasoline", ylab="Frequency", 
     sub=paste("Skewness:", round(e1071::skewness(Transportation_df$NY.Conventional.Gasoline.Spot.Price....gal.), 2)))  # density plot for 'speed'
polygon(density(Transportation_df$NY.Conventional.Gasoline.Spot.Price....gal.), col="red")

#Checking for Correlation Between the Columns and Plotting them
pairs.panels(Transportation_df)

######################### RANDOM SAMPLING OF DATASET ##########################

#Setting Seed and Dividing the Data Set into Training, Testing, Validation Samples
set.seed(123)
sample.train <- sample(seq_len(nrow(Transportation_df)),size = floor(0.60*nrow(Transportation_df)))
sample.test <- sample(seq_len(nrow(Transportation_df)),size = floor(0.20*nrow(Transportation_df)))
sample.validate <- sample(seq_len(nrow(Transportation_df)),size = floor(0.20*nrow(Transportation_df)))

trans_train <- Transportation_df[sample.train,]
trans_validate <- Transportation_df[sample.validate,]
trans_test <- Transportation_df[sample.test,]


############################ LINEAR REGRESSION ################################

#Building Model on the Whole Data Set
linearC <- lm(NY.Conventional.Gasoline.Spot.Price....gal.
        ~WTI.Crude.Oil.Spot.Price....barrel.,data = Transportation_df)

summary(linearC)

linearG <- lm(NY.Ultra.Low.Sulfur.Diesel.Spot.Price....gal.
              ~Brent.Crude.Oil.Spot.Price....barrel.,data = Transportation_df)

summary(linearG)

#Building the Linear Models and Plotting Regression line

#Model 1
lm1<-lm(NY.Conventional.Gasoline.Spot.Price....gal.
        ~WTI.Crude.Oil.Spot.Price....barrel.,data = trans_train)

plot(x=trans_train$WTI.Crude.Oil.Spot.Price....barrel.,y=trans_train$NY.Conventional.Gasoline.Spot.Price....gal.
     ,xlab = "WTI Crude Oil $/Ga",ylab = "Conventional Gas $/Ga")
abline(lm1,col="green")

#Summarizing Model 1
summary(lm1)
AIC(lm1)

#Model 2
lm2<-lm(NY.Ultra.Low.Sulfur.Diesel.Spot.Price....gal.
        ~Brent.Crude.Oil.Spot.Price....barrel.,data = trans_train)

plot(x=trans_train$Brent.Crude.Oil.Spot.Price....barrel.,
     y=trans_train$NY.Ultra.Low.Sulfur.Diesel.Spot.Price....gal.
     ,xlab="Brent Crude Oil $/Ba",ylab="Ultra Low Sulfur Diesel $/Ga")
abline(lm2, col="red")

#Summarizing Model 2
summary(lm2)
AIC(lm2)

#Predicting the Linear Models
lt1 <- predict(lm1,data=trans_test$NY.Conventional.Gasoline.Spot.Price....gal.)
lt2 <- predict(lm2,data=trans_test$NY.Ultra.Low.Sulfur.Diesel.Spot.Price....gal.)

#Calculating prediction accuracy
actual_pred1 <- as.data.frame(cbind(actual=trans_test$NY.Conventional.Gasoline.Spot.Price....gal.,
                                    predicted= lt1))
cor_accuracy1 <- cor(actual_pred1)
head(actual_pred1)

actual_pred2 <- as.data.frame(cbind(actual=trans_test$NY.Ultra.Low.Sulfur.Diesel.Spot.Price....gal.,
                                    predicted= lt2))
cor_accuracy2 <- cor(actual_pred2)
head(actual_pred2)

#Min Max Accuracy(Higher the better)
min_max_accuracy1 <-mean(apply(actual_pred1, 1, min)/apply(actual_pred1, 1, max))
min_max_accuracy2 <-mean(apply(actual_pred2, 1, min)/apply(actual_pred2, 1, max))

#Root Mean Square Error(Lower the better)
rmserr1 <- caret::RMSE(actual_pred1$predicted,actual_pred1$actual)
rmserr2 <- caret::RMSE(actual_pred2$predicted,actual_pred2$actual)

#Mean Absolute Percentage Errors(Lower the better)
mape1 <- mean(abs(actual_pred1$predicted - actual_pred1$actual)/actual_pred1$actual)
mape1

mape2 <- mean(abs(actual_pred2$predicted - actual_pred2$actual)/actual_pred2$actual)
mape2

#Cross Validation of Linear Models 1 & 2
cvResults1 <- CVlm(data = Transportation_df,form.lm = formula(NY.Conventional.Gasoline.Spot.Price....gal.~WTI.Crude.Oil.Spot.Price....barrel.),
                   m = 3,dots = FALSE,seed = 29,plotit = c("Observed","Redidual"),
                   main ="Small symbols show cross-validation",legend.pos = "topleft",
                   printit = TRUE)
attr(cvResults1, 'ms')

cvResults2 <- CVlm(data = Transportation_df,form.lm = formula(NY.Ultra.Low.Sulfur.Diesel.Spot.Price....gal.~Brent.Crude.Oil.Spot.Price....barrel.),
                   m = 3,dots = FALSE,seed = 29,plotit = c("Observed","Redidual"),
                   main ="Small symbols show cross-validation",legend.pos = "topleft",
                   printit = TRUE)
attr(cvResults2, 'ms')

##################### MULTIPLE LINEAR REGRESSION ########################
#Building Model on the Whole Data Set
Multi_linearC <- lm(NY.Conventional.Gasoline.Spot.Price....gal.
                    ~WTI.Crude.Oil.Spot.Price....barrel. + Brent.Crude.Oil.Spot.Price....barrel. ,data = Transportation_df)

summary(Multi_linearC)

Multi_linearG <- lm(NY.Ultra.Low.Sulfur.Diesel.Spot.Price....gal.
                    ~WTI.Crude.Oil.Spot.Price....barrel. + Brent.Crude.Oil.Spot.Price....barrel.,data = Transportation_df)

summary(Multi_linearG)

#Building the MLR Models

#Model 1
Mlr1<-lm(NY.Conventional.Gasoline.Spot.Price....gal.
         ~WTI.Crude.Oil.Spot.Price....barrel. + Brent.Crude.Oil.Spot.Price....barrel.,data = trans_train)


#Summarizing Model 1
summary(Mlr1)
AIC(Mlr1)

#Model 2
Mlr2<-lm(NY.Ultra.Low.Sulfur.Diesel.Spot.Price....gal.
         ~Brent.Crude.Oil.Spot.Price....barrel. + WTI.Crude.Oil.Spot.Price....barrel.,data = trans_train)


#Summarizing Model 2
summary(Mlr2)
AIC(Mlr2)

#Predicting the Linear Models
Mlt1 <- predict(Mlr1,data=trans_test$NY.Conventional.Gasoline.Spot.Price....gal.)
Mlt2 <- predict(Mlr2,data=trans_test$NY.Ultra.Low.Sulfur.Diesel.Spot.Price....gal.)

#Calculating prediction accuracy
Multi_actual_pred1 <- as.data.frame(cbind(actual=trans_test$NY.Conventional.Gasoline.Spot.Price....gal.,
                                          predicted= Mlt1))
cor_accuracy1 <- cor(Multi_actual_pred1)
head(Multi_actual_pred1)

Multi_actual_pred2 <- as.data.frame(cbind(actual=trans_test$NY.Ultra.Low.Sulfur.Diesel.Spot.Price....gal.,
                                          predicted= Mlt2))
cor_accuracy2 <- cor(Multi_actual_pred2)
head(Multi_actual_pred2)

#Min Max Accuracy(Higher the better)
Multi_min_max_accuracy1 <-mean(apply(Multi_actual_pred1, 1, min)/apply(Multi_actual_pred1, 1, max))
Multi_min_max_accuracy2 <-mean(apply(Multi_actual_pred2, 1, min)/apply(Multi_actual_pred2, 1, max))

#Root Mean Square Error(Lower the better)
Multi_rmserr1 <- caret::RMSE(Multi_actual_pred1$predicted,Multi_actual_pred1$actual)
Multi_rmserr2 <- caret::RMSE(Multi_actual_pred2$predicted,Multi_actual_pred2$actual)

#Mean Absolute Percentage Errors(Lower the better)
Multi_mape1 <- mean(abs(Multi_actual_pred1$predicted - Multi_actual_pred1$actual)/Multi_actual_pred1$actual)
Multi_mape1

Multi_mape2 <- mean(abs(Multi_actual_pred2$predicted - Multi_actual_pred2$actual)/Multi_actual_pred2$actual)
Multi_mape2

#Cross Validation of Linear Models 1 & 2
Multi_cvResults1 <- cv.lm(data = Transportation_df,form.lm = formula(NY.Conventional.Gasoline.Spot.Price....gal.~WTI.Crude.Oil.Spot.Price....barrel. + Brent.Crude.Oil.Spot.Price....barrel.),
                          m = 3,dots = FALSE,seed = 29,plotit = c("Observed","Redidual"),
                          main ="Small symbols show cross-validation",legend.pos = "topleft",
                          printit = TRUE)
attr(Multi_cvResults1, 'ms')

Multi_cvResults2 <- cv.lm(data = Transportation_df,form.lm = formula(NY.Ultra.Low.Sulfur.Diesel.Spot.Price....gal.~Brent.Crude.Oil.Spot.Price....barrel. + WTI.Crude.Oil.Spot.Price....barrel.),
                          m = 3,dots = FALSE,seed = 29,plotit = c("Observed","Redidual"),
                          main ="Small symbols show cross-validation",legend.pos = "topleft",
                          printit = TRUE)
attr(Multi_cvResults2, 'ms')


##################### RANDOM FOREST REGRESSION ########################

#Applying Random Forest Regression to the training dataset and predicting using testing data

#Random Forest training for Gasoline
randomFG <- randomForest(NY.Conventional.Gasoline.Spot.Price....gal.~.,data = trans_train[,-1],importance=TRUE,na.action = na.omit)

round(importance(randomFG),2)

#Predicting Gasoline Price Using Testing Dataset and Finding RMSE Values 
predG <- predict(randomFG,newdata = trans_test[,-2])

rmseG <- RMSE(predG,trans_test$NY.Conventional.Gasoline.Spot.Price....gal.)
rmseG

mseG <- MSE(predG,trans_test$NY.Conventional.Gasoline.Spot.Price....gal.)
mseG

#Correlation Between the Prediction and the Actual Gasoline Price
corG <- cor(predG,trans_test$NY.Conventional.Gasoline.Spot.Price....gal.)
corG

#Plotting the Random Forest object (Error vs Number of Trees)
plot(randomFG)

#Random Forest training for Diesel
randomFD <- randomForest(NY.Ultra.Low.Sulfur.Diesel.Spot.Price....gal.~.,data = trans_train[,-1],importance=TRUE,na.action = na.omit)

round(importance(randomFD),2)

#Predicting Diesel Price Using Testing Dataset and Finding RMSE, MSE Values 
predD <- predict(randomFD,newdata = trans_test[,-3])

rmseD <- RMSE(predD,trans_test$NY.Ultra.Low.Sulfur.Diesel.Spot.Price....gal.)
rmseD

mseD <- MSE(predD,trans_test$NY.Ultra.Low.Sulfur.Diesel.Spot.Price....gal.)
mseD
#Correlation Between the Prediction and the Actual Diesel Price
corD <- cor(predD,trans_test$NY.Ultra.Low.Sulfur.Diesel.Spot.Price....gal.)
corD

#Plotting the Random Forest object (Error vs Number of Trees)
plot(randomFG)

#Random Forest Cross-Validation for Training and Testing Dataset

#Cross-Validation for Gasoline Price on Training and Testing 
resultG1 <- rfcv(trans_train[,-1],trans_train$NY.Conventional.Gasoline.Spot.Price....gal.,step = 0.5)
with(resultG1, plot(n.var, error.cv, log="x", type="o", lwd=2))

resultG2 <- rfcv(trans_test[,-1],trans_test$NY.Conventional.Gasoline.Spot.Price....gal.,step = 0.5)
with(resultG2, plot(n.var, error.cv, log="x", type="o", lwd=2))

#Cross-Validation for Diesel Price on Training and Testing 
resultD1 <- rfcv(trans_train[,-1],trans_train$NY.Ultra.Low.Sulfur.Diesel.Spot.Price....gal.,step = 0.5)
with(resultD1, plot(n.var, error.cv, log="x", type="o", lwd=2))

resultD2 <- rfcv(trans_test[,-1],trans_test$NY.Ultra.Low.Sulfur.Diesel.Spot.Price....gal.,step = 0.5)
with(resultD2, plot(n.var, error.cv, log="x", type="o", lwd=2))

######################## TIME SERIES ANALYSIS ########################

Transport.df <- temp

#Using ts() for time series and plotting it
df <-ts(Transport.df$NY.Jet.Fuel.Price....gal.,frequency = 12,start = c(2006,1))
plot.ts(df)

# 12 month moving average
sm <- ma(df, order=12) 
lines(sm, col="blue")  

#Smoothing and plotting the dataframe
k<- SMA(df,n=10)
plot.ts(k)

#Using decompose function to split the time series components into season, trend and irregularities of the seasonal data
decom <- decompose(df)

#Sesonally adjusting
decompose1 <-df- decom$seasonal

#plotting seasonally adjusted time series
plot(decompose1)

#Forecasting using simple exponential smoothing
forecast <- HoltWinters(df,beta=FALSE,gamma = FALSE)

#Getting the fitted value
forecast$fitted

#Getting Sum of squared errors
forecast$SSE

#Plotting the orignal time series against the forecasted
plot(forecast)

#Predicting the next 9 years of fuel price using forecast.HoltWinters()
future <- forecast:::forecast.HoltWinters(forecast,h=9)

future

#Plotting the predictions made by the function
forecast:::plot.forecast(future)

#Plotting the residuals for checking constant variance of forecast errors 
plot(future$residuals)

