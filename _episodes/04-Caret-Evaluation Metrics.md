---
title: "Evaluation Metrics with caret"
teaching: 40
exercises: 0
questions:
- "How do we measure the accuracy of ML model"
objectives:
- "Learn different metrics with caret"
keypoints:
- "Caret"
---

# 4 Evaluation Metrics 

Here, we will discuss how we can evaluate our prediction. This process is somewhat different depending on whether we predict numerical outouts (regression) or categorical outputs (classification).

## 4.1 Classification model 

Let's do our first prediction. Let's predict the species of irises from the 4 variables (sepal / petal width and length). First, let's train the model:

```r
train_inputs=training[,1:4]
train_outputs=training[,5]
model <- train(train_inputs, train_outputs, method="lda")
```

Here, we have three classes (three species of irises), and we make the model which predicts the species from the four input variables. We use Fisher's linear discriminant analysis for this purpose. It's a simple classifier based on Ronald Fisher's seminal 1936 paper. Now, let's apply the model to the test set:

```r
predictions <- predict(model,testing)
```

Now, how good was our prediction? A quick way to evaluate the accuracy is to see how many times the predicted species were equal to observed species:

```r
mean(predictions==testing$Species)
```

This is termed *classification accuracy*, and is the most basic metric of our prediction. We can get more detailed metrics by computing confusion matrix (here, we not only see how many observations were correctly predicted, but also look at mistakes: which species are confused with other species).

```r
confusionMatrix (predictions,testing$Species)
```

Really good accuracy! We only made two mistakes: two examples of *versicolor* were mistaken for *virginica*.

We can visualize the confusion matrix:

```r
library(reshape2)
cm_df <- melt(cm$table)
ggplot(cm_df, aes(x = Prediction, y = Reference, fill = value)) +
 geom_raster() + scale_fill_distiller(palette = "Spectral") 
``` 


- Evaluation Metric is an essential part in any Machine Learning project.
- It measures how good or bad is your Machine Learning model
- Different Evaluation Metrics are used for Regression model (Continuous output) or Classification model (Categorical output).

## 4.2 Regression model Evaluation Metrics

Now, let's do rergession -- that is, let's try to predict a numerical (continuous) outcome. We will use the `mtcars` dataset to predict miles-per-gallon from the car's weight, number of cylinders, and other variables. This time, we will do a leave-one-out: for each car, we will exclude it from the training set; use all remaining cars to train the model; and apply the model to predict the MPG of the excluded car. We will use Generalized Linear Regression (GLM) as the method of our prediction.

```r
predictions <- rep (0, dim(mtcars)[1])
for (i in 1:dim(mtcars)[1]) {
  training <- mtcars[-i,]
  testing  <- mtcars[i,] 
  train_inputs=training[,2:11]
  train_outputs=training[,1]
  test_inputs=testing[,2:11]
  model <- train(train_inputs, train_outputs, method="glm")
  predictions[i] <- predict(model,test_inputs)
}
```

Now, once we are done looping, we will get the `predictions` vector which contains the predicted value of MOG for each car. To visually see how it compares to actual MPG values, we can make a scatter plot with `qplot` function:

```r
qplot (predictions, mtcars$mpg)
```

To evaluate our prediction numerically, we can compute Pearson's correlation coefficient:

```r
cor (predictions, mtcars$mpg)
cor.test (predictions, mtcars$mpg)
```

Pretty good! However, when evaluating your prediction, never forget to visually inspect the scatter plot. It will tell you a lot more than you can get from looking at a single number of the correlation coefficient.

![image](https://en.wikipedia.org/wiki/Anscombe%27s_quartet#/media/File:Anscombe's_quartet_3.svg)



### 4.1.1 Correlation Coefficient (R) or Coefficient of Determination (R2):

![image](https://user-images.githubusercontent.com/43855029/120700259-72274900-c47f-11eb-8959-a4bbe4eafccc.png)

```r
cor(prediction,testing)
cor.test(prediction,testing)
```

### 4.1.2 Root Mean Square Error (RMSE) or Mean Square Error (MSE)

![image](https://user-images.githubusercontent.com/43855029/120700533-c5010080-c47f-11eb-8050-b1cd8c63746e.png)

The postResample function gives RMSE, R2 and MAE at the same time:

```r
postResample(prediction,testing$Ozone)

```

## 4.2. Classification model Evaluation Metrics

### 4.2.1 Confusion Matrix
- A confusion matrix is a technique for summarizing the performance of a classification algorithm.
- You can learn more about Confusion Matrix [here](https://www.analyticsvidhya.com/blog/2020/04/confusion-matrix-machine-learning/)

For binary output (classification problem with only 2 output type, also most popular):

![image](https://user-images.githubusercontent.com/43855029/120687356-efe35880-c46f-11eb-950f-5feef237a4c1.png)
 
 ```r
 confusionMatrix(predict,testing)
 ```



