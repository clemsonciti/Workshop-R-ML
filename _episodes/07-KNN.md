---
title: "Nearest Neighbours Classification"
teaching: 20
exercises: 0
questions:
- "How to use K-Nearest Neighbour in Machine Learning model"
objectives:
- "Learn how to use KNN in ML model"
keypoints:
- "KNN"
---

## K-Nearest Neighbour

Nearest-neighbour classification is very simple and intuitive. Consider a two-dimensional graphical representation of our data:

<img src="../fig/twoclasses.png" width="300" />

The idea is: given a point (an observation), we assign it to the same class as its nearest neighbour. So there is no training, and in the testing phase, we compute the [Euclidean distances](https://en.wikipedia.org/wiki/Euclidean_distance) to all training data points and assign the observation to the same class as the training data point which is the closest. So it is really fast, but it only works on data where classes form very tight clusters:

<img src="https://user-images.githubusercontent.com/43855029/114582045-3d043480-9c4e-11eb-8698-e1c31840401a.png" width="400" />

If the clusters are less tight, the performance suffers; if the classes overlap, the performance gets really bad. This method is extremely sensitive to outliers and to noisy data.

It is possible to stabilize it, by considering not just the nearest neighbour, but a set of nearest neighbours. We can assign it to the class that gets the majority votes of 3 nearest neighbours, or 6 nearest neighbours. The number of nearest neighbours to be used is denoted with K; choise of K might impact our classification:

<img src="https://user-images.githubusercontent.com/43855029/114582319-7a68c200-9c4e-11eb-93f2-37324c034784.png" width="400" />

So, what is the best K? There is no general answer because it depends on the data. K is called the [hyperparameter](https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning)) of the classifier: 


It works really fast, but the performance is usually pretty bad. 


- Simplicity but powerful and fast for certain task
- Work for both classification and regression
- Named as Instance Based Learning; Non-parametrics; Lazy learner
- Work well with small number of inputs



### 13.1 Explanation

![image](https://user-images.githubusercontent.com/43855029/114582162-573e1280-9c4e-11eb-8a17-e0d91a38452e.png)

- In KNN, the most important parameter is the K group and the distance computed between points.
- Euclide distance:

![image](https://user-images.githubusercontent.com/43855029/114582319-7a68c200-9c4e-11eb-93f2-37324c034784.png)

### 13.2 Implementation
```r
library(caret)
data(iris)
set.seed(123)
indT <- createDataPartition(y=iris$Species,p=0.6,list=FALSE)
training <- iris[indT,]
testing  <- iris[-indT,]

ModFit_KNN <- train(Species~.,training,method="knn",preProc=c("center","scale"),tuneLength=20)

ggplot(ModFit_KNN$results,aes(k,AccuracySD))+
      geom_point(color="blue")+
      labs(title=paste("Optimum K is ",ModFit_KNN$bestTune),
           y="Error")
      
predict_KNN<- predict(ModFit_KNN,newdata=testing)
confusionMatrix(testing$Species,predict_KNN)
```
![image](https://user-images.githubusercontent.com/43855029/114583370-86a14f00-9c4f-11eb-96a0-59b3c5376952.png)

