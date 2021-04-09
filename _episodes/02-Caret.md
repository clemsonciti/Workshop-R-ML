---
title: "Introduction to Caret"
teaching: 40
exercises: 0
questions:
- "What is Caret"
objectives:
- "Master Caret package for Machine Learning"
keypoints:
- "Caret"
---

## What is Caret
The caret package (short for **C**lassification **A**nd **RE**gression **T**raining) is a set of functions that attempt to streamline the process for creating predictive models. The package contains tools for:

data splitting
pre-processing
feature selection
model tuning using resampling
variable importance estimation
as well as other functionality.

There are many different modeling functions in R. Some have different syntax for model training and/or prediction. The package started off as a way to provide a uniform interface the functions themselves, as well as a way to standardize common tasks (such parameter tuning and variable importance).

The current release version can be found on CRAN and the project is hosted on github.
Caret was developed by [Max Kuhn](https://topepo.github.io/caret/index.html)
Here only touch some of the very basic command that is useful for our Machine Learning class.

## Install Caret
In R console:
```r
install.packages("caret", dependencies = c("Depends", "Suggests"))
```
In R studio:
```
Select Tools\Install Packages and select caret from CRAN
```
Once installed, load the caret package to make sure that it works:
```r
library(caret)
```
