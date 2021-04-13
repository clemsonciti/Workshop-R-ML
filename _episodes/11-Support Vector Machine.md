---
title: "Support Vector Machine"
teaching: 20
exercises: 0
questions:
- "How to use Support Vector Machine in Machine Learning model"
objectives:
- "Learn how to use SVM in ML model"
keypoints:
- "SVM"
---

## Support Vector Machine
The objective of the support vector machine (SVM) algorithm is to find a hyperplane in an N-dimensional space that distinctly classifies the data points.

### Applications of Support Vector Machine:
![image](https://user-images.githubusercontent.com/43855029/114576381-1394da00-9c49-11eb-95b1-cff9d87c6029.png)

### Explanation
- To separate the two classes of data points, there are many possible hyperplanes that could be chosen

![image](https://user-images.githubusercontent.com/43855029/114577032-af264a80-9c49-11eb-8e6c-b45120743f0d.png)

- SVM's objective is to find a plane that has the maximum margin, i.e the maximum distance between data points of both classes.
Maximizing the margin distance provides some reinforcement so that future data points can be classified with more confidence.

![image](https://user-images.githubusercontent.com/43855029/114576981-a2a1f200-9c49-11eb-9921-b0bff879c97e.png)

- Example of hyperplane in 2D and 3D position:

![image](https://user-images.githubusercontent.com/43855029/114577340-eac11480-9c49-11eb-8ff9-4aa3e61b1c86.png)

- Support vectors (**SVs**) are data points that are closer to the hyperplane and influence the position and orientation of the hyperplane.
Using **SVs** to maximize the margin of the classifier.
Removing **SVs** will change the position of the hyperplane. These are the points that help us build our SVM.

![image](https://user-images.githubusercontent.com/43855029/114577489-09271000-9c4a-11eb-8b4a-b7837463288f.png)