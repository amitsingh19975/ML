# Machine Learning Library in C++

I'm just learning ML and this is a immature library.  I'm slowly building it and learning about model. This lib uses **Eigen** for linear algebra and no other external lib. I created **Frame** similar to **Python Pandas**, **PrettyPrinter** for printing Frame, vector, Eigen, **CSVParser** for parsing csv data, etc.


# Libraries

- **Frame** for data manipulation
- **PrettyPrinter** for printing frame data, vector, Eigen
- **CSVParser** for parsing csv data and directly convert it to Frame
- **PCA** for dimensionality reduction
- **Series**
- **Metric** for confusion matrix
- **Core** base class of model
-	## ML
	* **PolyRegression**
	* **LinearRegression**
	* **LogisticRegression**
	* **KMean** working on it!

# Functions and Members

## Core

- ``virtual void train(Frame* X_train, Frame* y_train)``
-  ``virtual void train(std::shared_ptr<Frame>& X_train, std::shared_ptr<Frame>& y_train)``
- ``virtual Frame* predict(Frame* X_test)``
-  ``Frame* predict(std::shared_ptr<Frame>& X_test)``
## Frame

## Series

## Pretty Printer

## PCA

## CSVParser

## PolyRegression

## LinearRegression

## LogisticRegression

## KMean

