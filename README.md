# Real_Estate_Price_Prediction

## Overview
In this project, I have applied basic machine learning concepts on data collected for housing prices in Boston, Massachusetts area to predict the selling price of a new home.
First, I have explored the data to obtain important features and descriptive statistics about the dataset, I used `correlation matrix` to identify highly correlated features. Also I used `Pipeline` in preprocessing the data for handling missing values using `Imputer` and for Scaling the parameters using `Standard Scalar`.
Next, I split the data into testing and training subsets, by using `StratifiedShuffleSplit` for appropriately distributing data between training and testing set
and determined a suitable performance metric for this problem as `RMSE` and `K fold Cross validation`. 
Then analyzed performance graphs for a learning algorithm with varying parameters and training set sizes. 
This enables me to pick the optimal model that best generalizes for unseen data. I have used `Linear Regression`,`Decision Tree` and `Random Forest`, among which `Random Forest Regressor` performed better then others. I have stored this model in a file using `joblib` module.
Finally, I have tested this optimal model on test dataset and predicted the selling price.

## Highlights
This project is designed to get acquainted to working with datasets in Python and applying basic machine learning techniques using NumPy and Scikit-Learn. Before being expected to use many of the available algorithms in the sklearn library, I personally get to learn a lot from this project including different Data Science and ML concept

Things I have learned by completing this project:
- How to use NumPy and Pandas to investigate the latent features of a dataset.
- How to analyze various attributes by performing EDA.
- Choosing the best Model based on Model Evaluation Techinique
- Storing the Model for later use
- Using pipelines

and many more

## Software and Libraries
This project uses the following software and Python libraries:

- [Python](https://www.python.org/download/releases/3.0/)
- [NumPy](http://www.numpy.org/)
- [pandas](http://pandas.pydata.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [matplotlib](http://matplotlib.org/)
- [Jupyter notebook](https://jupyter.org/)
- [Joblib](http://pypi.python.org/pypi/joblib)

## Project
- Consist of four important file:

- `House Pricing.ipynb` : This is a actual jupyter Notebook in which I have done all the analysis and and model building.
- `HousingData.csv ` : This is project dataset in Dataset DIR
- `Real_Estate.py ` : Final code without analysis
- `Model Usage.py ` : Code for using trained model on Test dataset and different inputs

In terminal or command prompt, navigate to folder containing project file and then use the command `jupyter notebook House Pricing.ipynb` to open up a browser window or tab to work with your notebook.
or you can simply use `Real_Estate.py` and `Model_Usage.py` without analysis.
