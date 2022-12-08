from joblib import dump,load
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

test_set= pd.read_csv("TestSet.csv")

test_set_x=test_set.drop('MEDV',axis=1)
test_set_y=test_set["MEDV"].copy()

model = load("Price_model.joblib")
test_pred=model.predict(test_set_x)

mse=mean_squared_error(test_set_y,test_pred)
rmse= np.sqrt(mse)
print("Root Mean Squared Error for test set: ",rmse)

#checking the price prediction with different input 
print("checking the price prediction with different input.......")
test_input= np.array([[0.03498 ,0.0 , 1.89 ,0, 0.518 , 6.540 ,49.7 ,6.2669, 0, 452, 25.9, 289.96, 7.65]])
pred=model.predict(test_input)
pred= float(pred[0]*1000)
print("Given Input: ",test_input)
print(f"Price of House for given input is: "+"{:.2f}".format(pred)+" $")


'''
OUTPUT

Root Mean Squared Error for test set:  8.26983678799044
Given Input:  [[3.4980e-02 0.0000e+00 1.8900e+00 0.0000e+00 5.1800e-01 6.5400e+00
 4.9700e+01 6.2669e+00 0.0000e+00 4.5200e+02 2.5900e+01 2.8996e+02  7.6500e+00]]

Price of House for given input is: 23697.00 $

'''