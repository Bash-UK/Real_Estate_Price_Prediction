import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit,cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from joblib import dump,load

#Function for cross validation and Model Evaluation
def cross_val(housing_y_train, housing_tr, model_list, model_dict):
    rmse_score=[]
    for model in model_list:
        sc= cross_val_score(model,housing_tr, housing_y_train,scoring="neg_mean_squared_error",cv=10)
        sc=np.sqrt(-sc)
        rmse_score.append(sc)

    for model in model_dict:
        print(model_dict[model]+": ")
        print("Scores: ",rmse_score[model])
        print("Mean: ",rmse_score[model].mean())
        print("Standard Deviation: ",rmse_score[model].std())
        print()
    return rmse_score       

#function for choosing best model
def best_model(rmse_score,model_dict):
    lowest_error=0
    for i in range(0,3):
        print("RMSE mean: ",rmse_score[i].mean())
        if (rmse_score[i].mean() < rmse_score[lowest_error].mean()):
            lowest_error=i
            
    
    best= model_dict.get(lowest_error,"NULL")
    return best




# Start of implementation
housing=pd.read_csv("Datasets/HousingData.csv")

#Train Test Split
#Stratified Shuffled split for appropriatly distributing the records based on CHAS value 0 or 1
split=StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing,housing['CHAS']):
    strat_train_set= housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

strat_test_set.to_csv("TestSet.csv",index=False)

housing_x_train= strat_train_set.drop('MEDV',axis=1)
housing_y_train= strat_train_set["MEDV"].copy()
'''
* Using only important features identified after training the model
housing_x= strat_train_set[["LSTAT","RM","DIS","CRIM","NOX"]]
housing_labels = strat_train_set["MEDV"].copy()
'''
housing_x_test= strat_test_set.drop("MEDV",axis=1)
housing_y_test= strat_test_set["MEDV"].copy()
'''
* Using Only Important features
housing_x_test = strat_test_set[["LSTAT","RM","DIS","CRIM","NOX"]]
housing_y_test= strat_test_set["MEDV"].copy()
'''

#creating a pipeline for handling missing values and scaling the values into particular range
my_pipeline=Pipeline([
    ('imputer',SimpleImputer(strategy='median')),
    ('std_scalar',StandardScaler())
])

housing_tr= my_pipeline.fit_transform(housing_x_train)
#Model initialization
model_lr= LinearRegression()
model_DT = DecisionTreeRegressor()
model_RF = RandomForestRegressor()

#Model Training 
model_lr.fit(housing_tr,housing_y_train)
model_DT.fit(housing_tr, housing_y_train)
model_RF.fit(housing_tr,housing_y_train)
model_list=[model_lr,model_DT,model_RF]
#Model Prediction
housing_pred_LR = model_lr.predict(housing_tr)
housing_pred_DT = model_DT.predict(housing_tr)
housing_pred_RF = model_RF.predict(housing_tr)

#Model Evaluation using Cross Validation
model_dict={0:'Linear Regression',1:"Decision Tree",2:"Random Forest"}

rmse_score = cross_val(housing_y_train, housing_tr, model_list, model_dict)

#choosing the best model
best=best_model(rmse_score,model_dict)
print("Best Performance model: ",best)

#Storing the model into file 
dump(model_RF,"Price_model.joblib")
print("Model Dumped into file Price_model.joblib")


''' This is to Identify Important feature for training model 
plt.figure(figsize=(10,9))
feat_importances = pd.Series(model.feature_importances_, index = housing_x.columns)
feat_importances.nlargest(9).plot(kind='barh');
'''






