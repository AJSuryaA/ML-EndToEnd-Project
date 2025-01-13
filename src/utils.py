import os
import sys
import dill
from exception import CustomException

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

import pandas as pd, numpy as np

def save_object(file_path, obj):
    try:
        dir_path= os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok= True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_path)

    except Exception as e:
        return CustomException(e, sys)
    

def evaluate_models(x_train, y_train, x_test, y_test, models, params):
    try:
        report= {}

        for i in range(len(models)):
            model= list(models.values())[i]
            param= params[list(models.keys())[i]]

            gs= GridSearchCV(model, param, cv= 3)
            gs.fit(x_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(x_train,y_train)

            y_train_pred= model.predict(x_train)
            y_pred_test= model.predict(x_test)

            train_model_score= r2_score(y_train,y_train_pred)
            test_model_score= r2_score(y_test,y_pred_test)

            report[list(models.keys())[i]]= test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
    

