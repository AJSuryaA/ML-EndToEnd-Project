import os
import sys
from dataclasses import dataclass
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from exception import CustomException
from logger import logging
from utils import save_object, evaluate_models

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor
)
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score

@dataclass
class ModelTrainerConfig:
    trained_model_path= os.path.join("datasplits", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.modeltrainerconfig= ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("--splitting x_train x_test y_train y_test-- --started")
            x_train, y_train, x_test, y_test= (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models= {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                # "XGBRegressor": XGBRegressor(),
                # "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }
        
            params= {
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                # "XGBRegressor":{
                #     'learning_rate':[.1,.01,.05,.001],
                #     'n_estimators': [8,16,32,64,128,256]
                # },
                # "CatBoosting Regressor":{
                #     'depth': [6,8,10],
                #     'learning_rate': [0.01, 0.05, 0.1],
                #     'iterations': [30, 50, 100]
                # },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }

            # logging.info(f"Train array shape: {train_array.shape}")
            # logging.info(f"Test array shape: {test_array.shape}")
            # logging.info(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
            # logging.info(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")
            # logging.info(f"x_train: {x_train.shape}, y_train: {y_train.shape}")
            # logging.info(f"x_test: {x_test.shape}, y_test: {y_test.shape}")



            model_report:dict= evaluate_models(x_train= x_train, y_train= y_train, x_test= x_test, y_test= y_test, models= models, params= params)

            best_model_score= max(sorted(model_report.values()))

            best_model_name= list(models.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model= models[best_model_name]

            if best_model_score < 0.65:
                raise CustomException("no best model found")
            
            logging.info("--best model found--")

            save_object(
                file_path= self.modeltrainerconfig.trained_model_path,
                obj= best_model
            )

            predicted= best_model.predict(x_test)

            r2_score_value= r2_score(y_test, predicted)
            return r2_score_value


        except Exception as e:
            raise CustomException(e, sys)
