import os
import sys
from dataclasses import dataclass

import pandas as pd, numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from exception import CustomException
from logger import logging
from utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path= os.path.join('datasplits',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config= DataTransformationConfig()

    def get_data_transformer_object(self):
        
        '''
        this fun is responsible for data transformation
        '''

        try:
            numerical_columns= ["writing_score","reading_score"]
            categorical_columns=[
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]

            nummerical_pipeline= Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy= "median")),
                    ("scaler", StandardScaler())
                ]
            )

            categorical_pipeline= Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy= "most_frequent")),
                    ("encoder", OneHotEncoder()),
                    # ("scaler", StandardScaler(with_mean= False))
                ]
            )

            logging.info(f"categorical columns : {categorical_columns}")
            logging.info(f"numeric columns : {numerical_columns}")

            preprocessor= ColumnTransformer(
                [
                    ("numerical_pipeline", nummerical_pipeline, numerical_columns),
                    ("categorical_pipelines", categorical_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df= pd.read_csv(train_path)
            test_df= pd.read_csv(test_path)

            logging.info("--read_csv (train_path, test_path)-- --completed--")

            logging.info("--obtaining preprocessing obj-- --started--")
            preprocessing_obj= self.get_data_transformer_object()
            logging.info("--obtaining preprocessing obj-- --completed--")

            target_column= "math_score"
            numerical_columns= ["writing_score","reading_score"]

            input_feature_train_df= train_df.drop(columns=[target_column], axis= 1)
            target_feature_train_df= train_df[target_column]

            input_feature_test_df= test_df.drop(columns=[target_column], axis= 1)
            target_feature_test_df= test_df[target_column]

            logging.info("--applying preprocessing obj on train and test df-- --started--")
            # logging.info(f"Type of data: {type(input_feature_test_df)}")


            input_feature_train_array= preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_array= preprocessing_obj.transform(input_feature_test_df)

            train_array= np.c_[
                input_feature_train_array, np.array(target_feature_train_df)
            ]

            test_array= np.c_[
                input_feature_test_array, np.array(target_feature_test_df)
            ]

            logging.info("--applying preprocessing obj on train and test df-- --completed--")
            logging.info("--save preeprocessing object-- --started--")

            save_object(
                file_path= self.data_transformation_config.preprocessor_obj_file_path,
                obj= preprocessing_obj
            )

            logging.info("--save preeprocessing object-- --completed--")

            return(
                train_array,
                test_array,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)