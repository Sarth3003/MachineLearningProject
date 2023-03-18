import sys
import os
import pandas as pd
import numpy as np
from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.loggers import logging
from src.exception import CustomException
from src.utils import  save_object


@dataclass
class DataTransforamtionConfig:
    preprocessor_obj_file_path = os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransforamtionConfig()

    def get_data_transformer_object(self):
        '''
        This function will do the data transformation   
        '''

        try:
            numerical_columns = ['reading_score', 'writing_score']
            categorical_columns =   [
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
                ]
            
            nummerical_pipeline = Pipeline(
                steps=[
                    ("imputation", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )
            
            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info("Numerical columns pipeline created")

            
            categorical_pipeline = Pipeline(
                steps=[
                ("imputation", SimpleImputer(strategy="most_frequent")),
                ("encoding", OneHotEncoder()),
                ("Scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info("Categorical columns pipeline created")
            preprocessor = ColumnTransformer(
                [
                ("num_pipeline",nummerical_pipeline ,numerical_columns),
                ("cat_pipeline",categorical_pipeline ,categorical_columns)
                ]
            )
            logging.info("preprocessor object created")

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)    
        

    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("test and train data importing completed")
            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column = "math_score"
            numerical_columns = ['reading_score', 'writing_score']

            input_feature_train_df = train_df.drop(columns=[target_column], axis =1)
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = train_df.drop(columns=[target_column], axis =1)
            target_feature_test_df = train_df[target_column]

            logging.info("Applying preprocessing object on training dataframe and testing dataframe.")

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e,sys)    
