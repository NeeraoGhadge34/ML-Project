import sys
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts",'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_obj(self):
        try:
            num_feature = ['reading_score','writing_score']
            cat_feature = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']       
                   
            num_pipeline = Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy='median')),
                ("scaler",StandardScaler())
                ]
            )     

            cat_pipeline = Pipeline(
                steps= [
                ("imputer",SimpleImputer(strategy='most_frequent')),
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
                ]
            )
            
            logging.info(f"Numerical features: {num_feature}")
            logging.info(f"Categorical features: {cat_feature}")

            preprocessor = ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,num_feature),
                ("cat_pipeline",cat_pipeline,cat_feature)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("reading train and test file is completed")

            logging.info("obtaining preprocessing object")
            preprocessing_object = self.get_data_transformer_obj()

            target_feature = 'math_score'

            input_feature_train_df = train_df.drop(columns=[target_feature],axis=1)
            input_feature_test_df = test_df.drop(columns=[target_feature],axis=1)

            target_feature_train_df = train_df[target_feature]         
            target_feature_test_df = test_df[target_feature]

            logging.info("Applying preprocessing object on train and test dataframe")
            input_feature_train_arr = preprocessing_object.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_object.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
            
            logging.info(f"Saved preprocessing object.")

            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path,
                        obj=preprocessing_object)

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        
        except Exception as e:
            raise CustomException(e,sys)