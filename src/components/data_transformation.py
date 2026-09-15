import sys
from dataclasses import dataclass
import os

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomerException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join(
        "artifacts", "preprocessor.pkl"
    )


class DataTransformation:

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        This function is responsible for data transformation.
        """

        try:

            numerical_col = [
                "writing_score",
                "reading_score"
            ]

            categorical_col = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
               "test_preparation_course"
            ]

            # Numerical pipeline
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            # Categorical pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "one_hot_encoder",
                        OneHotEncoder(
                            handle_unknown="ignore",
                            sparse_output=False
                        )
                    ),
                    ("scaler", StandardScaler())
                ]
            )

            logging.info("Numerical Columns encoding completed")
            logging.info("Categorical Columns encoding completed")

            # Column Transformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_col),
                    ("cat_pipeline", cat_pipeline, categorical_col)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomerException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):

        try:

            # Read train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and test data reading completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_columns_name = "math_score"

            # Separate input and target features
            input_feature_train_df = train_df.drop(
                columns=[target_columns_name]
            )

            target_feature_train_df = train_df[target_columns_name]

            input_feature_test_df = test_df.drop(
                columns=[target_columns_name]
            )

            target_feature_test_df = test_df[target_columns_name]

            logging.info(
                "Applying preprocessing object on training "
                "and testing dataframe."
            )

            # Fit on training data and transform training data
            input_feature_train_arr = preprocessing_obj.fit_transform(
                input_feature_train_df
            )

            # Transform testing data
            input_feature_test_arr = preprocessing_obj.transform(
                input_feature_test_df
            )

            # Add target column to transformed data
            train_arr = np.c_[
                input_feature_train_arr,
                np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr,
                np.array(target_feature_test_df)
            ]

            logging.info("Saved preprocessing object")

            # Save preprocessing object
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomerException(e, sys)