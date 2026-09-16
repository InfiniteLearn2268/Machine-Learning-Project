import os 
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRFRegressor
from src.exception import CustomerException
from src.logger import logging
from src.utils import save_object
from src.utils import evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            print("Starting model training...")
            logging.info("Split training and test input data")
            X_train,y_train ,X_test,y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            models = {
                "Random Forest":RandomForestRegressor(),
                "Decision Tree":DecisionTreeRegressor(),
                "Gradient Boosting":GradientBoostingRegressor(),
                "Linear Regression":LinearRegression(),
                "K-Neigbors Regressor":KNeighborsRegressor(),
                "XGB Regressor":XGBRFRegressor(),
                "CatBoosting Regressor":CatBoostRegressor(verbose=False),
                "Adaboost Regressor":AdaBoostRegressor()
            }
            
            params = {
    
    "Random Forest": {
        "n_estimators": [50, 100, 200, 300],
        "criterion": ["squared_error", "absolute_error", "friedman_mse", "poisson"],
        "max_depth": [None, 5, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
        "bootstrap": [True, False]
    },

    "Decision Tree": {
        "criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"],
        "splitter": ["best", "random"],
        "max_depth": [None, 5, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
        "max_leaf_nodes": [None, 10, 20, 50]
    },

    "Gradient Boosting": {
        "loss": ["squared_error", "huber", "absolute_error", "quantile"],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "n_estimators": [50, 100, 200, 300],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "criterion": ["friedman_mse", "squared_error"],
        "max_depth": [3, 5, 7, 10],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None]
    },

    "Linear Regression": {
        "fit_intercept": [True, False],
        "positive": [True, False]
    },

    "K-Neighbors Regressor": {
        "n_neighbors": [3, 5, 7, 9, 11],
        "weights": ["uniform", "distance"],
        "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
        "leaf_size": [10, 20, 30, 40, 50],
        "p": [1, 2]
    },

    "XGB Regressor": {
        "n_estimators": [50, 100, 200, 300],
        "max_depth": [3, 5, 7, 10],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "min_child_weight": [1, 3, 5, 7],
        "gamma": [0, 0.1, 0.2, 0.5],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "reg_alpha": [0, 0.01, 0.1, 1],
        "reg_lambda": [1, 1.5, 2, 5]
    },

    "CatBoosting Regressor": {
        "iterations": [100, 200, 300, 500],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "depth": [4, 6, 8, 10],
        "l2_leaf_reg": [1, 3, 5, 7, 10],
        "loss_function": ["RMSE", "MAE"],
        "subsample": [0.7, 0.8, 0.9, 1.0]
    },

    "Adaboost Regressor": {
        "n_estimators": [50, 100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0],
        "loss": ["linear", "square", "exponential"]
    }
}
            model_report:dict = evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,params=params)
            
            # To get best model score from dict
            best_model_score = max(sorted(model_report.values()))
            
            # To get best model name from dict 
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            print("Model training completed")
            print("Model Report:", model_report)
            
            best_model = models[best_model_name]
            
            print("Best Model:", best_model_name)
            print("Best Score:", best_model_score)
            
            if best_model_score < 0.6:
                raise CustomerException("No best model founded ...")
            logging.info(f"Best found model on both training and testing dataset")
            
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test,predicted)
            print("Final R2 Score:", r2_square)
            return r2_square
        
        
        except Exception as e:
            raise CustomerException(e,sys)
        