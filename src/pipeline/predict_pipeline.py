import sys
import pandas as pd 
from src.exception import CustomerException
from src.utils import load_object

class predictPipepline:
    def __init__(self):
        pass
    def predict(self,features):
        try:
            model_path = 'artifacts\model.pkl'
            preprocessor_path = 'artifacts\preprocessor.pkl'
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path = preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e :
            raise CustomerException(e,sys)



class CustomData:
    def __init__(self,
                 gender :str,
                 race_ethnicity:int,
                 parental_level_of_education,
                 lunch:str,
                 test_preparation_course:str,
                 reading_score :int,
                 writing_score :int):
        self.gender = gender
        self.race_e = race_ethnicity
        self.ple = parental_level_of_education
        self.luc = lunch
        self.tps =test_preparation_course
        self.rs = reading_score
        self.ws = writing_score
        
    def get_data_as_data_frame(self):
        try:
            data = {
    "gender": [self.gender],
    "race_ethnicity": [self.race_e],
    "parental_level_of_education": [self.ple],
    "lunch": [self.luc],
    "test_preparation_course": [self.tps],
    "reading_score": [self.rs],
    "writing_score": [self.ws]
}

            return pd.DataFrame(data)
        except Exception as e:
            raise CustomerException(e,sys)
        
        