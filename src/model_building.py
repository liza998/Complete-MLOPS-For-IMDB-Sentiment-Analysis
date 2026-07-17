import pandas as pd 
import os 
import logging
import yaml
import numpy as np 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
import re
import pickle

log_dir = 'logs'
os.makedirs(log_dir,exist_ok=True)

logger = logging.getLogger('Data Ingestion')
logger.setLevel("DEBUG")

console = logging.StreamHandler()
console.setLevel("DEBUG")

filepath = os.path.join(log_dir,"model_building.py")
file =logging.FileHandler(filepath)
file.setLevel("DEBUG")

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
file.setFormatter(formatter)

logger.addHandler(console)
logger.addHandler(file)

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise

def load_dataset(data_url : str) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_url)
        logger.debug("Data Loaded from %s", data_url)
        return df
    
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise


def model_train(X_train: np.ndarray, y_train: np.ndarray, params: dict):
    trained_models = {}
    try:
        if  X_train.shape[0] != y_train.shape[0]:
            raise ValueError("The number of samples in X_train and y_train must be the same.")
        clf = {
    'LogisticRegression': LogisticRegression(),
    'MultinomialNB': MultinomialNB(**params['model_building']['MultinomialNB']),
    'DecisionTreeClassifier': DecisionTreeClassifier(**params['model_building']['DecisionTreeClassifier']),
    'KNeighborsClassifier': KNeighborsClassifier(**params['model_building']['KNeighborsClassifier']),
    'RandomForestClassifier': RandomForestClassifier(**params['model_building']['RandomForestClassifier']),
    'AdaBoostClassifier': AdaBoostClassifier(**params['model_building']['AdaBoostClassifier']),
    #'GradientBoostingClassifier': GradientBoostingClassifier(**params['model_building']['GradientBoostingClassifier'])
}
    
        for name, model in clf.items():
            print(f'Model Building using the {name}')
            model.fit(X_train,y_train)
            trained_models[name] = model
        return trained_models
            
    except ValueError as e:
        logger.error('ValueError during model training: %s', e)
        raise
    except Exception as e:
        logger.error('Error during model training: %s', e)
        raise



def save_model(model, file_path: str)-> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path,"wb") as file:
            pickle.dump(model ,file)
            logger.debug("Model is Successfully Stored")
    except FileNotFoundError as e:
        logger.error('File path not found: %s', e)
        raise
    except Exception as e:
        logger.error('Error occurred while saving the model: %s', e)
        raise






def main():
    try:
        #train_models = {}
        params = load_params('params.yaml')
        train_data = load_dataset('./Feature_Dataset_3/Feature/train_tfidf.csv')
        X_train = train_data.iloc[:,:-1].values
        y_train = train_data.iloc[:,-1].values
        train_models = model_train(X_train,y_train, params)
        for name, model in train_models.items():
            model_path = f'./models/{name}/{name}.pkl'
            save_model(model,model_path)  
        logger.debug("All Models are saved Successfully")
        
    except Exception as e:
        logger.error("Failed to Complete Feature Engineering process %s", e)
        print(f"error: {e}")



        
if __name__ == "__main__":
    main()
