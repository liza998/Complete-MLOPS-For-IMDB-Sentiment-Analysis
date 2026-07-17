import pandas as pd 
import os 
import logging
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
import re

log_dir = 'logs'
os.makedirs(log_dir,exist_ok=True)

logger = logging.getLogger('Data Ingestion')
logger.setLevel("DEBUG")

console = logging.StreamHandler()
console.setLevel("DEBUG")

filepath = os.path.join(log_dir,"data_ingestion.py")
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

def applytfidfVectorize(train_data : pd.DataFrame, test_data: pd.DataFrame,  max_features: int)-> tuple:
    try:
        vector = TfidfVectorizer(max_features=max_features)
        X_train = train_data["text"].values
        X_test = test_data["text"].values
        y_train = train_data["target"].values
        y_test = test_data["target"].values


        X_train_bow = vector.fit_transform(X_train)
        X_test_bow = vector.transform(X_test)
        print(f"Test shape is {X_test_bow.shape}")
        
        train_df = pd.DataFrame(X_train_bow.toarray())
        test_df = pd.DataFrame(X_test_bow.toarray())
        train_df["Label"] = y_train
        test_df["Label"] = y_test

        logger.debug("Successfully Created Features Dataset Using TfIdfVectrizer")
        return train_df,test_df
    
    except Exception as e:
        logger.error('Error during Bag of Words transformation: %s', e)
        raise



def save_data(df: pd.DataFrame, filepath: str) -> None:
    try: 
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath,index=False)
        logger.debug("Data Saved to %s", filepath)
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise






def main():
    try:
        params = load_params('params.yaml')
        max_features = params['feature_engineering']['max_features']
        train_data = load_dataset('./Process_Data/Processed/train_processed_data.csv')
        test_data = load_dataset('./Process_Data/Processed/test_processed_data.csv')
        feature_train_data,feature_test_data = applytfidfVectorize(train_data,test_data,max_features=max_features)
        # Save Data
        dir = "Feature_Dataset_3"
        os.makedirs(dir, exist_ok=True)
        save_data(feature_train_data,os.path.join('./Feature_Dataset_3','Feature','train_tfidf.csv'))
        save_data(feature_test_data,os.path.join('./Feature_Dataset_3','Feature','test_tfidf.csv'))
    except Exception as e:
        logger.error("Failed to Complete Feature Engineering process %s", e)
        print(f"error: {e}")



        
if __name__ == "__main__":
    main()
