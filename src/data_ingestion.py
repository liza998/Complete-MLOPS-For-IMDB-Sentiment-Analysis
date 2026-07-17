import pandas as pd 
import os 
import logging
import yaml
from sklearn.model_selection import train_test_split
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
        df = pd.read_csv(data_url,encoding="latin1",on_bad_lines="skip")
        logger.debug("Data Loaded from %s", data_url)
        return df
    
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise

def data_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df.drop(columns=['Unnamed: 0'], axis=1,inplace=True)
        df.rename(columns={'review':'text','sentiment': 'target'},inplace=True)
        logger.debug("Data Preprocessing is Completed")
        return df
    except KeyError as e:
        logger.error("missing columns in Dataframe", "%s", e)
        raise
    except Exception as e:
        logger.error("Unexpcted error during Preprocessing","%s",e)
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the train and test datasets."""
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)
        logger.debug('Train and test data saved to %s', raw_data_path)
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise

def main():
    try:
        params = load_params('params.yaml')
        test_size = params['data_ingestion']['test_size']
        data_path = "https://raw.githubusercontent.com/liza998/IMDB_Data/refs/heads/main/IMDB%20Dataset%20(1).csv"
        df = load_dataset(data_path)
        final_df = data_preprocessing(df)
        train_data, test_data = train_test_split(df,test_size=test_size,random_state=42)
        save_data(train_data,test_data,'./data')
        logger.debug("Successfully Completed Data Ingestion")
    except Exception as e:
        logger.error('Failed to complete the data ingestion process: %s', e)
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
