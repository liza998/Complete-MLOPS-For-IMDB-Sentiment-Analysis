import pandas as pd 
import os 
import logging
import yaml
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
import re
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("punkt")
import string
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

def transform(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+'," ",text)
    text = re.sub(r'@\w+|#\w+'," ",text)
    text = "".join([char for char in text if char.isalpha() or char.isdigit() or char.isspace()])
    Stopwords = stopwords.words("english")
    lemmatizer = WordNetLemmatizer ()
    word = text.split()
    words = [lemmatizer.lemmatize(w) for w in word if w not in Stopwords]
    text = " ".join(words)
    return text

def preprocessing(df, text_column= 'text', target_column='targer'):
    try:
        logger.debug("Starting Preprocessing")
        lb = LabelEncoder()
        df[target_column] = lb.fit_transform(df[target_column])
        logger.debug("Label Encoding is Complete")
        df = df.drop_duplicates(keep='first')
        logger.debug("Duplicates are removed")
        df.loc[:,text_column] = df[text_column].apply(transform)
        logger.debug("Data Transformation is Complete")
        return df
    except KeyError as e: 
        logger.error('Column not found: %s', e)
        raise
    except Exception as e: 
        logger.error('Error during text normalization: %s', e)
        raise

def main(text_column='text',target_column='target'):
    try:
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logger.debug("Train & Test Data are loaded Successfully")
        train_processed_data = preprocessing(train_data,text_column,target_column)
        test_processed_data = preprocessing(test_data,text_column,target_column)
        # Save Data
        dir = "Process_Data"
        os.makedirs(dir, exist_ok=True)
        data_path = os.path.join('./Process_Data', 'Processed')
        os.makedirs( data_path,exist_ok=True)
        train_processed_data.to_csv(os.path.join(data_path,'train_processed_data.csv'), index=False)
        test_processed_data.to_csv(os.path.join(data_path,'test_processed_data.csv'), index=False)
        logger.debug("Processed Data Save to %s", data_path)
    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
    except pd.errors.EmptyDataError as e:
        logger.error('No data: %s', e)
    except Exception as e:
        logger.error('Failed to complete the data transformation process: %s', e)
        print(f"Error: {e}")






    
if __name__ == "__main__":
    main()
