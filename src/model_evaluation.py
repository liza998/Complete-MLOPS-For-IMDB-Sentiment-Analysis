import pandas as pd 
import os 
import logging
import yaml
import numpy as np 
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt 
import seaborn as sns
import pickle 
import json 
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score,confusion_matrix,classification_report
import re
from pathlib import Path
import pickle
import dagshub
import dvclive 
from  dvclive import Live
import os 

import dagshub
dagshub.init(repo_owner='lizarizwana65', repo_name='Complete-MLOPS-For-IMDB-Sentiment-Analysis', mlflow=True)
mlflow.set_experiment("BestModel Hypertune_experiment")

log_dir = 'logs'
os.makedirs(log_dir,exist_ok=True)

logger = logging.getLogger('Model Evaluation')
logger.setLevel("DEBUG")

console = logging.StreamHandler()
console.setLevel("DEBUG")

filepath = os.path.join(log_dir,"model_evaluation.py")
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

def load_model(data_url : str) -> pd.DataFrame:
    try:
        with open(data_url, "rb") as file:
            model = pickle.load(file)
        logger.debug("Model is Loaded successfully")
        return model
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise


def model_evaluation(model, X_test: np.ndarray, y_test: np.ndarray,model_path):
    try:
        model_name = model_path.parent.name
        y_pred = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:,1]
        accuracy = accuracy_score(y_test,y_pred)
        precision = precision_score(y_test,y_pred)
        recall = recall_score(y_test,y_pred)
        f1score = f1_score(y_test,y_pred)
        Confusion_matrix = confusion_matrix(y_test,y_pred)
        Classification_report =classification_report(y_test,y_pred)
        metrics_dict ={
            "Model": model_name,
            "accuracy": accuracy,
            "precision": precision,
            "recall" : recall,
            "f1_score": f1score,

        }
        logger.debug("All metrics are loaded successfully")
        return metrics_dict,Confusion_matrix,Classification_report
        
     
        
    except ValueError as e:
        logger.error('ValueError during model evaluation: %s', e)
        raise
    except Exception as e:
        logger.error('Error during model evaluation: %s', e)
        raise



def savemetrics(metrics_dict: dict, filepath: str):
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as file:
            json.dump(metrics_dict,file)
        logger.debug("Metrics are saved %s", filepath)
    except Exception as e:
        logger.error('Error occurred while saving the metrics: %s', e)
        raise






def main():
    try:
        #train_models = {}
        params = load_params('params.yaml')
        test_data = load_dataset('./Feature_Dataset_3/Feature/test_tfidf.csv')
        X_test = test_data.iloc[:,:-1].values
        y_test = test_data.iloc[:,-1].values
        for model_path in Path("./models").rglob("*.pkl"):
            model = load_model(model_path)
            metrics_dict,Confusion_matrix,Classification_report = model_evaluation(model,X_test,y_test,model_path)
            # Skip models with accuracy < 0.80
            if metrics_dict["accuracy"] < 0.80:
                logger.info(f"Skipping {metrics_dict['Model']} because accuracy is {metrics_dict['accuracy']:.4f}")
                continue
            os.makedirs("report", exist_ok=True)
            savemetrics(metrics_dict,"./report/BestModel.json")
            model_name = metrics_dict["Model"]
            with mlflow.start_run():
                mlflow.log_param("Model", model_name)
                mlflow.log_params(params["model_building"][model_name])
            for name, metrics in metrics_dict.items():
                if  name != 'Model':
                    mlflow.log_metric(name, metrics)
            cm = Confusion_matrix
            plt.figure(figsize=(10,12))
            sns.heatmap(cm, annot=True, cmap='Blues')
            plt.ylabel("Actual")
            plt.xlabel("Predict")
            plt.title("Confusion Matrix")
            print("confusion Matrix is created") 
            plt.savefig("confusion_matrix.png")
            print("saved into pngformat")
            mlflow.log_artifact("confusion_matrix.png")
            with open("classification_report.txt","w") as file:
                file.write(Classification_report)
            mlflow.log_artifact("classification_report.txt")
            print("Confusion matrix is Created")
            mlflow.log_artifact(__file__)
        
            with Live(save_dvc_exp=True) as live:
                for name,metrics in metrics_dict.items():
                    live.log_metric(name,metrics)
                    live.log_params(params["model_building"][model_name])

            
            
        
        
    except Exception as e:
        logger.error("Failed to Complete Feature Engineering process %s", e)
        print(f"error: {e}")



        
if __name__ == "__main__":
    main()
