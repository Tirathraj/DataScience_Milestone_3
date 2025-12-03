import json
import requests
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class ServingClient:
    
    #def __init__(self, ip: str = "0.0.0.0", port: int = 5000, features=None):
    
    #Modified default port number to match flask port
    
    
    def __init__(self, ip: str = "127.0.0.1", port: int = 8080, features=None):
        
        self.base_url = f"http://{ip}:{port}"
        logger.info(f"Initializing client; base URL: {self.base_url}")

        if features is None:
            features = ["distance"]
        self.features = features

        # any other potential initialization
        
        #Added for Milestone 3 (Tirath)
        self.predict_url = f"{self.base_url}/predict"
        self.log_url = f"{self.base_url}/logs"
        self.registry_url = f"{self.base_url}/download_registry_model"

        #  self.predict_url = f"{self.base_url}//predict"
        # self.log_url = f"{self.base_url}//logs"
        # self.registry_url = f"{self.base_url}//download_registry_model"

        print(f"Serving Client Initialized")

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Formats the inputs into an appropriate payload for a POST request, and queries the
        prediction service. Retrieves the response from the server, and processes it back into a
        dataframe that corresponds index-wise to the input dataframe.
        
        Args:
            X (Dataframe): Input dataframe to submit to the prediction service.
        """

        #raise NotImplementedError("TODO: implement this function")

        #Convert df to list of dicts for JSON
        json_data = X.to_dict(orient = "records")

        #Send request to flask and try to capture response
        try:
            response = requests.post(self.predict_url, json = json_data)
        except Exception as e:
            print(f"Server Error: {e}")
            raise RuntimeError()

        if(response.status_code !=200):
            print(f"Error code {response.status_code}: {response.text}")
            raise RuntimeError()

        #Convert json response to dict
        response_dict = response.json()

        predictions = response_dict['predictions']
        return predictions

    def logs(self) -> dict:
        """Get server logs"""
        #raise NotImplementedError("TODO: implement this function")

        try:
            response = requests.get(self.log_url)

        except Exception as e:
            print(f"Log Error: {e}")
            raise RuntimeError()

        if(response.status_code !=200):
            print(f"Error code {response.status_code}: {response.text}")
            raise RuntimeError()

        log_dict = response.json()
        return log_dict
        

    def download_registry_model(self, workspace: str, model: str, version: str) -> dict:
        """
        Triggers a "model swap" in the service; the workspace, model, and model version are
        specified and the service looks for this model in the model registry and tries to
        download it. 

        See more here:

            https://www.comet.ml/docs/python-sdk/API/#apidownload_registry_model
        
        Args:
            workspace (str): The Comet ML workspace
            model (str): The model in the Comet ML registry to download
            version (str): The model version to download
        """
        #raise NotImplementedError("TODO: implement this function")

        #Send POST Request to Flask
        json_data = {
            "workspace" : workspace,
            "model" : model,
            "version" : version
        }
        
        try:
            response = requests.post(self.registry_url, json = json_data)

        except Exception as e:
            print(f"Model Registry Error: {e}")
            raise RuntimeError()

        if(response.status_code !=200):
            print(f"Error code {response.status_code}: {response.text}")
            raise RuntimeError()

        response_dict = response.json()
        model_name = response_dict["model"]
        print(f"Model switched to: {model_name}")
        
        return response_dict        
