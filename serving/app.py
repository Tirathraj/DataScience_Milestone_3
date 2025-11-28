"""
If you are in the same directory as this file (app.py), you can run run the app using gunicorn:

    $ gunicorn --bind 0.0.0.0:<PORT> app:app

gunicorn can be installed via:

    $ pip install gunicorn

"""
import os
from pathlib import Path
import logging

from flask import Flask, jsonify, request, abort
import sklearn
import pandas as pd
import joblib
import wandb


import ift6758


LOG_FILE = os.environ.get("FLASK_LOG", "flask.log")

MODEL = None
MODEL_ID = None
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


# app = Flask(__name__)
#
#
# @app.before_first_request
# def before_first_request():
#     """
#     Hook to handle any initialization before the first request (e.g. load model,
#     setup logging handler, etc.)
#     """
#     logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
#
#     # also log to console; source: https://stackoverflow.com/questions/14058453/making-python-loggers-output-all-messages-to-stdout-in-addition-to-log-file
#     handler = logging.StreamHandler()
#     handler.setLevel(logging.INFO)
#     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     handler.setFormatter(formatter)
#     logging.getLogger().addHandler(handler)
#
#     app.logger.info("Starting service")
#
#     global MODEL, CURRENT_MODEL
#     artifact_id = "IFT6758-2025-A10/Logistic Regression/Distance:v1"
#     try:
#         api = wandb.Api()
#         artifact = api.artifact(artifact_id, type="model")
#         artifact_dir = artifact.download()
#         model_path = Path(artifact_dir) / "log_reg_Distance.pkl"
#         MODEL = joblib.load(model_path)
#         CURRENT_MODEL = artifact_id
#
#         app.logger.info(f"Loaded {CURRENT_MODEL} model from {model_path}")
#     except Exception as e:
#         app.logger.exception(f"Error while trying to download default model {artifact_id}: {e}")
#         MODEL = None

def create_app():
    app = Flask(__name__)
    with app.app_context():
        """
        Hook to handle any initialization before the first request (e.g. load model,
        setup logging handler, etc.)
        """
        logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        # also log to console; source: https://stackoverflow.com/questions/14058453/making-python-loggers-output-all-messages-to-stdout-in-addition-to-log-file
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logging.getLogger().addHandler(handler)

        app.logger.info("Starting service")

        global MODEL, MODEL_ID
        artifact_id = "IFT6758-2025-A10/Logistic Regression/Distance:v1"
        try:
            cache_path = MODEL_DIR / "Distance_v1.pkl"
            if cache_path.exists():
                MODEL = joblib.load(cache_path)
                MODEL_ID = artifact_id
                app.logger.info(f"Loaded default model from cache: {cache_path}")
            else:
                api = wandb.Api()
                artifact = api.artifact(artifact_id, type="model")
                artifact_dir = artifact.download()
                model_path = Path(artifact_dir) / "log_reg_Distance.pkl"
                model = joblib.load(model_path)
                joblib.dump(model, cache_path)
                app.logger.info(f"Saved default model to cache: {cache_path}")

            MODEL = joblib.load(cache_path)
            MODEL_ID = artifact_id

            app.logger.info(f"Successfully loaded default model: {MODEL_ID} from {cache_path}")
        except Exception as e:
            app.logger.exception(f"Error while trying to download default model {artifact_id}: {e}")
            MODEL = None

    return app

app = create_app()

@app.route("/logs", methods=["GET"])
def logs():
    """Reads data from the log file and returns them as the response"""

    try:
        with open(LOG_FILE, "r") as f:
            lines = [line.strip() for line in f]
            response = {"logs": lines}
            app.logger.info(f"Showing logs from {LOG_FILE}")
            return jsonify(response), 200
    except Exception as e:
        app.logger.exception(f"Couldn't read log file: {e}")
        return jsonify({"logs": [], "error": str(e)}), 500

@app.route("/download_registry_model", methods=["POST"])
def download_registry_model():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/download_registry_model

    The comet API key should be retrieved from the ${COMET_API_KEY} environment variable.

    Recommend (but not required) json with the schema:

        {
            workspace: (required),
            model: (required),
            version: (required),
            ... (other fields if needed) ...
        }

    """
    # Get POST json data
    json_data = request.get_json()
    app.logger.info(json_data)

    global MODEL, MODEL_ID
    try:
        workspace = json_data["workspace"]
        model_name = json_data["model"]
        version = json_data["version"]
    except KeyError as e:
        app.logger.error(f"Missing field: {e}")
        return jsonify({"error": f"Missing field: {e}"}), 400

    artifact_name = f"{workspace}/{model_name}:{version}"
    model_path = MODEL_DIR / f"{model_name}_{version}.pkl"
    old_model, old_id = MODEL, MODEL_ID

    try:
        if model_path.exists():
            app.logger.info(f"Loading model from {model_path} ...")
            MODEL = joblib.load(model_path)
            MODEL_ID = artifact_name
            app.logger.info(f"Loaded and changed model to {MODEL_ID}")
        else:
            api = wandb.Api()
            artifact = api.artifact(artifact_name, type="model")
            app.logger.info(f"Downloading model {artifact_name} from Wandb")
            artifact_dir = artifact.download()
            pkl_files = list(Path(artifact_dir).glob("*.pkl"))
            MODEL = joblib.load(pkl_files[0])
            joblib.dump(MODEL, model_path)
            MODEL_ID = artifact_name
            app.logger.info(f"Downloaded and changed model to {MODEL_ID}")

        return jsonify({"model": MODEL_ID}), 200

    except Exception as e:
        MODEL, MODEL_ID = old_model, old_id
        app.logger.exception(f"Failed to download/load model: {e}")
        app.logger.info(f"Keeping model currently in use: {MODEL_ID}")
        return jsonify({"error": str(e), "model": MODEL_ID}), 500


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/predict

    Returns predictions
    """
    # Get POST json data
    json_data = request.get_json()
    app.logger.info(json_data)

    global MODEL, MODEL_ID
    if MODEL is None:
        app.logger.error(f"Called predict with model: {MODEL}")
        return jsonify({"error": f"Called predict with model: {MODEL}"}), 500

    X = pd.DataFrame(json_data)
    y_pred_prob = MODEL.predict_proba(X)[:,1]
    app.logger.info("Prediction done")
    return jsonify({"predictions": y_pred_prob.tolist(), "model": MODEL_ID, "features": X.columns.tolist()}), 200


#Tirath code Milestone 3
#Added main to run flask locally inside a console without gunicorn
if __name__=="__main__":
    print(f"Starting flask...")
    app.run(host="0.0.0.0", port = 8080, debug=True, use_reloader = False)
    
    #use reloading prevents flask windows bug and system exception
