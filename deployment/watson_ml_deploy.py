"""
watson_ml_deploy.py
--------------------
IBM Watson Machine Learning deployment pipeline for the Credit Card Approval
Prediction model. This pushes the trained scikit-learn model bundle
(model/model.pkl) to IBM Watson Machine Learning so it can be hosted on the
cloud and called for scalable, real-time predictions.

Fill in the placeholders below with your IBM Cloud credentials, then run:
    python deployment/watson_ml_deploy.py

Requires:
    pip install ibm-watson-machine-learning
"""

import pickle

from ibm_watson_machine_learning import APIClient

# ---------------------------------------------------------------------
# 1. IBM Cloud credentials -- fill these in
# ---------------------------------------------------------------------
WML_CREDENTIALS = {
    "url": "https://us-south.ml.cloud.ibm.com",   # region endpoint
    "apikey": "<YOUR_IBM_CLOUD_API_KEY>",
}
SPACE_ID = "<YOUR_DEPLOYMENT_SPACE_ID>"
MODEL_PATH = "model/model.pkl"
DEPLOYMENT_NAME = "credit-card-approval-prediction"

def main():
    client = APIClient(WML_CREDENTIALS)
    client.set.default_space(SPACE_ID)

    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    sk_model = bundle["model"]

    print(f"Deploying model: {bundle['model_name']}")

    # Choose a scikit-learn software specification compatible with your
    # installed scikit-learn version. List available specs with:
    #   client.software_specifications.list()
    sofware_spec_uid = client.software_specifications.get_id_by_name("runtime-23.1-py3.10")

    meta_props = {
        client.repository.ModelMetaNames.NAME: DEPLOYMENT_NAME,
        client.repository.ModelMetaNames.TYPE: "scikit-learn_1.1",
        client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: sofware_spec_uid,
    }

    published_model = client.repository.store_model(
        model=sk_model, meta_props=meta_props
    )
    model_uid = client.repository.get_model_id(published_model)
    print(f"Model stored in WML with ID: {model_uid}")

    deployment_meta = {
        client.deployments.ConfigurationMetaNames.NAME: f"{DEPLOYMENT_NAME}-deployment",
        client.deployments.ConfigurationMetaNames.ONLINE: {},
    }
    deployment = client.deployments.create(model_uid, meta_props=deployment_meta)
    deployment_uid = client.deployments.get_id(deployment)

    scoring_url = client.deployments.get_scoring_href(deployment)
    print(f"Deployment created: {deployment_uid}")
    print(f"Scoring endpoint: {scoring_url}")
    print("\nUse this endpoint from app.py (or any client) for real-time, "
          "cloud-hosted predictions instead of the local model.pkl file.")


if __name__ == "__main__":
    main()
