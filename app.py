"""
app.py
------
Flask web application for the Credit Card Approval Prediction project.

Loads the trained model bundle (model/model.pkl) and serves a simple
form-based UI where a credit analyst / compliance officer / prospective
customer can submit an applicant's profile and instantly receive an
approve / reject prediction with a confidence score.
"""

import pickle

import numpy as np
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

with open("model/model.pkl", "rb") as f:
    BUNDLE = pickle.load(f)

MODEL = BUNDLE["model"]
SCALER = BUNDLE["scaler"]
FEATURE_COLS = BUNDLE["feature_cols"]
NUMERIC_COLS = BUNDLE["numeric_cols"]
CATEGORICAL_COLS = BUNDLE["categorical_cols"]
REQUIRES_SCALING = BUNDLE["requires_scaling"]
MODEL_NAME = BUNDLE["model_name"]

INCOME_TYPES = ["Working", "Commercial associate", "Pensioner", "State servant", "Student"]
EDUCATION_TYPES = ["Secondary / secondary special", "Higher education",
                    "Incomplete higher", "Lower secondary", "Academic degree"]
FAMILY_STATUSES = ["Married", "Single / not married", "Civil marriage", "Separated", "Widow"]
HOUSING_TYPES = ["House / apartment", "With parents", "Municipal apartment",
                  "Rented apartment", "Office apartment", "Co-op apartment"]
OCCUPATION_TYPES = ["Laborers", "Core staff", "Sales staff", "Managers", "Drivers",
                     "High skill tech staff", "Accountants", "Medicine staff",
                     "Security staff", "Cooking staff", "Cleaning staff", "Other"]


def build_feature_row(form):
    """Convert raw form input into the one-hot encoded feature row the
    model expects, matching the training-time feature engineering."""

    row = {col: 0 for col in FEATURE_COLS}

    row["CODE_GENDER"] = 1 if form["gender"] == "M" else 0
    row["FLAG_OWN_CAR"] = 1 if form["own_car"] == "Y" else 0
    row["FLAG_OWN_REALTY"] = 1 if form["own_realty"] == "Y" else 0
    row["CNT_CHILDREN"] = int(form["children"])
    row["AMT_INCOME_TOTAL"] = float(form["income"])
    row["FLAG_WORK_PHONE"] = 1 if form.get("work_phone") == "on" else 0
    row["FLAG_PHONE"] = 1 if form.get("phone") == "on" else 0
    row["FLAG_EMAIL"] = 1 if form.get("email") == "on" else 0
    row["CNT_FAM_MEMBERS"] = int(form["fam_members"])
    row["PAST_DUE_COUNT"] = int(form["past_due_count"])
    row["CREDIT_INQUIRIES_6M"] = int(form["credit_inquiries"])
    row["EXISTING_LOANS"] = int(form["existing_loans"])
    row["ACCOUNT_AGE_MONTHS"] = int(form["account_age_months"])
    row["AGE_YEARS"] = float(form["age"])
    row["EMPLOYED_YEARS"] = float(form["employed_years"])

    # Scenario 2 feature engineering: multi-class past-due history ->
    # binary high-risk flag
    row["HIGH_RISK_FLAG"] = 1 if int(form["past_due_count"]) >= 2 else 0

    def set_onehot(prefix, value):
        key = f"{prefix}_{value}"
        if key in row:
            row[key] = 1

    set_onehot("NAME_INCOME_TYPE", form["income_type"])
    set_onehot("NAME_EDUCATION_TYPE", form["education_type"])
    set_onehot("NAME_FAMILY_STATUS", form["family_status"])
    set_onehot("NAME_HOUSING_TYPE", form["housing_type"])
    set_onehot("OCCUPATION_TYPE", form["occupation_type"])

    return pd.DataFrame([row], columns=FEATURE_COLS)


@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        income_types=INCOME_TYPES,
        education_types=EDUCATION_TYPES,
        family_statuses=FAMILY_STATUSES,
        housing_types=HOUSING_TYPES,
        occupation_types=OCCUPATION_TYPES,
        model_name=MODEL_NAME,
    )


@app.route("/predict", methods=["POST"])
def predict():
    X_row = build_feature_row(request.form)

    if REQUIRES_SCALING:
        X_row_scaled = X_row.copy()
        X_row_scaled[NUMERIC_COLS] = SCALER.transform(X_row[NUMERIC_COLS])
        proba = MODEL.predict_proba(X_row_scaled)[0, 1]
    else:
        proba = MODEL.predict_proba(X_row)[0, 1]

    approved = proba >= 0.5
    confidence = round(proba * 100, 1) if approved else round((1 - proba) * 100, 1)

    return render_template(
        "result.html",
        approved=approved,
        confidence=confidence,
        model_name=MODEL_NAME,
        applicant_name=request.form.get("applicant_name", "Applicant"),
    )


@app.route("/about")
def about():
    with open("model/metrics.json") as f:
        import json
        metrics = json.load(f)
    return render_template("about.html", metrics=metrics, model_name=MODEL_NAME)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
