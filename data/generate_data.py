"""
generate_data.py
-----------------
Generates a synthetic dataset that mirrors the structure of the classic
"Credit Card Approval Prediction" dataset (application record + credit
record joined and feature engineered). This lets the project run fully
end-to-end without requiring an external download.

Run:
    python generate_data.py
Produces:
    credit_card_approval.csv
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 8000  # number of applicants

# ---------------------------------------------------------------------
# Demographic / application-level features
# ---------------------------------------------------------------------
gender = np.random.choice(["M", "F"], size=N, p=[0.45, 0.55])
own_car = np.random.choice(["Y", "N"], size=N, p=[0.4, 0.6])
own_realty = np.random.choice(["Y", "N"], size=N, p=[0.55, 0.45])
children = np.random.choice([0, 1, 2, 3, 4], size=N, p=[0.55, 0.22, 0.15, 0.06, 0.02])

income = np.round(np.random.lognormal(mean=10.8, sigma=0.5, size=N), -2)
income = np.clip(income, 20000, 800000)

income_type = np.random.choice(
    ["Working", "Commercial associate", "Pensioner", "State servant", "Student"],
    size=N, p=[0.52, 0.23, 0.14, 0.09, 0.02]
)

education_type = np.random.choice(
    ["Secondary / secondary special", "Higher education", "Incomplete higher",
     "Lower secondary", "Academic degree"],
    size=N, p=[0.65, 0.24, 0.06, 0.04, 0.01]
)

family_status = np.random.choice(
    ["Married", "Single / not married", "Civil marriage", "Separated", "Widow"],
    size=N, p=[0.6, 0.2, 0.1, 0.06, 0.04]
)

housing_type = np.random.choice(
    ["House / apartment", "With parents", "Municipal apartment",
     "Rented apartment", "Office apartment", "Co-op apartment"],
    size=N, p=[0.72, 0.11, 0.08, 0.05, 0.02, 0.02]
)

age_years = np.random.randint(21, 70, size=N)
days_birth = -age_years * 365

employed_years = np.where(
    income_type == "Pensioner",
    0,
    np.random.randint(0, 40, size=N)
)
days_employed = np.where(employed_years == 0, 365243, -employed_years * 365)

occupation_type = np.random.choice(
    ["Laborers", "Core staff", "Sales staff", "Managers", "Drivers",
     "High skill tech staff", "Accountants", "Medicine staff",
     "Security staff", "Cooking staff", "Cleaning staff", "Other"],
    size=N
)

fam_members = children + np.where(
    family_status == "Married", 2,
    np.where(family_status == "Civil marriage", 2, 1)
)

flag_work_phone = np.random.choice([0, 1], size=N, p=[0.75, 0.25])
flag_phone = np.random.choice([0, 1], size=N, p=[0.6, 0.4])
flag_email = np.random.choice([0, 1], size=N, p=[0.7, 0.3])

# ---------------------------------------------------------------------
# Credit-history-derived features (from the "credit_record" table in the
# original dataset). PAST_DUE_COUNT simulates the number of months the
# applicant had a past-due balance (STATUS codes 1-5 in the raw data).
# ---------------------------------------------------------------------
past_due_count = np.random.poisson(lam=0.6, size=N)
credit_inquiries_6m = np.random.poisson(lam=1.2, size=N)
existing_loans = np.random.poisson(lam=1.0, size=N)
account_age_months = np.random.randint(1, 60, size=N)

# ---------------------------------------------------------------------
# Target construction: APPROVED = 1 (good applicant) / 0 (rejected)
# Built from a latent "risk score" so the relationships are learnable.
# ---------------------------------------------------------------------
risk_score = (
    -0.000006 * income
    + 0.55 * past_due_count
    + 0.18 * credit_inquiries_6m
    + 0.12 * existing_loans
    - 0.01 * age_years
    + np.where(income_type == "Student", 1.2, 0)
    + np.where(income_type == "Pensioner", -0.3, 0)
    + np.where(housing_type == "With parents", 0.3, 0)
    + np.where(education_type == "Higher education", -0.4, 0)
    + np.where(family_status == "Married", -0.2, 0)
    + np.random.normal(0, 0.6, size=N)
)

# Higher risk_score => more likely to be rejected
approval_prob = 1 / (1 + np.exp(risk_score - 1.0))
status = (np.random.rand(N) < approval_prob).astype(int)  # 1 = approved

df = pd.DataFrame({
    "ID": np.arange(1, N + 1),
    "CODE_GENDER": gender,
    "FLAG_OWN_CAR": own_car,
    "FLAG_OWN_REALTY": own_realty,
    "CNT_CHILDREN": children,
    "AMT_INCOME_TOTAL": income,
    "NAME_INCOME_TYPE": income_type,
    "NAME_EDUCATION_TYPE": education_type,
    "NAME_FAMILY_STATUS": family_status,
    "NAME_HOUSING_TYPE": housing_type,
    "DAYS_BIRTH": days_birth,
    "DAYS_EMPLOYED": days_employed,
    "FLAG_WORK_PHONE": flag_work_phone,
    "FLAG_PHONE": flag_phone,
    "FLAG_EMAIL": flag_email,
    "OCCUPATION_TYPE": occupation_type,
    "CNT_FAM_MEMBERS": fam_members,
    "PAST_DUE_COUNT": past_due_count,
    "CREDIT_INQUIRIES_6M": credit_inquiries_6m,
    "EXISTING_LOANS": existing_loans,
    "ACCOUNT_AGE_MONTHS": account_age_months,
    "APPROVED": status,
})

df.to_csv("credit_card_approval.csv", index=False)
print(f"Generated {len(df)} rows -> credit_card_approval.csv")
print(df["APPROVED"].value_counts(normalize=True))
