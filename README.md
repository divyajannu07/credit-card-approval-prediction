# Credit Card Approval Prediction

**Team:** Finance Group
**Workspace:** Kanban &mdash; Overview

> 🔗 **Demo:** _add your hosted demo link here_ (e.g. Render / Railway / IBM Cloud URL)
> 🔗 **GitHub:** _add your repository link here_
>
> Update the two lines above with your live links, then share this README (or the repo)
> with your mentor for review.

---

## 1. Project Overview

Banks and financial institutions receive thousands of credit card applications every
day. A significant portion are rejected due to factors such as high existing loan
balances, insufficient income levels, or excessive credit inquiries. Manually
reviewing every application is slow and error-prone at scale.

This project automates the credit card approval decision using machine learning. A
predictive model is trained on historical applicant data (financial + demographic
inputs) to determine whether an applicant is likely to be **approved** or **rejected**
&mdash; the way a real bank underwriting team would.

Four classification algorithms are trained and compared:

| # | Algorithm |
|---|-----------|
| 1 | Logistic Regression |
| 2 | Random Forest |
| 3 | XGBoost (Gradient Boosting) |
| 4 | Decision Tree |

The best-performing model (selected by F1 score) is saved and served through a
**Flask web application** with a clean, analyst-friendly UI. A stub IBM Watson
Machine Learning deployment script is included so the same pipeline can be pushed to
IBM Cloud for scalable, real-time hosting.

## 2. Business Scenarios Covered

- **Scenario 1 &mdash; Automated Credit Card Application Screening.** A credit
  analyst enters an applicant's income type, employment duration, and credit history
  into the web app and gets an instant approve/reject prediction to prioritize review.
- **Scenario 2 &mdash; High-Risk Applicant Identification & Compliance Review.** The
  feature engineering pipeline converts multi-class payment status codes
  (`PAST_DUE_COUNT`) into a binary `HIGH_RISK_FLAG`, letting compliance officers
  clearly identify applicants who should be ineligible for a new card.
- **Scenario 3 &mdash; Customer Self-Service Eligibility Check.** A prospective
  customer enters personal and financial details before formally applying, and
  instantly sees their predicted likelihood of approval &mdash; reducing unnecessary
  rejections and setting expectations up front.

## 3. Project Structure

```
credit-card-approval/
├── app.py                     # Flask web application (routes + prediction logic)
├── train.py                   # Model training pipeline (4 algorithms + selection)
├── requirements.txt
├── data/
│   ├── generate_data.py       # Generates the synthetic applicant dataset
│   └── credit_card_approval.csv
├── model/
│   ├── model.pkl              # Saved best model + scaler + feature schema
│   ├── metrics.json           # Accuracy / Precision / Recall / F1 / ROC-AUC per model
│   ├── model_comparison.png   # Bar chart comparing all 4 models
│   └── confusion_matrix.png   # Confusion matrix of the deployed model
├── deployment/
│   └── watson_ml_deploy.py    # IBM Watson ML deployment pipeline (stub)
├── templates/
│   ├── base.html
│   ├── index.html             # Applicant screening form
│   ├── result.html            # Prediction result page
│   └── about.html             # Model performance report
└── static/
    ├── style.css
    ├── model_comparison.png
    └── confusion_matrix.png
```

## 4. Dataset & Feature Engineering

The dataset (`data/generate_data.py`) mirrors the structure of the classic
application-record + credit-record credit approval dataset: demographic fields
(gender, family status, education, housing), financial fields (income, income type,
employment duration), and credit-bureau-style fields (past-due months, recent
inquiries, existing loans, account age).

Key engineering steps performed in `train.py`:

- `DAYS_BIRTH` / `DAYS_EMPLOYED` → `AGE_YEARS` / `EMPLOYED_YEARS`
- Multi-class past-due payment history → binary `HIGH_RISK_FLAG` (Scenario 2)
- One-hot encoding of categorical fields (income type, education, family status,
  housing type, occupation)
- Standard scaling of numeric features for the Logistic Regression model

## 5. Setup & Usage

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Re)generate the dataset — already included, but reproducible
python data/generate_data.py

# 3. Train all 4 models, evaluate, and save the best one
python train.py

# 4. Run the web app
python app.py
# Visit http://localhost:5000
```

## 6. Model Performance

Run `python train.py` to regenerate `model/metrics.json` and the charts below.
Example output on the bundled synthetic dataset:

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.768 | 0.781 | 0.964 | 0.865 | 0.670 |
| Random Forest | 0.754 | 0.775 | 0.958 | 0.859 | 0.658 |
| XGBoost | 0.748 | 0.775 | 0.941 | 0.852 | 0.613 |
| Decision Tree | 0.753 | 0.778 | 0.943 | 0.854 | 0.622 |

*(Your numbers may vary slightly depending on dataset regeneration and package
versions. The app always serves whichever model scored highest on F1.)*

## 7. IBM Watson Machine Learning Deployment

`deployment/watson_ml_deploy.py` contains a ready-to-fill pipeline for pushing
`model/model.pkl` to IBM Watson Machine Learning for cloud hosting and scalable,
real-time inference. Fill in your IBM Cloud API key, deployment space ID, and run:

```bash
python deployment/watson_ml_deploy.py
```

## 8. Skills Demonstrated

XGBoost · Machine Learning Algorithms · Artificial Intelligence · Decision Tree
Learning · NumPy · Python · Scikit-Learn · Matplotlib · Flask

## 9. Next Steps for Your Mentor Review

1. Push this project to a **GitHub** repository and paste the link at the top of
   this README.
2. Deploy `app.py` to a host (Render, Railway, PythonAnywhere, or IBM Cloud via the
   Watson ML script) and paste the live **Demo** link at the top of this README.
3. Update the Kanban board's "Overview" card with both links.
