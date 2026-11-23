"""The expected shape of the IBM Telco Customer Churn dataset.

This is the single source of truth for column names, allowed values and
plain-language descriptions. The validator, the data dictionary and (later)
the dashboard's help text all read from here.
"""

from __future__ import annotations

ID_COLUMN = "customerID"
TARGET_COLUMN = "Churn"

YES_NO = frozenset({"Yes", "No"})
_INTERNET_ADDON = frozenset({"Yes", "No", "No internet service"})

# Columns in the order they appear in the published file.
EXPECTED_COLUMNS: list[str] = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges", "Churn",
]

NUMERIC_COLUMNS: list[str] = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]

# Allowed values for every text (categorical) column.
CATEGORICAL_VALUES: dict[str, frozenset[str]] = {
    "gender": frozenset({"Male", "Female"}),
    "Partner": YES_NO,
    "Dependents": YES_NO,
    "PhoneService": YES_NO,
    "MultipleLines": frozenset({"Yes", "No", "No phone service"}),
    "InternetService": frozenset({"DSL", "Fiber optic", "No"}),
    "OnlineSecurity": _INTERNET_ADDON,
    "OnlineBackup": _INTERNET_ADDON,
    "DeviceProtection": _INTERNET_ADDON,
    "TechSupport": _INTERNET_ADDON,
    "StreamingTV": _INTERNET_ADDON,
    "StreamingMovies": _INTERNET_ADDON,
    "Contract": frozenset({"Month-to-month", "One year", "Two year"}),
    "PaperlessBilling": YES_NO,
    "PaymentMethod": frozenset({
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)",
    }),
    "Churn": YES_NO,
}

# Numeric columns restricted to a fixed set of codes.
NUMERIC_CODES: dict[str, frozenset[int]] = {"SeniorCitizen": frozenset({0, 1})}

# Plain-language descriptions, shown in docs and later in the dashboard.
COLUMN_DESCRIPTIONS: dict[str, str] = {
    "customerID": "Unique customer identifier.",
    "gender": "Customer gender (Male / Female).",
    "SeniorCitizen": "1 if the customer is 65 or older, otherwise 0.",
    "Partner": "Whether the customer has a partner.",
    "Dependents": "Whether the customer has dependents (e.g. children).",
    "tenure": "Number of months the customer has stayed with the company.",
    "PhoneService": "Whether the customer has a phone line.",
    "MultipleLines": "Whether the customer has more than one phone line.",
    "InternetService": "Type of internet connection: DSL, Fiber optic or none.",
    "OnlineSecurity": "Whether the customer pays for an online security add-on.",
    "OnlineBackup": "Whether the customer pays for an online backup add-on.",
    "DeviceProtection": "Whether the customer pays for device protection.",
    "TechSupport": "Whether the customer pays for premium technical support.",
    "StreamingTV": "Whether the customer streams TV through the service.",
    "StreamingMovies": "Whether the customer streams movies through the service.",
    "Contract": "Contract length: Month-to-month, One year or Two year.",
    "PaperlessBilling": "Whether the customer receives bills electronically.",
    "PaymentMethod": "How the customer pays.",
    "MonthlyCharges": "Amount charged to the customer each month (USD).",
    "TotalCharges": "Total amount charged over the customer's tenure (USD).",
    "Churn": "TARGET: whether the customer left within the last month (Yes / No).",
}
