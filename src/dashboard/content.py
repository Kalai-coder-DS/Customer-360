"""All explanatory text shown in the dashboard, kept in one place.

Every module answers the same six questions, as required by the project brief:
What is this? Why is it needed? How does it work? What data does it use?
What does the result mean? How does a business use it?
"""

from __future__ import annotations

SIX_QUESTIONS = [
    ("what", "What is this?"),
    ("why", "Why is it needed?"),
    ("how", "How does it work?"),
    ("data", "What data does it use?"),
    ("meaning", "What does the result mean?"),
    ("business", "How does a business use it?"),
]

WORKFLOW = [
    ("Customer data", "Telco customer records, 21 columns"),
    ("Data cleaning", "Fix blanks, duplicates and types"),
    ("Customer analysis", "Who churns? Statistics and charts"),
    ("Segmentation", "Group similar customers"),
    ("Churn prediction", "Score every customer's risk"),
    ("Risk explanation", "Why is this customer at risk?"),
    ("Retention action", "Who to contact and what to offer"),
]

BUSINESS_PROBLEM = (
    "Losing a customer costs a subscription business their future revenue, and "
    "winning a new one usually costs more than keeping an existing one. Customer360 AI "
    "answers six questions: who our customers are, which ones are valuable, which are "
    "likely to leave, why they might leave, which groups need attention, and what to do "
    "about it."
)

DATA_NOTICE = (
    "Data: IBM Telco Customer Churn sample (a fictional telecom company). Results "
    "illustrate the methods and are not real business outcomes."
)

EXPLAINERS: dict[str, dict[str, str]] = {
    "overview": {
        "what": "A one-page summary of the customer base: how many customers there are, "
                "how many left, how much revenue is at risk and where the main problems sit.",
        "why": "Leaders need the big picture in seconds before diving into details.",
        "how": "Counts and averages from the cleaned data, combined with the churn model's "
               "risk scores and the retention plan.",
        "data": "Cleaned customer table, churn predictions and the retention plan.",
        "meaning": "Churn rate is the share of customers who left in the last month. Revenue "
                   "at risk adds up each customer's churn probability × monthly charges × 12 "
                   "months; it is an estimate, not a forecast.",
        "business": "Set priorities: which problem is biggest, how urgent it is and where to "
                    "look next.",
    },
    "profile": {
        "what": "Everything known about one customer on a single page: who they are, what "
                "they use and pay, their segment, churn risk, the reasons behind it and the "
                "recommended action.",
        "why": "Retention agents talk to one customer at a time and need context fast.",
        "how": "Looks up the customer in each table produced by the pipeline and shows the "
               "model's explanation for their score.",
        "data": "Clean customer record, segment, churn prediction, explanation and retention "
                "plan for the selected customer.",
        "meaning": "The churn probability is the model's estimate based on similar past "
                   "customers. Coral factors push the estimate up, violet factors push it down. "
                   "They show association, not proven cause.",
        "business": "Prepare a personal, relevant conversation: address the real risk "
                    "factors instead of offering a generic discount.",
    },
    "segmentation": {
        "what": "Groups of customers who look alike, found in two ways: simple RFM-style "
                "scores and K-Means clustering.",
        "why": "Different customers need different treatment. Segments turn 7,000 individuals "
               "into a handful of groups a team can plan for.",
        "how": "RFM-style scores rate each customer 1–5 on Loyalty (tenure), Engagement "
               "(number of services) and Monetary value (monthly charges). True RFM needs "
               "purchase dates, which this data does not have, so these are labelled "
               "substitutes. K-Means groups customers with similar tenure, charges and "
               "services; the number of groups is chosen by the silhouette score.",
        "data": "Tenure, monthly charges and number of services. Churn is NOT used to build "
                "segments, only to describe them afterwards.",
        "meaning": "Each segment has a name, a profile, a churn rate and a share of revenue. "
                   "A segment with high churn and a large revenue share is a priority.",
        "business": "Design campaigns per segment: onboarding for new customers, rewards for "
                    "loyal high-value customers, cross-selling for loyal low-spenders.",
    },
    "prediction": {
        "what": "A machine learning model that estimates each customer's probability of "
                "churning, plus how well it performs on customers it has never seen.",
        "why": "Group statistics show where churn is common; a model scores each individual "
               "so the team knows exactly whom to contact first.",
        "how": "80% of customers train the models and 20% are held back as a final test. "
               "Logistic Regression and Random Forest are compared with 5-fold "
               "cross-validation against a baseline; the best is chosen and tested once. "
               "Preprocessing is learned from training data only to prevent data leakage.",
        "data": "19 customer characteristics plus 3 engineered features. The customer ID and "
                "the churn label itself are never used as inputs.",
        "meaning": "ROC-AUC: how often the model ranks a churner above a stayer (0.5 = coin "
                   "flip). Recall: share of churners caught. Precision: share of flagged "
                   "customers who really churn. Accuracy alone is misleading because most "
                   "customers stay.",
        "business": "Contact High-risk customers first and choose the decision threshold to "
                    "match the budget: cheap actions can tolerate more false alarms.",
    },
    "explain": {
        "what": "Explanations of the churn model: which factors it relies on overall, and "
                "why it gives one customer their score.",
        "why": "A score without reasons is hard to trust and hard to act on.",
        "how": "Shapley values (the idea behind SHAP) split the difference between a "
               "customer's prediction and an average customer's into fair shares per factor. "
               "Global importance measures how much accuracy drops when a factor is shuffled.",
        "data": "The trained model, the cleaned customer data and 200 reference customers "
                "who define 'an average customer'.",
        "meaning": "A positive (coral) contribution raises the predicted risk, a negative "
                   "(violet) one lowers it. These show what the model associates with churn, "
                   "not what causes it.",
        "business": "Agents see the reasons behind each score; managers check the model makes "
                    "sense; analysts pick which retention experiments to run.",
    },
    "retention": {
        "what": "A retention plan: who to contact first, which offer suits each customer and "
                "what the campaign might be worth.",
        "why": "Budgets and call-centre time are limited, so effort must go where the risk "
               "and the value are highest.",
        "how": "Value at risk = churn probability × monthly charges × 12. Customers are "
               "placed in priority tiers by risk and value. Actions are suggested only when "
               "they apply to the customer and target a factor that raises their risk.",
        "data": "Churn predictions, explanations, segments and a playbook of actions with "
                "assumed costs and success rates.",
        "meaning": "The call list is ordered by value at risk. Impact figures are estimates "
                   "built on stated, adjustable assumptions, not measured results.",
        "business": "Work through the call list, compare campaign options and turn the most "
                    "common actions into controlled experiments to learn real success rates.",
    },
    "data": {
        "what": "Upload, validate, store and score customer files, and download every report.",
        "why": "Real data is messy; checking it before analysis prevents wrong conclusions.",
        "how": "Uploaded CSVs are checked for missing columns, blank values, duplicates and "
               "unexpected values, saved to the SQLite database and optionally scored with "
               "the trained model.",
        "data": "Any CSV with the Telco column layout.",
        "meaning": "Errors block a file from being used; warnings are issues the cleaning "
                   "step can fix.",
        "business": "Refresh the analysis with new customer data and share results with "
                    "colleagues as CSV or report files.",
    },
}

GLOSSARY = {
    "Churn": "A customer leaving (cancelling the service). In this data, leaving within the "
             "last month.",
    "Churn rate": "Share of customers who churned.",
    "Tenure": "Number of months a customer has been with the company.",
    "Segment": "A group of customers with similar characteristics.",
    "Silhouette score": "How clearly separated clusters are, from -1 (poor) to 1 (very clear).",
    "Probability": "The model's estimated chance (0–100%) that a customer will churn.",
    "Threshold": "The probability at or above which a customer is flagged as a likely churner.",
    "ROC-AUC": "How well the model ranks churners above non-churners; 0.5 = guessing, 1 = perfect.",
    "Precision": "Of customers flagged as churners, the share who actually churned.",
    "Recall": "Of customers who actually churned, the share the model flagged.",
    "F1 score": "A single number balancing precision and recall.",
    "Confusion matrix": "Table of correct flags, false alarms, missed churners and correct "
                        "'leave alone' decisions.",
    "Data leakage": "When information the model would not have in real use sneaks into "
                    "training, making results look better than they are.",
    "Cross-validation": "Training and testing on different slices of the data several times "
                        "for a stable estimate.",
    "Shapley value": "A factor's fair share of the difference between this customer's "
                     "prediction and an average customer's.",
    "Value at risk": "Churn probability × monthly charges × horizon: expected revenue that "
                     "could be lost.",
    "Correlation vs causation": "Two things moving together does not prove one causes the "
                                "other.",
}

LIMITATIONS = [
    "The dataset describes a fictional company (IBM sample); results illustrate methods only.",
    "It is a single snapshot without dates, so true RFM analysis and time trends are not possible.",
    "Model explanations show association, not cause.",
    "Retention costs and success rates are illustrative assumptions, not measured results.",
    "Scores for existing customers come from the final model, which has seen them; honest "
    "performance figures are the test-set results.",
]
