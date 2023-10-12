import json
import pandas as pd
from render import render

mock_client_data = json.loads(
    """
{
  "CLIENT_ID": 9000,
  "CheckingStatus": "no_checking",
  "CreditHistory": "outstanding_credit",
  "LoanPurpose": "repairs",
  "ExistingSavings": "500_to_1000",
  "EmploymentDuration": "4_to_7",
  "Sex": "male",
  "OthersOnLoan": "co-applicant",
  "OwnsProperty": "unknown",
  "InstallmentPlans": "none",
  "Housing": "free",
  "Job": "management_self-employed",
  "Telephone": "yes",
  "ForeignWorker": "yes",
  "Risk": null,
  "LoanDuration": 31,
  "LoanAmount": 8411,
  "InstallmentPercent": 5,
  "CurrentResidenceDuration": 5,
  "Age": 46,
  "ExistingCreditsCount": 2,
  "Dependents": 2,
  "PredictedRisk": "Risk",
  "ExplainRisk": ["Age > 36.00", "CurrentResidenceDuration > 3.00",  "InstallmentPercent > 4.00"]
}
"""
)

with open("columns.json", "r") as f:
    column_info = json.loads(f.read())

content = render(
    "update_account.jinja",
    vars={
        "schema": column_info,
        "client_info_values": mock_client_data,
        "view_only_mode": True,
    },
)

with open("update_output.html", "w") as f:
    f.write(content)
