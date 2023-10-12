import json
from typing import Optional, Dict, Any
import pandas as pd
from render import render


def mock_form_defaults() -> Dict[str, Any]:
    return {
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
        "LoanDuration": 31,
        "LoanAmount": 8411,
        "InstallmentPercent": 5,
        "CurrentResidenceDuration": 5,
        "Age": 46,
        "ExistingCreditsCount": 2,
        "Dependents": 2,
    }


with open("columns.json", "r") as f:
    column_info = json.loads(f.read())

content = render(
    "add_account.jinja",
    vars={"schema": column_info, "form_defaults": mock_form_defaults()},
)

with open("output.html", "w") as f:
    f.write(content)
