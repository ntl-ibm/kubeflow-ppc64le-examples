from flask import (
    Blueprint,
    redirect,
    render_template,
    request,
    url_for,
    Response,
)
from database import DB2DataBaseConnection
import os
import json
from typing import Dict, Any
from flask import current_app
from http import HTTPStatus
import requests

COLUMN_INFO = json.loads(os.environ.get("COLUMN_INFO", {}))
PREDICT_URL = os.environ.get("PREDICT_URL", None)
EXPLAIN_URL = os.environ.get("EXPLAIN_URL", None)

bp = Blueprint("Accounts", __name__, url_prefix="/accounts")


def inject_ai(db_row: Dict[str, Any]) -> Dict[str, Any]:
    if PREDICT_URL:
        df = {k: {"0": v} for k, v in db_row.items()}
        predictions = requests.post(PREDICT_URL, json=df).json()
        current_app.logger.info(json.dumps(predictions, indent=2))

        db_row["PredictedRisk"] = predictions["predictions"][0]

        if db_row["PredictedRisk"] == "Risk" and EXPLAIN_URL:
            explain = requests.post(EXPLAIN_URL, json=df).json()
            current_app.logger.info(json.dumps(explain, indent=2))

            anchors = explain["explanations"][0]["anchor"]
            db_row["ExplainRisk"] = ",".join(anchors)

    return db_row


def create_account_defaults() -> Dict[str, Any]:
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


@bp.route("/", methods=["POST"])
def create_account():
    current_app.logger.setLevel("DEBUG")
    account_info = request.get_json(force=True)
    current_app.logger.info(json.dumps(account_info, indent=2))

    account_info = inject_ai(account_info)
    current_app.logger.info(json.dumps(account_info, indent=2))

    with DB2DataBaseConnection() as db:
        account_id = db.insert_account_from_row_dict(account_info)
        current_app.logger.info(f"Account id {account_id} was created")
        return Response(
            status=HTTPStatus.CREATED,
            headers=[
                (
                    "Location",
                    url_for("Accounts.retrieve_account_info", account_id=account_id),
                )
            ],
        )


@bp.route("/", methods=["GET"])
def list_or_create_accounts():
    request_new = bool(request.args.get("new", False))

    current_app.logger.info(f"COLUMN_INFO\n {json.dumps(COLUMN_INFO, indent=2)}")

    if request_new:
        return render_template(
            "add_account.jinja",
            schema=COLUMN_INFO,
            form_defaults=create_account_defaults(),
            create_account_url=url_for("Accounts.create_account"),
        )

    limit = int(request.args.get("limit", 100))
    offset = int(request.args.get("offset", 0))

    raise NotImplementedError()


@bp.route("/<account_id>", methods=["GET"])
def retrieve_account_info(account_id):
    for_edit = bool(request.args.get("for_edit", False))

    with DB2DataBaseConnection() as db:
        client_info = db.get_client_info(int(account_id))

    return render_template(
        "update_account.jinja",
        schema=COLUMN_INFO,
        client_info_values=client_info,
        view_only_mode=(not for_edit),
    )


@bp.route("/<account_id>", methods=["PUT"])
def update_account_info(account_id):
    updated_account_info = request.get_json(force=True)

    updated_account_info = inject_ai(updated_account_info)
    current_app.logger.info(json.dumps(updated_account_info, indent=2))

    with DB2DataBaseConnection() as db:
        db.update_account_from_row_change_dict(int(account_id), updated_account_info)

    return redirect(url_for("Accounts.retrieve_account_info", account_id=account_id))


@bp.route("/<account_id>", methods=["DELETE"])
def delete_account_info(account_id: int):
    raise NotImplementedError()
