from flask import (
    Blueprint,
    redirect,
    render_template,
    request,
    url_for,
)
from database import DB2DataBaseConnection
import os
import json
from typing import Dict, Any
from flask import current_app

COLUMN_INFO = json.loads(os.environ.get("COLUMN_INFO", {}))
bp = Blueprint("Accounts", __name__, url_prefix="/accounts")


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
    client_info = request.get_json(force=True)
    current_app.logger.info(json.dumps(client_info, indent=2))
    with DB2DataBaseConnection() as db:
        account_id = db.insert_account_from_row_dict(client_info)
        return redirect(url_for("Accounts.retrieve_account_info", account=account_id))


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
def retrieve_account_info(account_id: int):
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
def update_client_info(account_id: int):
    new_client_info = request.get_json(force=True)

    with DB2DataBaseConnection() as db:
        db.update_client_info_from_row_change_dict(int(account_id), new_client_info)

    return redirect(url_for(retrieve_account_info, account_id=account_id))


@bp.route("/<account_id>", methods=["DELETE"])
def delete_client_info(account_id: int):
    raise NotImplementedError()
