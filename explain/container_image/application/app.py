# Copyright 2023 IBM All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from flask import Flask, Response, render_template
import re
import accounts
import http
from database import DBConnection

app = Flask(__name__, instance_relative_config=True)
app.register_blueprint(accounts.bp)


@app.route("/alive", methods=["GET"])
def alive():
    return Response(status=http.HTTPStatus.OK)


@app.route("/", methods=["GET"])
def home():
    return render_template("home.jinja", database=DBConnection.__name__)


@app.template_global(name="human_readable")
def human_readable(text: str) -> str:
    # https://stackoverflow.com/questions/5020906/python-convert-camel-case-to-space-delimited-using-regex-and-taking-acronyms-in
    if not text:
        return ""

    split_titlecase = re.sub(r"((?<=[a-z])[A-Z]|(?<!\A)[A-Z](?=[a-z]))", r" \1", text)
    return split_titlecase.replace("_", " ")


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
