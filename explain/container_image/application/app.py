from flask import Flask, Response, render_template
import re
import accounts
import http

app = Flask(__name__, instance_relative_config=True)
app.register_blueprint(accounts.bp)


@app.route("/alive", methods=["GET"])
def alive():
    return Response(status=http.HTTPStatus.OK)


@app.route("/", methods=["GET"])
def home():
    return render_template("home.jinja")


@app.template_global(name="human_readable")
def human_readable(text: str) -> str:
    # https://stackoverflow.com/questions/5020906/python-convert-camel-case-to-space-delimited-using-regex-and-taking-acronyms-in
    if not text:
        return ""

    split_titlecase = re.sub(r"((?<=[a-z])[A-Z]|(?<!\A)[A-Z](?=[a-z]))", r" \1", text)
    return split_titlecase.replace("_", " ")


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
