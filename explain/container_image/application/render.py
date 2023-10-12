import jinja2
import re
from bs4 import BeautifulSoup as bs
from typing import Optional, Dict, Any


def _human_readable(text: str) -> str:
    # https://stackoverflow.com/questions/5020906/python-convert-camel-case-to-space-delimited-using-regex-and-taking-acronyms-in
    if not text:
        return ""

    split_titlecase = re.sub(r"((?<=[a-z])[A-Z]|(?<!\A)[A-Z](?=[a-z]))", r" \1", text)
    return split_titlecase.replace("_", " ")


def render(template: str, vars: Dict[str, Any]) -> str:
    environment = jinja2.Environment(loader=jinja2.FileSystemLoader("templates"))
    template = environment.get_template(template)
    template.globals.update({"human_readable": _human_readable})
    content = bs(
        template.render(**vars),
        features="html.parser",
    )
    return content.prettify()
