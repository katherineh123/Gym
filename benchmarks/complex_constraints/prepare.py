# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare the ComplexConstraints benchmark and its official judge prompt.

Benchmark: 75 prompts (CIF-001 to CIF-075) with 1,559 rubric criteria, CC BY 4.0 (Mehta et al., arXiv:2606.09118).
Surge AI published it as https://huggingface.co/datasets/surgeai/ComplexConstraints, which no longer serves it to
anonymous users. The rows come from a copy of that dataset's CSV, saved as an .xlsx, in the MIT-licensed repository
https://github.com/smarinacseas/failure-mode-id. The copy is pinned to a commit and checked against its sha256 and the
published counts.

Judge prompt: the template and response schema of the official harness's scorer.py
(https://github.com/surge-ai/complex-constraints), which carries no license. Both are read from scorer.py's syntax tree
at a pinned commit (the file is never imported or run).

Everything is written under `data/`, which is not committed.
"""

import ast
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, List
from xml.etree import ElementTree

import requests


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "complex_constraints_benchmark.jsonl"
JUDGE_PROMPT_FPATH = DATA_DIR / "complex_constraints_judge_prompt.json"

DATA_SOURCE_COMMIT = "49eb91959b7c969adab9a0bad85002647be7e6f9"  # pragma: allowlist secret
DATA_SOURCE_URL = (
    f"https://raw.githubusercontent.com/smarinacseas/failure-mode-id/{DATA_SOURCE_COMMIT}/data/ComplexConstraints.xlsx"
)
DATA_SOURCE_SHA256 = "57c26486dc4b39685d122cd4bcbef05377555cbdc507b6f97093d38cb52ff859"  # pragma: allowlist secret
EXPECTED_ROWS = 75
EXPECTED_CRITERIA = 1559
# The dataset's columns besides criterion_1 ... criterion_40.
ID_COLUMN, PROMPT_COLUMN = "benchmark_id", "prompt"
LABEL_COLUMNS = ("use_case", "instruction_type", "prompt_style")

JUDGE_SOURCE_COMMIT = "9ef1dd73d20030be94bdc399fd1c6ef7970f68d1"  # pragma: allowlist secret
JUDGE_SOURCE_URL = (
    f"https://raw.githubusercontent.com/surge-ai/complex-constraints/{JUDGE_SOURCE_COMMIT}/"
    "src/complex_constraints/scorer.py"
)
JUDGE_SOURCE_SHA256 = "abac94cf97677e5a1dc909ea1b9f46671c3da71bf71c507df57589e2b50521ef"  # pragma: allowlist secret
JUDGE_PLACEHOLDERS = ("{{ rubric.response_to_grade }}", "{{ rubric.criterion }}")

_XLSX_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
_CRITERION_COLUMN_RE = re.compile(r"criterion_(\d+)")


def download(url: str, sha256: str) -> bytes:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    digest = hashlib.sha256(response.content).hexdigest()
    if digest != sha256:
        raise ValueError(f"{url} has sha256 {digest}, expected {sha256}")
    return response.content


def _column_index(cell_ref: str) -> int:
    index = 0
    for letter in re.match(r"[A-Z]+", cell_ref).group(0):
        index = index * 26 + ord(letter) - ord("A") + 1
    return index - 1


def read_xlsx_rows(content: bytes) -> List[Dict[str, str]]:
    """Read the first worksheet of an .xlsx file as one dict per non-empty row, keyed by the header row."""
    archive = zipfile.ZipFile(io.BytesIO(content))
    shared: List[str] = []
    if "xl/sharedStrings.xml" in archive.namelist():
        for item in ElementTree.fromstring(archive.read("xl/sharedStrings.xml")).findall("m:si", _XLSX_NS):
            # Plain text is one <t>; rich text is runs of <r><t>. Phonetic hints (<rPh>) are not cell text.
            runs = [item.find("m:t", _XLSX_NS)] + [run.find("m:t", _XLSX_NS) for run in item.findall("m:r", _XLSX_NS)]
            shared.append("".join(run.text or "" for run in runs if run is not None))

    table: List[List[str]] = []
    sheet = ElementTree.fromstring(archive.read("xl/worksheets/sheet1.xml"))
    for row in sheet.iter(f"{{{_XLSX_NS['m']}}}row"):
        cells: Dict[int, str] = {}
        for cell in row.findall("m:c", _XLSX_NS):
            value = cell.find("m:v", _XLSX_NS)
            if cell.get("t") == "s" and value is not None:
                text = shared[int(value.text)]
            elif cell.get("t") == "inlineStr":
                text = "".join(node.text or "" for node in cell.iter(f"{{{_XLSX_NS['m']}}}t"))
            else:
                text = value.text if value is not None and value.text is not None else ""
            cells[_column_index(cell.get("r"))] = text
        if any(text.strip() for text in cells.values()):
            table.append([cells.get(index, "") for index in range(max(cells) + 1)])

    header, *body = table
    return [{name: (row[index] if index < len(row) else "") for index, name in enumerate(header)} for row in body]


def build_row(row: Dict[str, str]) -> Dict[str, Any]:
    """Turn one dataset row into a Gym task: the prompt is the only user turn, the criteria stay out of it."""
    task_id = row.get(ID_COLUMN, "").strip()
    numbers = {column: int(match.group(1)) for column in row if (match := _CRITERION_COLUMN_RE.fullmatch(column))}
    # Every non-empty criterion cell counts, in column order, as in the official loader.
    rubrics = [row[column].strip() for column in sorted(numbers, key=numbers.get) if row[column].strip()]
    if not task_id or not row.get(PROMPT_COLUMN, "").strip():
        raise ValueError(f"row {task_id or '?'}: expected a benchmark_id and a prompt")
    if not rubrics:
        raise ValueError(f"row {task_id}: has no rubric criteria")
    return {
        "task_id": task_id,
        **{column: row.get(column, "") for column in LABEL_COLUMNS},
        "rubrics": rubrics,
        "responses_create_params": {"input": [{"role": "user", "content": row[PROMPT_COLUMN]}]},
    }


def _schema_value(node: ast.AST) -> Any:
    """Evaluate the literal parts of scorer.py's schema declaration, where JSONSchema(...) calls become dicts."""
    if isinstance(node, ast.Call) and ast.unparse(node.func) == "JSONSchema" and not node.args:
        return {keyword.arg: _schema_value(keyword.value) for keyword in node.keywords}
    if isinstance(node, ast.Dict):
        return {_schema_value(key): _schema_value(value) for key, value in zip(node.keys, node.values)}
    if isinstance(node, ast.List):
        return [_schema_value(element) for element in node.elts]
    if isinstance(node, ast.Constant):
        return node.value
    raise ValueError(f"unexpected expression in the judge schema: {ast.unparse(node)}")


def extract_judge_prompt(source: str) -> Dict[str, Any]:
    """Read the judge template and response schema from scorer.py without running it."""
    values: Dict[str, ast.AST] = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            values[node.targets[0].id] = node.value

    template = values.get("JUDGE_PROMPT_TEMPLATE")
    if not (isinstance(template, ast.Constant) and isinstance(template.value, str)):
        raise ValueError("scorer.py has no string literal JUDGE_PROMPT_TEMPLATE")
    if not all(placeholder in template.value for placeholder in JUDGE_PLACEHOLDERS):
        raise ValueError(f"JUDGE_PROMPT_TEMPLATE lacks the placeholders {JUDGE_PLACEHOLDERS}")

    schema = values.get("RESPONSE_SCHEMA")
    if not (isinstance(schema, ast.Call) and ast.unparse(schema.func) == "ResponseSchema" and not schema.args):
        raise ValueError("scorer.py has no RESPONSE_SCHEMA = ResponseSchema(...) declaration")
    arguments = {keyword.arg: keyword.value for keyword in schema.keywords}
    if set(arguments) != {"name", "json_schema"}:
        raise ValueError(f"unexpected ResponseSchema arguments {sorted(arguments)}")
    response_schema = {"name": _schema_value(arguments["name"]), "schema": _schema_value(arguments["json_schema"])}
    return {"judge_prompt_template": template.value, "response_schema": response_schema}


def prepare_judge_prompt(output_fpath: Path = JUDGE_PROMPT_FPATH) -> Path:
    judge = extract_judge_prompt(download(JUDGE_SOURCE_URL, JUDGE_SOURCE_SHA256).decode("utf-8"))
    output_fpath.parent.mkdir(parents=True, exist_ok=True)
    output_fpath.write_text(
        json.dumps({"source": JUDGE_SOURCE_URL, "license": "none stated; not redistributed", **judge}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return output_fpath


def prepare(output_fpath: Path = OUTPUT_FPATH) -> Path:
    rows = [build_row(row) for row in read_xlsx_rows(download(DATA_SOURCE_URL, DATA_SOURCE_SHA256))]
    criteria = sum(len(row["rubrics"]) for row in rows)
    if len(rows) != EXPECTED_ROWS or criteria != EXPECTED_CRITERIA:
        raise ValueError(
            f"expected {EXPECTED_ROWS} rows and {EXPECTED_CRITERIA} criteria, got {len(rows)} and {criteria}"
        )
    if len({row["task_id"] for row in rows}) != len(rows):
        raise ValueError("benchmark_id values are not unique")
    prepare_judge_prompt(output_fpath.parent / JUDGE_PROMPT_FPATH.name)
    output_fpath.parent.mkdir(parents=True, exist_ok=True)
    with output_fpath.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return output_fpath


if __name__ == "__main__":
    print(prepare())
