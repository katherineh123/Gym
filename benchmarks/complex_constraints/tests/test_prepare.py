# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import hashlib
import io
import json
import zipfile
from unittest.mock import MagicMock, patch
from xml.sax.saxutils import escape

import pytest

from benchmarks.complex_constraints import prepare


SCORER_SOURCE = '''
from inspect_ai.model import ResponseSchema
from inspect_ai.util import JSONSchema

JUDGE_PROMPT_TEMPLATE = """Reply: {{ rubric.response_to_grade }}
Criterion: {{ rubric.criterion }}"""

RESPONSE_SCHEMA = ResponseSchema(
    name="verdict",
    json_schema=JSONSchema(
        type="object",
        required=["score"],
        properties={"score": JSONSchema(type="string", enum=["0", "1"], description="pass or fail")},
        additionalProperties=False,
    ),
)
'''


def column_letter(index: int) -> str:
    letters = ""
    index += 1
    while index:
        index, remainder = divmod(index - 1, 26)
        letters = chr(ord("A") + remainder) + letters
    return letters


def make_xlsx(table: list) -> bytes:
    """A minimal workbook: shared strings for most cells, one inline string, one rich-text cell and a blank row."""
    shared, rows = [], []
    for row_number, row in enumerate(table, 1):
        cells = []
        for column, value in enumerate(row):
            if value is None:
                continue
            ref = f"{column_letter(column)}{row_number}"
            if value.startswith("inline:"):
                text = escape(value.removeprefix("inline:"))
                cells.append(f'<c r="{ref}" t="inlineStr"><is><t>{text}</t></is></c>')
            else:
                shared.append(value)
                cells.append(f'<c r="{ref}" t="s"><v>{len(shared) - 1}</v></c>')
        rows.append(f'<row r="{row_number}">{"".join(cells)}</row>')
    rows.append(f'<row r="{len(table) + 1}"><c r="A{len(table) + 1}" t="s"><v>{len(shared)}</v></c></row>')
    items = [f"<si><t>{escape(text)}</t></si>" for text in shared]
    items.append('<si><t xml:space="preserve">  </t></si>')  # a whitespace-only cell: the row is skipped
    ns = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("xl/sharedStrings.xml", f"<sst {ns}>{''.join(items)}</sst>")
        archive.writestr(
            "xl/worksheets/sheet1.xml", f"<worksheet {ns}><sheetData>{''.join(rows)}</sheetData></worksheet>"
        )
    return buffer.getvalue()


HEADER = ["benchmark_id", "prompt", "use_case", "instruction_type", "prompt_style", "criterion_1", "criterion_2"]
TABLE = [
    HEADER,
    ["CIF-001", "Plan a café rota.", "Logistics", "Negative", "Direct prompting", "Covers Monday.", " No doubles. "],
    ["CIF-002", "inline:Sum the sales.", "Data", "Multistep", "Context prompting", "Total is 12.", None],
]


def test_read_xlsx_rows_reads_shared_inline_and_missing_cells_and_skips_blank_rows() -> None:
    rows = prepare.read_xlsx_rows(make_xlsx(TABLE))
    assert len(rows) == 2
    assert rows[0]["prompt"] == "Plan a café rota." and rows[0]["criterion_2"] == " No doubles. "
    assert rows[1]["prompt"] == "Sum the sales." and rows[1]["criterion_2"] == ""


def test_read_xlsx_rows_joins_rich_text_runs() -> None:
    ns = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
    shared = f"<sst {ns}><si><t>id</t></si><si><r><t>Bold </t></r><r><t>part</t></r><rPh><t>hint</t></rPh></si></sst>"
    sheet = (
        f'<worksheet {ns}><sheetData><row r="1"><c r="A1" t="s"><v>0</v></c></row>'
        '<row r="2"><c r="A2" t="s"><v>1</v></c></row></sheetData></worksheet>'
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("xl/sharedStrings.xml", shared)
        archive.writestr("xl/worksheets/sheet1.xml", sheet)
    assert prepare.read_xlsx_rows(buffer.getvalue()) == [{"id": "Bold part"}]


def test_build_row_keeps_the_prompt_and_every_criterion_in_column_order() -> None:
    row = {
        "benchmark_id": "CIF-007",
        "prompt": "Write the memo.\n",
        "use_case": "Communication",
        "instruction_type": "Implicit",
        "prompt_style": "Rambling/Stream-of-Consciousness",
        "criterion_10": "Tenth.",
        "criterion_2": " Second. ",
        "criterion_1": "First.",
        "criterion_3": "",
    }
    assert prepare.build_row(row) == {
        "task_id": "CIF-007",
        "use_case": "Communication",
        "instruction_type": "Implicit",
        "prompt_style": "Rambling/Stream-of-Consciousness",
        "rubrics": ["First.", "Second.", "Tenth."],
        "responses_create_params": {"input": [{"role": "user", "content": "Write the memo.\n"}]},
    }


@pytest.mark.parametrize(
    ("row", "error"),
    [
        ({"benchmark_id": "CIF-001", "prompt": "Hi", "criterion_1": " "}, "no rubric criteria"),
        ({"benchmark_id": "", "prompt": "Hi", "criterion_1": "a"}, "benchmark_id and a prompt"),
        ({"benchmark_id": "CIF-001", "prompt": "  ", "criterion_1": "a"}, "benchmark_id and a prompt"),
    ],
)
def test_build_row_rejects_rows_the_judge_cannot_score(row, error) -> None:
    with pytest.raises(ValueError, match=error):
        prepare.build_row(row)


def test_extract_judge_prompt_reads_the_template_and_schema_without_running_the_file() -> None:
    assert prepare.extract_judge_prompt(SCORER_SOURCE) == {
        "judge_prompt_template": "Reply: {{ rubric.response_to_grade }}\nCriterion: {{ rubric.criterion }}",
        "response_schema": {
            "name": "verdict",
            "schema": {
                "type": "object",
                "required": ["score"],
                "properties": {"score": {"type": "string", "enum": ["0", "1"], "description": "pass or fail"}},
                "additionalProperties": False,
            },
        },
    }


@pytest.mark.parametrize(
    ("source", "error"),
    [
        ("RESPONSE_SCHEMA = None\n", "no string literal JUDGE_PROMPT_TEMPLATE"),
        ('JUDGE_PROMPT_TEMPLATE = "{{ rubric.criterion }}"\n', "lacks the placeholders"),
        (SCORER_SOURCE.replace("RESPONSE_SCHEMA", "OTHER_SCHEMA"), "no RESPONSE_SCHEMA"),
        (SCORER_SOURCE.replace('name="verdict",', 'name="verdict", strict=True,'), "unexpected ResponseSchema"),
        (SCORER_SOURCE.replace('["0", "1"]', "list(SCORES)"), "unexpected expression"),
    ],
)
def test_extract_judge_prompt_rejects_unexpected_sources(source, error) -> None:
    with pytest.raises(ValueError, match=error):
        prepare.extract_judge_prompt(source)


def fake_get(responses: dict):
    def get(url, timeout):
        assert timeout == 60
        return MagicMock(content=responses[url])

    return get


def test_prepare_pins_both_sources_and_writes_rows_and_judge_prompt(tmp_path, monkeypatch) -> None:
    workbook, scorer = make_xlsx(TABLE), SCORER_SOURCE.encode()
    monkeypatch.setattr(prepare, "EXPECTED_ROWS", 2)
    monkeypatch.setattr(prepare, "EXPECTED_CRITERIA", 3)
    monkeypatch.setattr(prepare, "DATA_SOURCE_SHA256", hashlib.sha256(workbook).hexdigest())
    monkeypatch.setattr(prepare, "JUDGE_SOURCE_SHA256", hashlib.sha256(scorer).hexdigest())
    output = tmp_path / "complex_constraints_benchmark.jsonl"

    responses = {prepare.DATA_SOURCE_URL: workbook, prepare.JUDGE_SOURCE_URL: scorer}
    with patch.object(prepare.requests, "get", side_effect=fake_get(responses)) as get:
        assert prepare.prepare(output) == output

    assert {call.args[0] for call in get.call_args_list} == set(responses)
    assert prepare.DATA_SOURCE_COMMIT in prepare.DATA_SOURCE_URL
    assert prepare.JUDGE_SOURCE_COMMIT in prepare.JUDGE_SOURCE_URL
    written = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["task_id"] for row in written] == ["CIF-001", "CIF-002"]
    assert written[0]["rubrics"] == ["Covers Monday.", "No doubles."]
    assert written[1]["responses_create_params"]["input"][0]["content"] == "Sum the sales."
    assert "café" in output.read_text(encoding="utf-8")
    judge = json.loads((tmp_path / prepare.JUDGE_PROMPT_FPATH.name).read_text(encoding="utf-8"))
    assert judge["source"] == prepare.JUDGE_SOURCE_URL
    assert judge["response_schema"]["name"] == "verdict"
    assert judge["judge_prompt_template"].startswith("Reply: {{ rubric.response_to_grade }}")


def test_download_refuses_changed_content() -> None:
    with patch.object(prepare.requests, "get", return_value=MagicMock(content=b"tampered")):
        with pytest.raises(ValueError, match="expected"):
            prepare.download(prepare.DATA_SOURCE_URL, prepare.DATA_SOURCE_SHA256)


@pytest.mark.parametrize(
    ("table", "error"),
    [
        (TABLE, "expected 75 rows and 1559 criteria, got 2 and 3"),
        ([HEADER, TABLE[1], TABLE[1]], "not unique"),
    ],
)
def test_prepare_checks_counts_and_ids(tmp_path, monkeypatch, table, error) -> None:
    workbook = make_xlsx(table)
    monkeypatch.setattr(prepare, "DATA_SOURCE_SHA256", hashlib.sha256(workbook).hexdigest())
    if error == "not unique":
        monkeypatch.setattr(prepare, "EXPECTED_ROWS", 2)
        monkeypatch.setattr(prepare, "EXPECTED_CRITERIA", 4)
    with patch.object(prepare.requests, "get", return_value=MagicMock(content=workbook)):
        with pytest.raises(ValueError, match=error):
            prepare.prepare(tmp_path / "out.jsonl")
