import json

import pytest

from bosscrafts_model.jsonl import split_jsonl_records


@pytest.mark.parametrize("separator", ("\u2028", "\u2029", "\u0085"))
def test_unicode_line_separators_remain_inside_one_json_record(separator: str):
    payload = json.dumps({"value": f"before{separator}after"}, ensure_ascii=False)
    lines = split_jsonl_records(f"{payload}\n")
    assert lines == (payload,)
    assert json.loads(lines[0]) == {"value": f"before{separator}after"}


def test_trailing_lf_is_optional_but_empty_interior_record_is_rejected():
    assert split_jsonl_records('{"a":1}') == ('{"a":1}',)
    assert split_jsonl_records('{"a":1}\n') == ('{"a":1}',)
    with pytest.raises(ValueError, match="empty record"):
        split_jsonl_records('{"a":1}\n\n{"b":2}\n')
