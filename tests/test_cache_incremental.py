from bcnlt.cache import ProjectIndex
from bcnlt.incremental import RegionInvalidator


def test_project_index_tracks_symbol_definitions_and_references():
    index = ProjectIndex()
    index.define("sample.py", "module.fn.x", "x", (2, 4, 2, 5))
    index.reference("sample.py", "module.fn.x", "x", (3, 11, 3, 12))

    assert index.find_definition("x").symbol_id == "module.fn.x"
    assert index.references_to("module.fn.x")[0].surface == "x"


def test_region_invalidator_marks_only_changed_files_and_line_ranges():
    invalidator = RegionInvalidator()

    invalidator.mark_changed("sample.py", start_line=4, end_line=6)

    assert invalidator.changed_files() == {"sample.py"}
    assert invalidator.changed_ranges("sample.py") == [(4, 6)]
