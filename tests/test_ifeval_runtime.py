from __future__ import annotations

import hashlib
import io
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from bosscrafts_model.ifeval_runtime import (
    RemoteArchivePin,
    RemoteFilePin,
    _install_pinned_ifeval_runtime,
    _verify_pinned_ifeval_runtime,
)


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _zip(files: dict[str, bytes]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return output.getvalue()


def _fixture_pins() -> tuple[
    dict[str, RemoteFilePin],
    dict[str, RemoteArchivePin],
    dict[str, bytes],
]:
    payloads = {
        "https://example.test/evaluation_lib.py": b"VALUE = 1\n",
        "https://example.test/instructions.py": b"VALUE = 2\n",
        "https://example.test/LICENSE": b"Apache fixture\n",
        "https://example.test/punkt.zip": _zip(
            {"punkt/english.pickle": b"punkt-fixture"}
        ),
        "https://example.test/punkt_tab.zip": _zip(
            {"punkt_tab/english/abbrev_types.txt": b"tab-fixture"}
        ),
    }
    files = {
        "instruction_following_eval/evaluation_lib.py": RemoteFilePin(
            url="https://example.test/evaluation_lib.py",
            sha256=_sha256(payloads["https://example.test/evaluation_lib.py"]),
        ),
        "instruction_following_eval/instructions.py": RemoteFilePin(
            url="https://example.test/instructions.py",
            sha256=_sha256(payloads["https://example.test/instructions.py"]),
        ),
        "LICENSES/google-research-Apache-2.0.txt": RemoteFilePin(
            url="https://example.test/LICENSE",
            sha256=_sha256(payloads["https://example.test/LICENSE"]),
        ),
    }
    archives = {
        "punkt": RemoteArchivePin(
            url="https://example.test/punkt.zip",
            sha256=_sha256(payloads["https://example.test/punkt.zip"]),
            required_root="punkt",
            destination="nltk_data/tokenizers",
        ),
        "punkt_tab": RemoteArchivePin(
            url="https://example.test/punkt_tab.zip",
            sha256=_sha256(payloads["https://example.test/punkt_tab.zip"]),
            required_root="punkt_tab",
            destination="nltk_data/tokenizers",
        ),
    }
    return files, archives, payloads


def test_installs_and_reverifies_complete_no_clobber_runtime(tmp_path: Path) -> None:
    files, archives, payloads = _fixture_pins()
    fetched: list[str] = []

    def fetch(url: str) -> bytes:
        fetched.append(url)
        return payloads[url]

    root = tmp_path / "runtime"
    installed = _install_pinned_ifeval_runtime(
        root,
        evaluator_revision="a" * 40,
        nltk_revision="b" * 40,
        file_pins=files,
        archive_pins=archives,
        fetch_bytes=fetch,
    )

    assert installed.root == root.resolve()
    assert fetched == [
        pin.url for pin in (*files.values(), *archives.values())
    ]
    assert (root / "instruction_following_eval" / "__init__.py").is_file()
    assert (
        root / "nltk_data" / "tokenizers" / "punkt" / "english.pickle"
    ).read_bytes() == b"punkt-fixture"
    assert (
        root
        / "nltk_data"
        / "tokenizers"
        / "punkt_tab"
        / "english"
        / "abbrev_types.txt"
    ).read_bytes() == b"tab-fixture"
    reloaded = _verify_pinned_ifeval_runtime(
        root,
        evaluator_revision="a" * 40,
        nltk_revision="b" * 40,
        file_pins=files,
        archive_pins=archives,
    )
    assert reloaded.manifest_sha256 == installed.manifest_sha256

    with pytest.raises(ValueError, match="already exists"):
        _install_pinned_ifeval_runtime(
            root,
            evaluator_revision="a" * 40,
            nltk_revision="b" * 40,
            file_pins=files,
            archive_pins=archives,
            fetch_bytes=fetch,
        )

    target = root / "instruction_following_eval" / "instructions.py"
    target.write_bytes(b"tampered\n")
    with pytest.raises(ValueError, match="integrity|SHA-256"):
        _verify_pinned_ifeval_runtime(
            root,
            evaluator_revision="a" * 40,
            nltk_revision="b" * 40,
            file_pins=files,
            archive_pins=archives,
        )


def test_rejects_archive_traversal_and_source_digest_mismatch(tmp_path: Path) -> None:
    files, archives, payloads = _fixture_pins()
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(ValueError, match="already exists"):
        _install_pinned_ifeval_runtime(
            existing,
            evaluator_revision="a" * 40,
            nltk_revision="b" * 40,
            file_pins=files,
            archive_pins=archives,
            fetch_bytes=payloads.__getitem__,
        )

    payloads["https://example.test/punkt.zip"] = _zip(
        {"punkt/../../escape.txt": b"escape"}
    )
    archives["punkt"] = RemoteArchivePin(
        url="https://example.test/punkt.zip",
        sha256=_sha256(payloads["https://example.test/punkt.zip"]),
        required_root="punkt",
        destination="nltk_data/tokenizers",
    )

    with pytest.raises(ValueError, match="archive path"):
        _install_pinned_ifeval_runtime(
            tmp_path / "unsafe",
            evaluator_revision="a" * 40,
            nltk_revision="b" * 40,
            file_pins=files,
            archive_pins=archives,
            fetch_bytes=payloads.__getitem__,
        )

    files, archives, payloads = _fixture_pins()
    files["instruction_following_eval/evaluation_lib.py"] = RemoteFilePin(
        url="https://example.test/evaluation_lib.py",
        sha256="0" * 64,
    )
    with pytest.raises(ValueError, match="SHA-256"):
        _install_pinned_ifeval_runtime(
            tmp_path / "bad-digest",
            evaluator_revision="a" * 40,
            nltk_revision="b" * 40,
            file_pins=files,
            archive_pins=archives,
            fetch_bytes=payloads.__getitem__,
        )


def test_verifier_rejects_duplicate_manifest_keys_and_symlink_inventory(
    tmp_path: Path,
) -> None:
    files, archives, payloads = _fixture_pins()
    root = tmp_path / "runtime"
    _install_pinned_ifeval_runtime(
        root,
        evaluator_revision="a" * 40,
        nltk_revision="b" * 40,
        file_pins=files,
        archive_pins=archives,
        fetch_bytes=payloads.__getitem__,
    )
    manifest = root / "manifest.json"
    rendered = manifest.read_text(encoding="utf-8")
    manifest.write_text(
        rendered.replace(
            '"schema_version": "bosscrafts-ifeval-runtime-v1",',
            '"schema_version": "bosscrafts-ifeval-runtime-v1",\n'
            '  "schema_version": "bosscrafts-ifeval-runtime-v1",',
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate|manifest"):
        _verify_pinned_ifeval_runtime(
            root,
            evaluator_revision="a" * 40,
            nltk_revision="b" * 40,
            file_pins=files,
            archive_pins=archives,
        )

    if hasattr(Path, "symlink_to"):
        root2 = tmp_path / "runtime-symlink"
        _install_pinned_ifeval_runtime(
            root2,
            evaluator_revision="a" * 40,
            nltk_revision="b" * 40,
            file_pins=files,
            archive_pins=archives,
            fetch_bytes=payloads.__getitem__,
        )
        try:
            (root2 / "extra-link").symlink_to(root2 / "manifest.json")
        except OSError:
            pytest.skip("this host does not permit unprivileged symlink creation")
        with pytest.raises(ValueError, match="file inventory|type"):
            _verify_pinned_ifeval_runtime(
                root2,
                evaluator_revision="a" * 40,
                nltk_revision="b" * 40,
                file_pins=files,
                archive_pins=archives,
            )


def test_runtime_installer_script_exposes_no_clobber_output() -> None:
    project_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "install_ifeval_runtime.py"),
            "--help",
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--output-root" in completed.stdout
