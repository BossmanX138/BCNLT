import hashlib
import sqlite3
import tempfile
import unittest
from pathlib import Path

from bosscrafts_model.admission_progress import (
    CURRENT_ADMISSION_SCHEMA,
    read_admission_progress,
)
from bosscrafts_model.resumable_current_corpus import RESUMABLE_SCHEMA

ROOT = Path(__file__).resolve().parents[1]

class AdmissionProgressTests(unittest.TestCase):
    def test_current_schema_tracks_admission_writer(self):
        self.assertEqual(CURRENT_ADMISSION_SCHEMA, RESUMABLE_SCHEMA)

    def test_missing_database_is_not_created(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'artifacts') as d:
            p = Path(d) / 'missing.sqlite'
            with self.assertRaises(FileNotFoundError):
                read_admission_progress(p)
            self.assertFalse(p.exists())

    def _assert_reports_snapshot_without_modifying_database(self, schema_version: str):
        with tempfile.TemporaryDirectory(dir=ROOT / 'artifacts') as d:
            p = Path(d) / 'admission.sqlite'
            with sqlite3.connect(p) as c:
                c.execute('CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT)')
                c.executemany('INSERT INTO metadata VALUES(?,?)', [('schema_version',schema_version),('catalog_exhausted','false')])
                c.execute('CREATE TABLE documents(status TEXT,language TEXT,reason_code TEXT,semantic_token_count INTEGER,raw_token_count INTEGER)')
                c.executemany('INSERT INTO documents VALUES(?,?,?,?,?)', [('admitted','python',None,20,5),('rejected','rust','semantic_parse_failure',0,0)])
            c.close()
            before = hashlib.sha256(p.read_bytes()).hexdigest()
            result = read_admission_progress(p)
            self.assertEqual(result['schema_version'], schema_version)
            self.assertEqual(result['processed_total'], 2)
            self.assertEqual(result['admitted_count'], 1)
            self.assertEqual(result['rejected_count'], 1)
            self.assertEqual(result['semantic_token_count'], 20)
            self.assertFalse(result['catalog_exhausted'])
            self.assertFalse(result['training_ready'])
            self.assertEqual(result['rejection_reasons'], {'semantic_parse_failure': 1})
            self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(), before)

    def test_reports_v1_snapshot_without_modifying_database(self):
        self._assert_reports_snapshot_without_modifying_database(
            'bosscrafts-current-corpus-resumable-v1'
        )

    def test_reports_v4_snapshot_without_modifying_database(self):
        self._assert_reports_snapshot_without_modifying_database(
            'bosscrafts-current-corpus-resumable-v4'
        )

    def test_unknown_schema_fails_closed(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'artifacts') as d:
            p = Path(d) / 'bad.sqlite'
            with sqlite3.connect(p) as c:
                c.execute('CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT)')
                c.execute("INSERT INTO metadata VALUES('schema_version','other')")
            c.close()
            with self.assertRaisesRegex(ValueError, 'schema'):
                read_admission_progress(p)

