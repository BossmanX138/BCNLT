"""Read-only committed admission progress; never an authorization to train."""
from pathlib import Path
import sqlite3


def read_admission_progress(database: str | Path) -> dict:
    path = Path(database).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    connection = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True, timeout=10)
    try:
        connection.execute('PRAGMA query_only=ON')
        connection.execute('BEGIN')
        metadata = dict(connection.execute('SELECT key,value FROM metadata'))
        if metadata.get('schema_version') != 'bosscrafts-current-corpus-resumable-v1':
            raise ValueError('unsupported admission schema')
        if metadata.get('catalog_exhausted') not in ('true', 'false'):
            raise ValueError('invalid catalog exhaustion marker')
        rows = connection.execute('SELECT status,COUNT(*),COALESCE(SUM(semantic_token_count),0),COALESCE(SUM(raw_token_count),0) FROM documents GROUP BY status').fetchall()
        if any(row[0] not in ('admitted','rejected') for row in rows):
            raise ValueError('invalid document status')
        counts = {row[0]: row[1] for row in rows}
        return {
            'database_path': str(path),
            'processed_total': sum(counts.values()),
            'admitted_count': counts.get('admitted', 0),
            'rejected_count': counts.get('rejected', 0),
            'semantic_token_count': sum(row[2] for row in rows),
            'raw_token_count': sum(row[3] for row in rows),
            'catalog_exhausted': metadata['catalog_exhausted'] == 'true',
            'by_language': dict(connection.execute('SELECT language,COUNT(*) FROM documents GROUP BY language')),
            'rejection_reasons': dict(connection.execute("SELECT reason_code,COUNT(*) FROM documents WHERE status='rejected' GROUP BY reason_code")),
            'training_ready': False,
            'scope': 'Committed snapshot only; no parser, provenance, packing or training readiness validation.',
        }
    finally:
        connection.close()
