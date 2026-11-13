"""Inspect committed admission without starting a parser or creating a database."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bosscrafts_model.admission_progress import read_admission_progress

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, default=ROOT / 'artifacts/corpora/bcnlt-current-spec-65536-full-resumable-v1/admission.sqlite')
    args = parser.parse_args()
    print(json.dumps(read_admission_progress(args.database), indent=2, sort_keys=True))
