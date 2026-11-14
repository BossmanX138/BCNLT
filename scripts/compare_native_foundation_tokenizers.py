from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from bcnlt.languages import CANONICAL_LANGUAGE_IDS
from bcnlt.native_foundation.comparison import compare_tokenizers, select_heldout_records

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf-8")
def jsonl(path,rows):
    Path(path).write_text("".join(json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(",",":"))+"\n" for x in rows),encoding="utf-8")

def main(argv=None):
    p=argparse.ArgumentParser(description="Fair heldout comparison of isolated BCNLT tokenizers")
    p.add_argument("--accepted-corpus",required=True); p.add_argument("--induction-sample-dir",required=True)
    p.add_argument("--primary-tokenizer",required=True); p.add_argument("--control-tokenizer",required=True)
    p.add_argument("--structural-tokens",required=True); p.add_argument("--output-dir",required=True)
    p.add_argument("--english-train"); p.add_argument("--english-eval"); p.add_argument("--english-manifest")
    p.add_argument("--records-per-language",type=int,default=200); p.add_argument("--english-records",type=int,default=200)
    p.add_argument("--seed",type=int,default=65536); p.add_argument("--bootstrap-samples",type=int,default=2000)
    p.add_argument("--model-width",type=int,default=2048)
    a=p.parse_args(argv); out=Path(a.output_dir)
    if out.exists(): p.error(f"output already exists: {out}")
    out.mkdir(parents=True); write(out/"progress.json",{"status":"running","stage":"selecting_heldout"})
    records,selection=select_heldout_records(accepted_corpus_path=a.accepted_corpus,
      induction_sample_dir=a.induction_sample_dir,languages=CANONICAL_LANGUAGE_IDS,
      records_per_language=a.records_per_language,seed=a.seed,
      english_train_path=a.english_train,english_eval_path=a.english_eval,
      english_manifest_path=a.english_manifest,english_records=a.english_records)
    jsonl(out/"selected_records.jsonl",records); write(out/"selection_manifest.json",selection)
    write(out/"progress.json",{"status":"running","stage":"comparing","records":len(records)})
    from tokenizers import Tokenizer
    primary=Tokenizer.from_file(a.primary_tokenizer); control=Tokenizer.from_file(a.control_tokenizer)
    structural=json.loads(Path(a.structural_tokens).read_text(encoding="utf-8"))["tokens"]
    report=compare_tokenizers(records=records,primary_tokenizer=primary,control_tokenizer=control,
      structural_tokens=tuple(structural),seed=a.seed,bootstrap_samples=a.bootstrap_samples,model_width=a.model_width)
    per=report.pop("per_record"); jsonl(out/"per_record.jsonl",per)
    report["selection_manifest"]=selection
    report["inputs"]={"primary_tokenizer_sha256":sha(a.primary_tokenizer),
      "control_tokenizer_sha256":sha(a.control_tokenizer),"structural_tokens_sha256":sha(a.structural_tokens)}
    report["output_files_sha256"]={"selected_records.jsonl":sha(out/"selected_records.jsonl"),
      "selection_manifest.json":sha(out/"selection_manifest.json"),"per_record.jsonl":sha(out/"per_record.jsonl")}
    report["host_comparison_policy"]="both tokenizers evaluated sequentially in one process on one host"
    write(out/"report.json",report); write(out/"progress.json",{"status":"complete","stage":"complete","records":len(records),"report_sha256":sha(out/"report.json")})
    print(json.dumps({"output":str(out),"records":len(records),"report_sha256":sha(out/"report.json")},sort_keys=True))
    return 0
if __name__=="__main__": raise SystemExit(main())
