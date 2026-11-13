from __future__ import annotations
import hashlib, heapq, json, random, re, statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

SCHEMA="bcnlt-native-foundation-heldout-comparison-v1"
BEGIN="<|bcnlt:record:begin|>\n"; END="\n<|bcnlt:record:end|>"

def _sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
def _canon(v): return hashlib.sha256(json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def _wizard_digest(v): return hashlib.sha256(json.dumps(v,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
def _body(r): return r["prompt"]+"\n"+r["target"]
def _exact(s): return hashlib.sha256(s.replace("\r\n","\n").replace("\r","\n").strip().encode()).hexdigest()
def _near(s):
    s=s.replace("\r\n","\n").replace("\r","\n").strip()
    s=re.sub(r"(?<![\w.])(?:0[xX][0-9a-fA-F]+|\d+(?:\.\d+)?)(?![\w.])","<num>",s)
    return hashlib.sha256(re.sub(r"\s+"," ",s).encode()).hexdigest()
def _docs(r): return tuple(x for x in (r.get("source_document_ids") or []) if isinstance(x,str))

def _sample_fingerprints(directory):
    exact=set(); near=set()
    for p in sorted(Path(directory).glob("*.txt")):
        for part in p.read_text(encoding="utf-8").split(BEGIN)[1:]:
            if END in part:
                s=part.split(END,1)[0]; exact.add(_exact(s)); near.add(_near(s))
    return exact,near

def select_heldout_records(*,accepted_corpus_path,induction_sample_dir,languages,records_per_language,seed,
                           english_train_path=None,english_eval_path=None,english_manifest_path=None,
                           english_records=0):
    exact,near=_sample_fingerprints(induction_sample_dir); source=Path(accepted_corpus_path)
    digest=hashlib.sha256(); train_docs=set()
    for raw in source.open("rb"):
        digest.update(raw); r=json.loads(raw)
        if r.get("split")=="train": train_docs.update(_docs(r))
    heaps={x:[] for x in languages}; reject=Counter(); seen=set(); held=eligible=0
    for raw in source.open("rb"):
        r=json.loads(raw)
        if r.get("split") not in ("test","validation") or r.get("language") not in heaps: continue
        held+=1; rid=r.get("record_id"); p=r.get("prompt"); t=r.get("target")
        if not all(isinstance(x,str) and x for x in (rid,p,t)): reject["malformed_heldout_record"]+=1; continue
        if rid in seen: reject["duplicate_record_id"]+=1; continue
        seen.add(rid)
        if any(x in train_docs for x in _docs(r)): reject["train_source_document_overlap"]+=1; continue
        b=_body(r); e=_exact(b); n=_near(b)
        if e in exact: reject["exact_induction_duplicate"]+=1; continue
        if n in near: reject["near_induction_duplicate"]+=1; continue
        eligible+=1
        item={"record_id":rid,"language":r["language"],"split":r["split"],"prompt":p,"target":t,
              "template_id":r.get("template_id"),"source_document_ids":list(_docs(r)),
              "content_sha256":e,"provenance_sha256":_canon(r.get("provenance") or {})}
        score=int.from_bytes(hashlib.sha256(f"{seed}:{rid}:{e}".encode()).digest(),"big")
        h=heaps[r["language"]]; heapq.heappush(h,(-score,rid,item))
        if len(h)>records_per_language: heapq.heappop(h)
    selected=[]
    for lang in languages:
        if len(heaps[lang])!=records_per_language: raise ValueError(f"insufficient clean heldout records for {lang}")
        selected.extend(x[2] for x in heaps[lang])
    english_status="missing_true_english_holdout"; english_count=0; english_hashes={}
    if english_train_path and english_eval_path and english_manifest_path and english_records:
        train=json.loads(Path(english_train_path).read_text(encoding="utf-8"))
        ev=json.loads(Path(english_eval_path).read_text(encoding="utf-8"))
        em=json.loads(Path(english_manifest_path).read_text(encoding="utf-8"))
        if em.get("train")!=len(train) or em.get("eval")!=len(ev):
            raise ValueError("English split counts do not match manifest")
        if em.get("train_sha256")!=_wizard_digest(train) or em.get("eval_sha256")!=_wizard_digest(ev):
            raise ValueError("English split hashes do not match manifest")
        train_fp={_exact(x["instruction"]+"\n"+x["output"]) for x in train}
        train_ids={x.get("id") for x in train if isinstance(x.get("id"),str)}
        candidates=[]
        for i,x in enumerate(ev):
            b=x.get("instruction","")+"\n"+x.get("output",""); e=_exact(b)
            rid=x.get("id") or f"english-eval-{i:06d}"
            if e in train_fp or rid in train_ids: continue
            item={"record_id":rid,"language":"english","split":"eval","prompt":x["instruction"],"target":x["output"],"content_sha256":e}
            score=int.from_bytes(hashlib.sha256(f"{seed}:{rid}:{e}".encode()).digest(),"big")
            candidates.append((score,item))
        candidates.sort(key=lambda x:x[0]); chosen=[x[1] for x in candidates[:english_records]]
        if len(chosen)!=english_records: raise ValueError("insufficient disjoint English eval records")
        selected.extend(chosen); english_count=len(chosen); english_status="independent_eval_manifest_verified"
        english_hashes={"english_train_sha256":_sha(english_train_path),"english_eval_sha256":_sha(english_eval_path),
                        "english_manifest_sha256":_sha(english_manifest_path)}
    selected.sort(key=lambda x:(x["language"],x["record_id"]))
    manifest={"schema_version":SCHEMA,"seed":seed,"records_per_language":records_per_language,
      "languages":list(languages),"selected_records":len(selected),"selection_sha256":_canon(selected),
      "accepted_corpus_sha256":digest.hexdigest(),"induction_sample_manifest_sha256":_sha(Path(induction_sample_dir)/"manifest.json"),
      "heldout_records_seen":held,"heldout_records_eligible_after_leakage_checks":eligible,
      "train_source_document_count":len(train_docs),"rejections":dict(sorted(reject.items())),
      "english_status":english_status,"english_records_selected":english_count,
      "leakage_controls":{"allowed_splits":["test","validation"],"train_source_document_overlap_rejected":True,
      "exact_induction_text_duplicate_rejected":True,"normalized_near_duplicate_rejected":True},**english_hashes}
    return selected,manifest

def _aggregate(rows):
    b=sum(x["bytes"] for x in rows); p=sum(x["primary_tokens"] for x in rows); c=sum(x["control_tokens"] for x in rows)
    d=[x["paired_token_savings"] for x in rows]
    return {"records":len(rows),"bytes":b,"primary_tokens":p,"control_tokens":c,"paired_token_savings":sum(d),
      "mean_paired_token_savings":statistics.fmean(d),"median_paired_token_savings":statistics.median(d),
      "primary_bytes_per_token":b/p,"control_bytes_per_token":b/c,
      "primary_unknown_emissions":sum(x["primary_unknown_emissions"] for x in rows),
      "control_unknown_emissions":sum(x["control_unknown_emissions"] for x in rows),
      "primary_roundtrip_failures":sum(not x["primary_roundtrip_exact"] for x in rows),
      "control_roundtrip_failures":sum(not x["control_roundtrip_exact"] for x in rows),
      "primary_wins":sum(x>0 for x in d),"ties":sum(x==0 for x in d),"control_wins":sum(x<0 for x in d)}

def compare_tokenizers(*,records:Sequence[dict],primary_tokenizer,control_tokenizer,structural_tokens,seed,bootstrap_samples,model_width):
    pu=primary_tokenizer.token_to_id("<|unk|>"); cu=control_tokenizer.token_to_id("<|unk|>"); rows=[]
    for r in records:
        s=_body(r); pi=list(primary_tokenizer.encode(s).ids); ci=list(control_tokenizer.encode(s).ids); b=len(s.encode())
        rows.append({"record_id":r["record_id"],"language":r["language"],"split":r["split"],"bytes":b,
          "primary_tokens":len(pi),"control_tokens":len(ci),"paired_token_savings":len(ci)-len(pi),
          "primary_bytes_per_token":b/len(pi),"control_bytes_per_token":b/len(ci),
          "primary_roundtrip_exact":primary_tokenizer.decode(pi,skip_special_tokens=False)==s,
          "control_roundtrip_exact":control_tokenizer.decode(ci,skip_special_tokens=False)==s,
          "primary_unknown_emissions":pi.count(pu) if pu is not None else 0,
          "control_unknown_emissions":ci.count(cu) if cu is not None else 0,"content_sha256":r.get("content_sha256") or _exact(s)})
    groups=defaultdict(list)
    for r in rows: groups[r["language"]].append(r)
    dif=[r["paired_token_savings"] for r in rows]; rng=random.Random(seed); n=len(dif)
    means=sorted(statistics.fmean(dif[rng.randrange(n)] for _ in range(n)) for _ in range(bootstrap_samples))
    pv=primary_tokenizer.get_vocab_size(); cv=control_tokenizer.get_vocab_size(); extra=pv-cv
    return {"schema_version":SCHEMA,"per_record":rows,"per_language":{k:_aggregate(v) for k,v in sorted(groups.items())},
      "overall":_aggregate(rows),"paired_uncertainty":{"method":"paired_bootstrap_mean_token_savings","seed":seed,
      "samples":bootstrap_samples,"mean":statistics.fmean(dif),"confidence_level":.95,
      "ci_low":means[max(0,int(.025*bootstrap_samples)-1)],"ci_high":means[min(bootstrap_samples-1,int(.975*bootstrap_samples))]},
      "structural_atomicity":{"tokens_checked":len(structural_tokens),
      "primary_atomic_tokens":sum(len(primary_tokenizer.encode(x).ids)==1 for x in structural_tokens),
      "control_atomic_tokens":sum(len(control_tokenizer.encode(x).ids)==1 for x in structural_tokens),
      "primary_passed":all(len(primary_tokenizer.encode(x).ids)==1 for x in structural_tokens),
      "control_passed":all(len(control_tokenizer.encode(x).ids)==1 for x in structural_tokens)},
      "resource_implications":{"primary_vocabulary_size":pv,"control_vocabulary_size":cv,"extra_vocabulary_rows":extra,
      "model_width":model_width,"extra_tied_embedding_parameters":extra*model_width,
      "extra_tied_embedding_bytes_fp16_or_bf16":extra*model_width*2,"extra_tied_embedding_bytes_fp32":extra*model_width*4},
      "promotion_decision":"not_evaluated"}
