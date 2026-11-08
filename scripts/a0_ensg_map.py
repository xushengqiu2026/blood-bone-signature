# -*- coding: utf-8 -*-
"""GSE187429 的 Ensembl gene ID -> HGNC symbol（mygene.info 批量查询，结果缓存）"""
import os, time, pickle, requests
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(BASE, "data")
CACHE = os.path.join(DATA, "ensg2symbol_GSE187429.pkl")
tpm = pd.read_csv(os.path.join(DATA, "GSE187429_TPM.tsv.gz"), sep="\t", index_col=0)
keep = tpm[(tpm > 1).sum(axis=1) >= 4]
ids = [str(g).split(".")[0] for g in keep.index]
print("需要映射的 ENSG:", len(ids), flush=True)

out = {}
if os.path.exists(CACHE):
    out = pickle.load(open(CACHE, "rb"))
    print("已有缓存", len(out), flush=True)
todo = [g for g in ids if g not in out]
S = requests.Session(); S.headers.update({"User-Agent": "Mozilla/5.0"})
for k in range(0, len(todo), 400):
    batch = todo[k:k+400]
    for attempt in range(3):
        try:
            r = S.post("https://mygene.info/v3/query",
                       data={"q": ",".join(batch), "scopes": "ensembl.gene",
                             "fields": "symbol", "species": "human"}, timeout=40)
            if r.status_code == 200:
                for j in r.json():
                    if "symbol" in j:
                        out[j["query"]] = str(j["symbol"]).upper()
                break
        except Exception as e:
            print("  retry", type(e).__name__, flush=True)
        time.sleep(2)
    out.update({g: None for g in batch if g not in out})
    print(f"  {k+len(batch)}/{len(todo)} mapped {sum(1 for v in out.values() if v)}", flush=True)
    pickle.dump(out, open(CACHE, "wb"))
    time.sleep(0.3)
print("映射成功:", sum(1 for v in out.values() if v), "/", len(out), flush=True)
