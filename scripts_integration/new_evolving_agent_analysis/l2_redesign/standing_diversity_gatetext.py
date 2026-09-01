"""Re-measure standing-set redundancy with the SAME text the gate embeds."""
import itertools, json, sys
from pathlib import Path
sys.path.insert(0, str(Path("Self-Evolving-Agent").resolve()))
from dotenv import load_dotenv; load_dotenv()
from evolving_common.llm_client import embed_texts_nvidia
from evolving_common.governor.l2_promotion import _entry_dedup_text, _cosine

ROOT=Path("runs_evolving/gpt-oss-120b/median")
for d in sorted(ROOT.glob("*l2redesign*")):
    rows=[json.loads(l) for l in open(d/"l2_standing.jsonl")]
    l1={}
    for line in open(d/"shared_l1.jsonl"):
        e=json.loads(line); l1[str(e.get("entry_id")).strip()]=e
    name=d.name.replace("base_agent_gpt_oss_120b_",""); name=name[:name.find("_itr30")]
    texts, titles, missing = [], [], 0
    for r in rows:
        eid=str(r.get("entry_id")).strip()
        ent=l1.get(eid) or (r.get("entry") if isinstance(r.get("entry"),dict) else None)
        if ent is None: missing+=1; continue
        texts.append(_entry_dedup_text(ent)); titles.append(str(r.get("title"))[:38])
    vecs=embed_texts_nvidia(texts)
    sims=sorted(((_cosine(vecs[a],vecs[b]),titles[a],titles[b])
                 for a,b in itertools.combinations(range(len(texts)),2)),reverse=True)
    n80=sum(1 for s,_,_ in sims if s>=0.80)
    print(f"\n{name}: {len(texts)} rules (missing {missing}), {len(sims)} pairs, "
          f">=0.80: {n80}, max {sims[0][0]:.3f}   [gate's own _entry_dedup_text]")
    for s,t1,t2 in sims[:3]: print(f"    {s:.3f}  {t1:<38} | {t2}")
