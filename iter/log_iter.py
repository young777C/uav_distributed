#!/usr/bin/env python3
"""Append one event to the ACoT data<->training iteration ledger.

Both agents call this to record what they did, so the dashboard has a single
source of truth. Ledger = iter/iteration_ledger.jsonl (append-only, one JSON/line).

Examples:
  # data side announces a new dataset version
  python iter/log_iter.py --type data_version --round 3 --agent data --version mvp_v3 \
      --config band=10,recycle=1,recovery=1,all_car=1,yaw_deadband=5 \
      --metrics episodes=8,leak_floor=0.44,covis=0.53,off_screen=0.12,jitter_occ_deg=0.59 \
      --gates leak=waived,covis=pass,language=pending \
      --note "accept 44% position; recovery+covis+jitter fixed" \
      --artifact videos=/data/mvp_check/videos

  # training side reports an MVP-validation result on a version
  python iter/log_iter.py --type train_result --round 3 --agent training --version mvp_v3 \
      --metrics mis_follow=0.03,act_mse=0.05,lang_gap=0.001 --gates language=fail \
      --note "no-lang == with-lang -> language not load-bearing yet"

  # set / reset the iteration cycle
  python iter/log_iter.py --type cycle --config days=2 --note "round 3 start"
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

LEDGER = Path(__file__).with_name("iteration_ledger.jsonl")


def kv(s: str) -> dict:
    """'a=1,b=2.5,c=pass,d=true' -> {'a':1,'b':2.5,'c':'pass','d':True}"""
    out = {}
    for part in (s or "").split(","):
        part = part.strip()
        if not part or "=" not in part:
            continue
        k, v = part.split("=", 1)
        k, v = k.strip(), v.strip()
        if v.lower() in ("true", "false"):
            v = v.lower() == "true"
        else:
            for cast in (int, float):
                try:
                    v = cast(v); break
                except ValueError:
                    pass
        out[k] = v
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", required=True,
                    help="data_version | train_result | gate | problem | cycle | note")
    ap.add_argument("--agent", default="data")
    ap.add_argument("--round", type=int, default=0)
    ap.add_argument("--version", default="")
    ap.add_argument("--metrics", default="", help="k=v,k=v (numbers/bools auto-cast)")
    ap.add_argument("--gates", default="", help="k=v,k=v e.g. leak=pass,language=pending")
    ap.add_argument("--config", default="", help="k=v,k=v of the generation config")
    ap.add_argument("--artifact", default="", help="k=path,k=path")
    ap.add_argument("--note", default="")
    ap.add_argument("--ts", default="", help="override timestamp (YYYY-mm-ddTHH:MM); else now")
    a = ap.parse_args()
    rec = {
        "ts": a.ts or datetime.now().strftime("%Y-%m-%dT%H:%M"),
        "type": a.type, "agent": a.agent, "round": a.round, "version": a.version,
        "metrics": kv(a.metrics), "gates": kv(a.gates), "config": kv(a.config),
        "artifacts": kv(a.artifact), "note": a.note,
    }
    with open(LEDGER, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[ledger] +{rec['type']} {rec['version']} r{rec['round']}")


if __name__ == "__main__":
    main()
