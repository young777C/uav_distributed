#!/usr/bin/env python3
"""Live terminal dashboard for the ACoT data<->training iteration loop.

Read-only: reads iter/iteration_ledger.jsonl and PROBES live process/GPU state
(docker exec / ps / nvidia-smi) — it never touches the data-gen or training runs,
so it is safe to keep open in a dedicated zellij session:

    zellij -s version_iter -l iter/dashboard.kdl     # create + attach
    zellij attach version_iter                       # re-attach any time

  --once   render a single snapshot and exit (for testing / `watch`).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

HERE = Path(__file__).resolve().parent
LEDGER = HERE / "iteration_ledger.jsonl"
CONTAINER = "cyh-carla"
REFRESH = 3.0
GATE_ICON = {"pass": "🟢", "fail": "🔴", "waived": "🟡", "pending": "…",
             "low": "🟡", "half": "🟡"}


def sh(cmd: str, timeout: float = 5) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=timeout).stdout.strip()
    except Exception:
        return ""


def dexec(inner: str, timeout: float = 5) -> str:
    return sh(f"docker exec {CONTAINER} bash -lc {json.dumps(inner)}", timeout)


def read_ledger() -> list:
    recs = []
    if LEDGER.exists():
        for ln in LEDGER.read_text().splitlines():
            ln = ln.strip()
            if ln:
                try:
                    recs.append(json.loads(ln))
                except Exception:
                    pass
    return recs


def probe() -> dict:
    st = {}
    dg = dexec("ps -eo args | grep -E '[r]un_resilient|[g]enerate_batch' | head -1")
    st["datagen_cmd"] = dg
    st["datagen_prog"] = ""
    if dg:
        m = re.search(r"run_resilient\.sh\s+\S+\s+(\S+)\s+(\d+)", dg) or \
            re.search(r"--output\s+(\S+).*?--num-episodes\s+(\d+)", dg)
        if m:
            outd, n = m.group(1), m.group(2)
            cnt = dexec(f"ls {outd}/*.h5 2>/dev/null | wc -l").strip() or "0"
            st["datagen_prog"] = f"{cnt}/{n}  → {outd}"
    st["train_cmd"] = sh("ps -eo args | grep '[t]rain.stage2' | head -1")
    st["train_log"] = sh(f"tail -1 {HERE.parent}/runs/stage2/train.log 2>/dev/null")
    st["gpu"] = sh("nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader")
    st["carla"] = dexec("ps -eo args | grep -oE 'rpc-port=[0-9]+' | sort -u | tr '\\n' ' '")
    return st


def _pct(x):
    return f"{x*100:.0f}%" if isinstance(x, (int, float)) else "—"


def render(recs: list, st: dict):
    now = datetime.now()
    cyc = next((r for r in reversed(recs) if r["type"] == "cycle"), None)  # latest cycle
    cyc_txt = "—"
    if cyc:
        try:
            start = datetime.strptime(cyc["ts"], "%Y-%m-%dT%H:%M")
            days = cyc.get("config", {}).get("days", 2)
            rem = max(start + timedelta(days=days) - now, timedelta())
            cyc_txt = f"周期{days}天 · 剩余 {rem.days}d{rem.seconds // 3600}h"
        except Exception:
            pass
    cur_round = max([r.get("round", 0) for r in recs] or [0])

    stage = "数据生成" if st["datagen_cmd"] else "训练" if st["train_cmd"] else "待命/问题回填"
    loop = Text()
    for i, s in enumerate(["数据生成", "门禁QC", "训练", "MVP验证", "问题回填"]):
        loop.append(f" ●{s} " if s == stage else f" {s} ",
                    style="bold black on green" if s == stage else "dim")
        loop.append("→" if i < 4 else "↺", style="dim")
    header = Panel(Group(
        Text(f"ACoT 数据↔训练 迭代看板    {cyc_txt}    · 第 {cur_round} 轮 · {now:%m-%d %H:%M}",
             style="bold cyan"), loop), border_style="cyan")

    pt = Table.grid(padding=(0, 2))
    pt.add_column(style="bold cyan", justify="right")
    pt.add_column()
    pt.add_row("数据生成", f"[green]● 运行中[/]  {st['datagen_prog']}"
               if st["datagen_cmd"] else "[dim]空闲[/]")
    pt.add_row("训练", f"[green]● 运行中[/]  {st['train_log']}"
               if st["train_cmd"] else "[dim]空闲[/]")
    pt.add_row("CARLA", st["carla"] or "[dim]无 server[/]")
    gpu = Text()
    for line in (st["gpu"] or "").splitlines():
        try:
            idx, mem, util = [x.strip() for x in line.split(",")]
            busy = int(re.sub(r"\D", "", util)) > 5 or int(mem.split()[0]) > 1500
            gpu.append(f"G{idx}:{util.replace(' %','%')} ", style="green" if busy else "dim")
        except Exception:
            pass
    pt.add_row("GPU", gpu)
    procpanel = Panel(pt, title="当前进程(实时探测)", border_style="green")

    vt = Table(header_style="bold", expand=True)
    for c in ["版本", "轮", "leak", "共视", "出画", "抖动°", "mis_follow", "门禁", "变更"]:
        vt.add_column(c, overflow="fold")
    vers = {}
    for r in recs:
        if r["type"] in ("data_version", "train_result"):
            v = r["version"]
            d = vers.setdefault(v, {"round": r.get("round", 0), "m": {}, "g": {}, "note": ""})
            d["m"].update(r.get("metrics", {}))
            d["g"].update(r.get("gates", {}))
            d["round"] = r.get("round", d["round"])
            if r["type"] == "data_version":
                d["note"] = r.get("note", "")
    for v, d in vers.items():
        m = d["m"]
        jit = m.get("jitter_occ_deg")
        vt.add_row(v, str(d["round"]), _pct(m.get("leak_floor")), _pct(m.get("covis")),
                   _pct(m.get("off_screen")),
                   f"{jit:.2f}" if isinstance(jit, (int, float)) else "—",
                   _pct(m.get("mis_follow")),
                   " ".join(f"{k}{GATE_ICON.get(str(x), x)}" for k, x in d["g"].items()) or "—",
                   d["note"])
    vpanel = Panel(vt, title="版本登记表(账本)", border_style="cyan")

    ptxt = Text()
    for r in [r for r in recs if r["type"] == "problem"][-4:]:
        g = r.get("gates", {})
        tag = " ".join(f"{k}{GATE_ICON.get(str(x), x)}" for k, x in g.items())
        ptxt.append(f"{tag}  {r.get('note','')}\n")
    ppanel = Panel(ptxt if ptxt.plain else Text("—"),
                   title="待办问题 (data-fix-spec: (a)语言指定 (b)位置 (c)连续性 (d)共视)",
                   border_style="yellow")
    return Group(header, procpanel, vpanel, ppanel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    if a.once:
        Console().print(render(read_ledger(), probe()))
        return
    console = Console()
    with Live(console=console, screen=True, refresh_per_second=4) as live:
        while True:
            try:
                live.update(render(read_ledger(), probe()))
            except Exception as e:
                live.update(Text(f"dashboard error: {e}"))
            time.sleep(REFRESH)


if __name__ == "__main__":
    main()
