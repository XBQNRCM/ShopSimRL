#!/usr/bin/env python3
"""Check public data, local documentation links, and the built bilingual site."""
from html.parser import HTMLParser
import hashlib
import csv
import gzip
import json
import math
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links, self.ids, self.translations = [], set(), 0

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if "id" in a:
            if a["id"] in self.ids:
                raise ValueError(f"Duplicate HTML id: {a['id']}")
            self.ids.add(a["id"])
        for key in ["src", "href"]:
            if key in a:
                self.links.append(a[key])
        if "data-en" in a:
            assert a.get("data-zh") and a["data-en"], "Missing bilingual text"
            self.translations += 1
        if tag == "img":
            assert a.get("alt"), "Image needs an accessible description"


def main():
    errors = []
    docs = list((ROOT / "docs").rglob("*.md")) + [ROOT / "README.md", ROOT / "README.zh-CN.md"]
    docs += list((ROOT / "artifacts").rglob("README.md"))
    for path in docs:
        content = re.sub(r"```.*?```", "", path.read_text(encoding="utf-8"), flags=re.S)
        for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", content):
            url = urlsplit(target)
            if url.scheme or url.netloc or not url.path:
                continue
            if not (path.parent / unquote(url.path)).exists():
                errors.append(f"Broken Markdown link: {path.relative_to(ROOT)} -> {target}")
    pages={}
    for path in (ROOT / "_site").rglob("*.html"):
        page=Page();content=path.read_text(encoding="utf-8");page.feed(content)
        assert 'MATHPLACEHOLDER' not in content and '{{behavior_' not in content
        pages[path.resolve()]=page
    assert pages[(ROOT / "_site/index.html").resolve()].translations >= 100
    for path,page in pages.items():
        for target in page.links:
            url=urlsplit(target)
            if url.scheme or url.netloc:continue
            destination=(path.parent/unquote(url.path)).resolve() if url.path else path
            if not destination.exists():errors.append(f"Missing site asset: {path.relative_to(ROOT)} -> {target}")
            if url.fragment and destination in pages and unquote(url.fragment) not in pages[destination].ids:
                errors.append(f"Missing site anchor: {path.relative_to(ROOT)} -> {target}")
    manifest = json.loads((ROOT / "_site/build-manifest.json").read_text(encoding="utf-8"))
    for path, digest in manifest.items():
        assert hashlib.sha256((ROOT / "_site" / path).read_bytes()).hexdigest() == digest
    for svg in (ROOT / "_site").rglob("*.svg"):
        ET.parse(svg)
    d = json.loads((ROOT / "artifacts/analysis/analysis.json").read_text(encoding="utf-8"))
    outcomes = json.loads((ROOT / "artifacts/analysis/test_task_outcomes.json").read_text(encoding="utf-8"))
    assert len(outcomes) == 400 and len({r['task_id'] for r in outcomes}) == 400
    for arm, summary in d["test"].items():
        for metric, key in [("reward", "reward_mean"), ("success", "success_rate")]:
            mean = sum(r[arm][metric] for r in outcomes)/len(outcomes)
            assert math.isclose(mean, summary["primary"][key], abs_tol=1e-12)
        success = f"{summary['primary']['success_rate']*100:.2f}%"
        reward = f"{summary['primary']['reward_mean']:.4f}"
        for path in [ROOT / "README.md", ROOT / "README.zh-CN.md", ROOT / "site/index.html"]:
            text = path.read_text(encoding="utf-8")
            assert success in text and reward in text, f"Outdated numbers: {path.name}"
    provenance = json.loads((ROOT / "artifacts/analysis/provenance.json").read_text(encoding="utf-8"))
    for path, digest in provenance["inputs"].items():
        assert hashlib.sha256((ROOT / path).read_bytes().replace(b'\r\n',b'\n')).hexdigest() == digest, f"Analysis input changed: {path}"
    assert [len(b["skills"]) for b in d["skill_banks"]] == [10,7,5,8,8]
    assert sum(r["generated_episodes"] for r in d["rounds"]) == 20480
    assert sum(r["trainer_episodes"] for r in d["rounds"]) == 10240
    with (ROOT / "artifacts/analysis/training_steps.csv").open(encoding="utf-8") as f:
        assert [int(r["step"]) for r in csv.DictReader(f)]==list(range(1,81))
    behavior_dir=ROOT / "artifacts/analysis/behavior"
    behavior=json.loads((behavior_dir / "behavior.json").read_text(encoding="utf-8"))
    assert behavior["extraction"]["train_turns"]==110997
    assert sum(r["all"]["n"]+r["technical_n"] for r in behavior["rounds"])==20480
    assert all(r["all"]["n"]==r["all"]["token_complete_episodes"] for r in behavior["rounds"])
    assert all(v["n"]==v["token_complete_episodes"]==400 for v in behavior["evaluation"].values())
    for name,digest in json.loads((behavior_dir / "provenance.json").read_text()).items():
        assert hashlib.sha256((behavior_dir/name).read_bytes()).hexdigest()==digest
    for name,total in [("train_episodes.csv.gz",20480),("eval_episodes.csv.gz",3600),("train_turns.csv.gz",110997)]:
        with gzip.open(behavior_dir/name,"rt",encoding="utf-8") as f:
            assert sum(1 for _ in csv.DictReader(f))==total
    wb=ROOT / "artifacts/analysis/wandb"
    w=json.loads((wb / "summary.json").read_text())
    assert len(w["runs"])==4 and all(r["state"]=="finished" for r in w["runs"])
    assert [r["step"] for r in w["steps"]]==list(range(1,81))
    assert [r["update"] for r in w["updates"]]==list(range(1,161))
    assert w["reconciliation"]=={"compared_numeric_values":3118,"max_abs_error":0.0}
    for run in w["runs"]:
        assert hashlib.sha256((wb / (run["id"]+".json")).read_bytes()).hexdigest()==run["sha256"]
    assert not (wb / "m1fqyxwx.json").exists()
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Checked {len(docs)} documents, {len(pages)} pages, {sum(len(p.links) for p in pages.values())} links, {sum(p.translations for p in pages.values())} bilingual fields, numerical coverage, SVGs and hashes.")


if __name__ == "__main__":
    main()
