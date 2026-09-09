#!/usr/bin/env python3
"""Extract and analyze shopping behavior without making model/environment calls.

Default: rebuild from public numerical episode extracts.
--refresh-local: read full local training/evaluation traces; tokenizer is required.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/analysis/behavior"
EVAL_RUNS = {"val_base": "qwen35-4b-val-base-free",
             **{f"val_{20*(i+1)}": f"round-{i:03}-bare-val" for i in range(4)},
             "test_base_free": "qwen35-4b-test-base-free", "test_base_s0": "qwen35-4b-test-base-s0",
             "test_iter80_free": "qwen35-4b-test-iter80-free", "test_iter80_st": "qwen35-4b-test-iter80-st"}
METRICS = ["model_steps", "actions", "valid_actions", "tokens_per_turn", "generated_tokens",
           "first_turn_tokens", "searches", "unique_queries", "product_opens", "unique_products",
           "product_revisits", "backtracks", "pagination", "option_selections", "option_changes",
           "protocol_errors", "invalid_actions", "reasoning_chars_per_turn", "first_product_rank"]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def write_csv_gz(path, rows):
    with Path(path).open("wb") as file:
        with gzip.GzipFile(filename="", mode="wb", fileobj=file, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)


def extract_episode(trace, *, cohort, group_index=-1, round_index=-1, scored_override=None, tokenizer=None):
    """Count actual state transitions. Protocol repairs are model turns, not actions."""
    final = trace.get("final") or {}
    scored = trace.get("status") == "completed" and final.get("done") is True and isinstance(final.get("reward"), (float,int))
    if scored_override is not None:
        scored = scored and scored_override
    reward = float(final.get("reward", 0)) if scored else None
    success = int(final.get("reward_detail", {}).get("r_success", 0)) if scored else None
    assignment = trace.get("provenance", {}).get("skills", {}).get("assignment", {})
    steps = trace.get("steps", [])
    r = {"cohort": cohort, "round": round_index, "rollout_step": group_index//32+1 if group_index>=0 else -1,
         "group_index": group_index, "episode_id": trace["episode_id"], "task_id": trace["job"]["task_id"],
         "sample_id": trace["job"]["sample_id"], "mode": assignment.get("mode", "evaluation"),
         "scored": int(scored), "success": success, "reward": reward,
         "outcome": ("success" if success else "failure") if scored else "technical",
         "termination": final.get("termination_reason", "technical"), "model_steps": len(steps),
         "actions": 0, "valid_actions": 0, "searches": 0, "unique_queries": 0, "product_opens": 0,
         "unique_products": 0, "product_revisits": 0, "backtracks": 0, "pagination": 0,
         "option_selections": 0, "option_changes": 0, "protocol_errors": 0, "invalid_actions": 0,
         "generated_tokens": 0, "token_turns": 0, "tokens_per_turn": None, "first_turn_tokens": None,
         "raw_chars": 0, "raw_turns": 0, "reasoning_chars": 0, "reasoning_chars_per_turn": None,
         "first_product_rank": None, "exposed_products": 0, "query_chars": 0,
         "token_basis": "reencoded_raw_output" if tokenizer else "api_completion_tokens"}
    for phase in ["search_home", "search_results", "product_detail", "other", "protocol_repair"]:
        r[phase+"_turns"] = 0
        r[phase+"_tokens"] = 0
    before = (trace.get("reset") or {}).get("observation_state", {})
    queries, seen_products, exposed = set(), set(), set()
    turn_rows, action_path = [], []
    for step in steps:
        model = step.get("model") or {}
        env = step.get("environment") or {}
        after = env.get("observation_state") or before
        action = step.get("action")
        feedback = env.get("action_feedback") or {}
        valid = bool(action and env and feedback.get("valid") is True)
        protocol = step.get("protocol_error") or model.get("protocol_error")
        phase = before.get("page_type", "other")
        if phase not in ["search_home", "search_results", "product_detail"]:
            phase = "other"
        if protocol:
            r["protocol_errors"] += 1
            phase = "protocol_repair"
        raw = model.get("raw_output")
        tokens = None
        if tokenizer and isinstance(raw, str):
            tokens = len(tokenizer.encode(raw, add_special_tokens=False).ids)
        elif not tokenizer:
            tokens = (model.get("usage") or {}).get("completion_tokens")
        if tokens is not None:
            tokens = int(tokens)
            r["token_turns"] += 1
            r["generated_tokens"] += tokens
            r[phase+"_tokens"] += tokens
            if step["step_index"] == 1:
                r["first_turn_tokens"] = tokens
        r[phase+"_turns"] += 1
        if isinstance(raw, str):
            r["raw_chars"] += len(raw)
            r["raw_turns"] += 1
        reasoning_chars = len(model.get("reasoning") or "")
        r["reasoning_chars"] += reasoning_chars
        for product in before.get("products", []):
            exposed.add(str(product["asin"]))
        label = "repair" if protocol else "no_action"
        if action:
            r["actions"] += 1
            if valid:
                r["valid_actions"] += 1
            elif feedback.get("valid") is False:
                r["invalid_actions"] += 1
            label = "invalid_action" if not valid else "other"
        if valid and action.startswith("search["):
            query = action[7:-1]
            r["searches"] += 1
            r["query_chars"] += len(query)
            queries.add(re.sub(r"\s+", " ", query).strip().casefold())
            label = "search"
        elif valid and action.startswith("click["):
            target = action[6:-1]
            products = {str(p["asin"]): p for p in before.get("products", [])}
            if before.get("page_type") == "search_results" and target in products and after.get("page_type") == "product_detail":
                r["product_opens"] += 1
                if target in seen_products:
                    r["product_revisits"] += 1
                seen_products.add(target)
                if r["first_product_rank"] is None:
                    r["first_product_rank"] = products[target].get("rank")
                label = "open_product"
            elif target == "buy now":
                label = "purchase"
            elif target == "back to search":
                r["backtracks"] += 1
                label = "backtrack"
            elif target in ["< prev", "next >"]:
                key = "pagination" if before.get("page_type") == "search_results" else "backtracks"
                r[key] += 1
                label = "paginate" if key == "pagination" else "backtrack"
            else:
                options = before.get("available_options") or {}
                matches = [(axis,value) for axis,values in options.items() for value in values if f"{axis}={value}" == target]
                if matches:
                    axis,value = matches[0]
                    r["option_selections"] += 1
                    previous = (before.get("selected_options") or {}).get(axis)
                    if previous is not None and previous != value:
                        r["option_changes"] += 1
                    label = "select_option"
        action_path.append(label)
        turn_rows.append({"cohort": cohort, "round": round_index, "rollout_step": r["rollout_step"],
                          "group_index": group_index, "episode_id": trace["episode_id"], "task_id": r["task_id"],
                          "outcome": r["outcome"], "turn": step["step_index"], "phase": phase, "action_type": label,
                          "tokens": tokens, "raw_chars": len(raw) if isinstance(raw,str) else None,
                          "reasoning_chars": reasoning_chars})
        if env:
            before = after
    r["unique_queries"], r["unique_products"], r["exposed_products"] = len(queries), len(seen_products), len(exposed)
    if r["token_turns"] == r["model_steps"] and r["token_turns"]:
        r["tokens_per_turn"] = r["generated_tokens"]/r["token_turns"]
    if r["model_steps"]:
        r["reasoning_chars_per_turn"] = r["reasoning_chars"]/r["model_steps"]
    return r, turn_rows, action_path


def refresh(tokenizer_path):
    from tokenizers import Tokenizer, __version__ as tokenizers_version
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    jobs = []
    for i in range(4):
        for group in sorted((ROOT / f"runs/slime-training/round-{i:03}").glob("group-*")):
            triage = read(group / "triage.json")
            scored = {int(x["sample_index"]): x["scored"] for x in triage["training_rollouts"]}
            for path in sorted(group.glob("rollout-*.json")):
                jobs.append((path, i, int(group.name.split("-")[1]), scored))
    if len(jobs) != 20480:
        raise ValueError(f"Expected 20480 training episode files; found {len(jobs)}")

    def worker(job):
        path,i,group,scored=job
        raw=path.read_bytes(); trace=json.loads(raw)
        result = extract_episode(trace, cohort="train", group_index=group, round_index=i,
                                 scored_override=scored[trace["job"]["sample_id"]], tokenizer=tokenizer)
        return result, hashlib.sha256(raw).hexdigest()

    episodes, turns, digests = [], [], []
    with ThreadPoolExecutor(max_workers=8) as executor:
        for n, ((row, turn_rows, _), digest) in enumerate(executor.map(worker,jobs),1):
            episodes.append(row);turns.extend(turn_rows);digests.append(digest)
            if n%2560 == 0:
                print(f"Training traces: {n}/20480", flush=True)
    write_csv_gz(OUT / "train_episodes.csv.gz", episodes)
    write_csv_gz(OUT / "train_turns.csv.gz", turns)
    eval_rows, eval_turns, case_pool, eval_sources = [], [], {}, {}
    for cohort,name in EVAL_RUNS.items():
        path=ROOT / "runs" / name / "traces.jsonl"
        records={}
        for line in path.open(encoding="utf-8"):
            d=json.loads(line)
            if d.get("status")!="completed" or not (d.get("final") or {}).get("done"):
                continue
            key=(d["job"]["split"],d["job"]["task_id"],d["job"]["sample_id"])
            if key in records: raise ValueError(f"Duplicate completed task: {cohort}/{key}")
            records[key]=d
        if len(records)!=400: raise ValueError(f"Incomplete fixed evaluation: {cohort}")
        for key,trace in sorted(records.items()):
            row,turn_rows,path_labels=extract_episode(trace,cohort=cohort)
            eval_rows.append(row);eval_turns.extend(turn_rows)
            if cohort in ["val_base","val_80"]:
                case_pool.setdefault(str(row["task_id"]),{})[cohort]={
                    "metrics":row,"actions":path_labels}
        eval_sources[cohort]={"source":path.relative_to(ROOT).as_posix(),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
        print(f"Evaluation: {cohort} (400 tasks)",flush=True)
    write_csv_gz(OUT / "eval_episodes.csv.gz",eval_rows)
    write_csv_gz(OUT / "eval_turns.csv.gz",eval_turns)
    eligible=[(key,pair) for key,pair in case_pool.items() if len(pair)==2]
    both_success=[x for x in eligible if all(v["metrics"]["success"] for v in x[1].values())]
    improved=[x for x in eligible if not x[1]["val_base"]["metrics"]["success"] and x[1]["val_80"]["metrics"]["success"]]
    regressed=[x for x in eligible if x[1]["val_base"]["metrics"]["success"] and not x[1]["val_80"]["metrics"]["success"]]
    cases=[]
    for kind,pool,sort_key in [
        ("largest_step_reduction_both_success",both_success,lambda x:x[1]["val_base"]["metrics"]["model_steps"]-x[1]["val_80"]["metrics"]["model_steps"]),
        ("failure_to_success_largest_token_reduction",improved,lambda x:x[1]["val_base"]["metrics"]["generated_tokens"]-x[1]["val_80"]["metrics"]["generated_tokens"]),
        ("success_to_failure_longest_final",regressed,lambda x:x[1]["val_80"]["metrics"]["model_steps"])]:
        if pool:
            task,pair=max(pool,key=lambda x:(sort_key(x),-int(x[0])))
            cases.append({"selection_rule":kind,"task_id":int(task),"conditions":pair})
    write(OUT / "cases.json",cases)
    write(OUT / "extraction.json",{
        "schema_version":"shopsimrl-behavior-extraction-v1","train_files":len(jobs),"train_turns":len(turns),
        "train_manifest_sha256":hashlib.sha256("\n".join(f"{job[0].relative_to(ROOT).as_posix()} {digest}" for job,digest in zip(jobs,digests)).encode()).hexdigest(),
        "tokenizer":{"source":"https://huggingface.co/Qwen/Qwen3.5-4B/resolve/main/tokenizer.json",
                     "sha256":hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),"tokenizers_version":tokenizers_version,
                     "add_special_tokens":False,"basis":"re-encoding saved raw_output, NOT original sampled token IDs",
                     "matches_training_tokenizer_hash":"unverified; historical tokenizer hash not recorded"},
        "eval_sources":eval_sources})


TEXT_FIELDS={"cohort","episode_id","outcome","mode","termination","token_basis","phase","action_type"}
def load_csv(name):
    with gzip.open(OUT / name,"rt",encoding="utf-8",newline="") as file:
        return [{k:(v if k in TEXT_FIELDS else float(v) if v else None) for k,v in row.items()} for row in csv.DictReader(file)]


def describe(rows, metrics=METRICS):
    result={"n":len(rows),"tasks":len({r["task_id"] for r in rows}),"metrics":{}}
    scored=[r for r in rows if r["scored"]]
    result["success_rate"]=float(np.mean([r["success"] for r in scored])) if scored else None
    result["reward_mean"]=float(np.mean([r["reward"] for r in scored])) if scored else None
    for metric in metrics:
        values=[r[metric] for r in rows if r[metric] is not None]
        result["metrics"][metric]={"n":len(values),"mean":float(np.mean(values)) if values else None,
                                   "median":float(np.median(values)) if values else None,
                                   "p90":float(np.quantile(values,.9)) if values else None}
    complete=[r for r in rows if r["token_turns"]==r["model_steps"] and r["model_steps"]]
    result["pooled_tokens_per_turn"]=(sum(r["generated_tokens"] for r in complete)/sum(r["model_steps"] for r in complete)) if complete else None
    result["token_complete_episodes"]=len(complete)
    result["rates"]={k:sum(bool(r[k]) for r in rows)/len(rows) if rows else None
                     for k in ["protocol_errors","product_revisits","option_changes","backtracks"]}
    result["terminations"]=dict(Counter(r["termination"] for r in rows))
    return result


def paired_tasks(a,b,metrics):
    aa=defaultdict(list);bb=defaultdict(list)
    for r in a: aa[r["task_id"]].append(r)
    for r in b: bb[r["task_id"]].append(r)
    keys=sorted(aa.keys() & bb.keys())
    result={"paired_tasks":len(keys),"metrics":{}}
    for metric in metrics:
        values=[]
        for key in keys:
            av=[r[metric] for r in aa[key] if r[metric] is not None]
            bv=[r[metric] for r in bb[key] if r[metric] is not None]
            if av and bv: values.append((float(np.mean(av)),float(np.mean(bv))))
        if not values:continue
        va,vb=np.asarray(values).T;delta=vb-va
        seed=int.from_bytes(hashlib.sha256(f"20260909:{metric}".encode()).digest()[:4],"big")
        rng=np.random.default_rng(seed)
        boot=delta[rng.integers(0,len(delta),size=(5000,len(delta)))].mean(axis=1)
        result["metrics"][metric]={"n":len(delta),"a_mean":float(va.mean()),"b_mean":float(vb.mean()),
                                   "difference":float(delta.mean()),"ci95":np.quantile(boot,[.025,.975]).tolist()}
    return result


def analyze():
    training=load_csv("train_episodes.csv.gz");evaluation=load_csv("eval_episodes.csv.gz")
    turns=load_csv("train_turns.csv.gz")
    if len(training)!=20480 or len(evaluation)!=3600: raise ValueError("Unexpected extraction size")
    expected={int(r["group_index"]):r for r in csv.DictReader((ROOT / "artifacts/analysis/training_groups.csv").open(encoding="utf-8"))}
    by_group=defaultdict(list)
    for r in training:by_group[int(r["group_index"])].append(r)
    for group,rows in by_group.items():
        assert len(rows)==8 and len({r["task_id"] for r in rows})==1 and len({r["mode"] for r in rows})==1
        good=[r for r in rows if r["scored"]]
        assert len(good)==int(expected[group]["scored"])
        if good: assert np.isclose(np.mean([r["reward"] for r in good]),float(expected[group]["reward_mean"]))
    rounds=[]
    for i in range(4):
        rows=[r for r in training if r["round"]==i]
        scored=[r for r in rows if r["scored"]]
        rounds.append({"round":i,"all":describe(scored),"success":describe([r for r in scored if r["success"]]),
                       "failure":describe([r for r in scored if not r["success"]]),
                       "technical_n":sum(not r["scored"] for r in rows),
                       "skill_free":describe([r for r in scored if r["mode"]=="skill_free"]),
                       "assisted":describe([r for r in scored if r["mode"]!="skill_free"])})
    step_rows=[]
    for step in range(1,81):
        rows=[r for r in training if r["rollout_step"]==step and r["scored"]]
        step_rows.append({"step":step,**{outcome:describe(rows if outcome=="all" else [r for r in rows if r["outcome"]==outcome]) for outcome in ["all","success","failure"]}})
    eval_summary={c:describe([r for r in evaluation if r["cohort"]==c]) for c in EVAL_RUNS}
    eval_outcomes={c:{o:describe([r for r in evaluation if r["cohort"]==c and r["outcome"]==o])
                      for o in ["success","failure"]} for c in EVAL_RUNS}
    base=[r for r in evaluation if r["cohort"]=="val_base"]; final=[r for r in evaluation if r["cohort"]=="val_80"]
    common_success={r["task_id"] for r in base if r["success"]}&{r["task_id"] for r in final if r["success"]}
    pairs={"fixed_val_all":paired_tasks(base,final,METRICS),
           "fixed_val_both_success":paired_tasks([r for r in base if r["task_id"] in common_success],[r for r in final if r["task_id"] in common_success],METRICS),
           "train_common_tasks_r0_r3":paired_tasks([r for r in training if r["round"]==0 and r["scored"]],[r for r in training if r["round"]==3 and r["scored"]],METRICS)}
    phases=[]
    for i in range(4):
        for phase in ["search_home","search_results","product_detail","protocol_repair"]:
            values=[r["tokens"] for r in turns if r["round"]==i and r["phase"]==phase and r["outcome"]!="technical" and r["tokens"] is not None]
            phases.append({"round":i,"phase":phase,"n":len(values),"mean":float(np.mean(values)) if values else None,
                           "median":float(np.median(values)) if values else None,"p90":float(np.quantile(values,.9)) if values else None})
    distributions=[]
    for i in range(4):
        for outcome in ["success","failure"]:
            rows=[r for r in training if r["round"]==i and r["outcome"]==outcome]
            for metric in ["model_steps","searches","unique_products","protocol_errors"]:
                distributions.append({"round":i,"outcome":outcome,"metric":metric,"counts":dict(sorted(Counter(int(r[metric]) for r in rows).items()))})
    result={"schema_version":"shopsimrl-behavior-analysis-v1","extraction":read(OUT / "extraction.json"),
            "rounds":rounds,"steps":step_rows,"evaluation":eval_summary,"evaluation_outcomes":eval_outcomes,"paired":pairs,"phases":phases,
            "distributions":distributions,"cases":read(OUT / "cases.json"),
            "uncertainty":{"method":"paired task bootstrap, 5000 resamples, numpy.default_rng, metric-hashed seed 20260909",
                           "train_pairing":"average siblings and repeated episodes within each task/round, then pair common tasks",
                           "success_conditioning":"descriptive/post-treatment; residual failure mix changes during learning"}}
    write(OUT / "behavior.json",result)
    write(OUT / "provenance.json",{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.glob("*.csv.gz"))})
    for row in rounds:
        print("Round",row["round"],"success/failure",row["success"]["n"],row["failure"]["n"],
              {m:round(row["all"]["metrics"][m]["mean"],3) for m in ["model_steps","tokens_per_turn","searches","unique_products","protocol_errors"]})
    print("Fixed validation",json.dumps(pairs["fixed_val_all"],ensure_ascii=True))
    print("Wrote",OUT / "behavior.json")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-local",action="store_true")
    parser.add_argument("--tokenizer",type=Path,default=ROOT / "data/analysis-tokenizer/tokenizer.json")
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if args.refresh_local:refresh(args.tokenizer)
    analyze()
