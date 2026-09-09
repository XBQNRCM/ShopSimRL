#!/usr/bin/env python3
"""Render standalone SVG/PNG research figures from the public analysis JSON."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/analysis/figures"
D = json.loads((ROOT / "artifacts/analysis/analysis.json").read_text(encoding="utf-8"))
BLUE, PURPLE, INK, GREY, TEAL = "#245ce5", "#8246c6", "#152238", "#6a778b", "#007f79"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.labelcolor": INK,
                     "text.color": INK, "xtick.color": GREY, "ytick.color": GREY,
                     "axes.edgecolor": "#d6dfed", "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "axes.titlelocation": "left", "svg.fonttype": "none",
                     "savefig.facecolor": "white", "figure.facecolor": "white"})
OUT.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight", metadata={"Date": None})
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", dpi=180)
    plt.close(fig)


def grid(ax):
    ax.grid(axis="y", color="#e8edf5", linewidth=.7)
    ax.set_axisbelow(True)


def rollmean(y, n=5):
    return [np.mean(y[max(0, i-n+1):i+1]) for i in range(len(y))]


steps = D["training_steps"]
x = [r["step"] for r in steps]
fig, axs = plt.subplots(2, 2, figsize=(12, 7.3), layout="constrained")
for ax, key, title, color in [
    (axs[0,0], "reward_mean", "A   Generated trajectory reward", BLUE),
    (axs[0,1], "skill_free_group_rate", "B   Skill-free sampling", PURPLE),
    (axs[1,0], "all_wrong_group_rate", "C   All-wrong groups", TEAL),
    (axs[1,1], "zero_std_group_drop_rate", "D   Zero-variance groups dropped", GREY)]:
    y = [r.get("rollout/shopsim/"+key) for r in steps]
    ax.plot(x, y, color=color, alpha=.22, lw=1)
    ax.plot(x, rollmean(y), color=color, lw=2, label="5-step trailing mean")
    for boundary in [20.5,40.5,60.5]: ax.axvline(boundary, color="#d6dfed", ls="--", lw=.8)
    ax.set(title=title, xlabel="Rollout step", xlim=(1,80), ylim=(0,1))
    if key != "reward_mean": ax.yaxis.set_major_formatter(PercentFormatter(1))
    if key == "skill_free_group_rate": ax.axhline(.2,color=INK,ls=":",lw=1,label="Fixed q = 0.20")
    grid(ax)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
save(fig,"training")

fig, axs = plt.subplots(1,2,figsize=(11.5,4.5),layout="constrained")
rs = D["rounds"]
for key, label, color in [("bare","Bare validation",BLUE),("masked","Random-mask gate",PURPLE)]:
    axs[0].plot([r["step"] for r in rs],[r[key]["reward"] for r in rs],"o-",color=color,lw=2,label=label)
axs[0].set(title="A   Fixed validation set · 400 tasks",xlabel="Rollout step",ylabel="Strict reward",ylim=(.7,.95))
axs[0].legend(frameon=False,fontsize=10);grid(axs[0])
for metric, color, offset in [("reward",BLUE,-.08),("success",PURPLE,.08)]:
    vals=[D["comparisons"][k][metric] for k in ["initial_skill","final_skill","training_free"]]
    yy=np.arange(3)+offset
    axs[1].errorbar([v["difference"]*100 for v in vals],yy,
                    xerr=[[100*(v["difference"]-v["ci95"][0]) for v in vals],
                          [100*(v["ci95"][1]-v["difference"]) for v in vals]],
                    fmt="o",capsize=4,color=color,label="Reward ×100" if metric=="reward" else "Success (pp)")
axs[1].set_yticks(range(3),["Initial skill\nBase + S0 − Base", "Final skill\nIter80 + ST − Iter80", "Training, skill-free\nIter80 − Base"])
axs[1].axvline(0,color=GREY,lw=1,ls="--")
axs[1].set(title="B   Paired test differences · 95% CI",xlabel="Difference",xlim=(-8,42))
axs[1].legend(frameon=False,fontsize=9,loc="lower right")
save(fig,"validation_and_effects")

fig, ax = plt.subplots(figsize=(9.8,4.7),layout="constrained")
arms=list(D["test"]); labels=["Base\nNo skill","Base\nInitial skill S0","Iter80\nNo skill","Iter80\nFinal skill ST"]
pos=np.arange(4)
for j,(metric,color,label) in enumerate([("reward_mean",BLUE,"Strict reward ×100"),("success_rate",PURPLE,"Success rate (%)")]):
    vals=[D["test"][a]["primary"][metric]*100 for a in arms]
    bars=ax.bar(pos+(j-.5)*.34,vals,width=.31,color=color,label=label,zorder=3)
    ax.bar_label(bars,fmt="%.2f",padding=4,fontsize=10)
ax.set_xticks(pos,labels);ax.set(ylim=(0,105),ylabel="Score / rate",title="Test performance · 400 shared tasks, one rollout per condition")
ax.legend(frameon=False,loc="upper left",ncol=2,fontsize=10);grid(ax)
save(fig,"test_results")

fig, axs=plt.subplots(1,2,figsize=(11.5,4.5),layout="constrained")
axs[0].plot([b["step"] for b in D["skill_banks"]],[len(b["skills"]) for b in D["skill_banks"]],"o-",color=PURPLE,lw=2)
for b in D["skill_banks"]: axs[0].annotate(str(len(b["skills"])),(b["step"],len(b["skills"])),xytext=(0,9),textcoords="offset points",ha="center")
axs[0].set(title="A   Active skill size",xlabel="Rollout step",ylabel="Chunks",ylim=(0,12),xticks=[0,20,40,60,80]);grid(axs[0])
xx=np.arange(4)
for shift,key,label,color in [(-.23,"all_wrong_groups","All-wrong",GREY),(0,"analyzed_cards","Failed full-skill retry",BLUE),(.23,"eligible_cards","Eligible analysis cards",PURPLE)]:
    axs[1].bar(xx+shift,[r[key] for r in rs],.21,label=label,color=color)
axs[1].set(title="B   Failure analysis funnel",xticks=xx,xticklabels=["R0","R1","R2","R3"],ylabel="Groups / cards")
axs[1].legend(frameon=False,fontsize=9);grid(axs[1])
save(fig,"skill_evolution")

# Preserve logical IDs across rewrites. A blank is absent from that gate;
# the value is the winning version, even if non-positive and retired.
ids=list(dict.fromkeys(c["logical_chunk_id"] for c in D["coefficients"]))
mat=np.full((len(ids),4),np.nan);selected=np.zeros((len(ids),4),dtype=bool)
for c in D["coefficients"]:
    if c["status"]=="rewrite_loser": continue
    row=ids.index(c["logical_chunk_id"]);mat[row,c["round"]]=c["coefficient"]
    selected[row,c["round"]]=c["status"]=="selected"
fig,ax=plt.subplots(figsize=(9,max(5,len(ids)*.29)),layout="constrained")
lim=float(np.nanmax(np.abs(mat)))
cmap=plt.get_cmap("RdBu").copy();cmap.set_bad("#f0f3f8")
im=ax.imshow(np.ma.masked_invalid(mat),cmap=cmap,norm=TwoSlopeNorm(vmin=-lim,vcenter=0,vmax=lim),aspect="auto")
ax.set_yticks(range(len(ids)),[i.replace("chunk-","c-").replace("online-","n-") for i in ids],fontsize=9)
ax.set_xticks(range(4),["Step 20","Step 40","Step 60","Step 80"])
for row,col in zip(*np.where(selected)):ax.text(col,row,"●",ha="center",va="center",color=INK,fontsize=7)
ax.set(title="Chunk contribution across gates · dot = retained",ylabel="Logical chunk ID (winning version)")
fig.colorbar(im,ax=ax,shrink=.7,label="Paired-delta OLS coefficient")
save(fig,"chunk_contributions")

fig,ax=plt.subplots(figsize=(10,4.6),layout="constrained")
components=["r_type","r_att","r_option","r_price","r_strict","r_success"]
for arm,label,color in [("base_free","Base, no skill",GREY),("iter80_free","Iter80, no skill",BLUE),("iter80_st","Iter80 + ST",PURPLE)]:
    ax.plot(components,[D["test"][arm]["reward_components"][m] for m in components],"o-",label=label,color=color,lw=2)
ax.set(ylim=(.45,1.02),ylabel="Mean reward component",title="Where performance changed · test 400")
ax.legend(frameon=False,loc="lower left");grid(ax)
save(fig,"reward_components")
print(f"Rendered six figures as SVG and PNG in {OUT}")
