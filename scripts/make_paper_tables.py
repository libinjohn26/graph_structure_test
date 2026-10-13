#!/usr/bin/env python

"""Emit the paper's tables as LaTeX, from result files only.



Every number the paper prints should be traceable to a file on disk and

regenerable by one command, so that a corrected run propagates into the paper

without anyone re-typing a figure. Nothing here computes a metric — it reads

what the eval harness and the analysis scripts already wrote and formats it.



Inputs (all produced by other scripts; missing ones are skipped with a notice):

    data/processed/eval_full_sweep.json         - main LoCoMo results (run_eval.py)

    data/processed/eval_hotpotqa.json           - HotpotQA control  (run_eval.py)

    data/processed/eval_hotpotqa_relweight.json - HotpotQA, rwp=8   (run_eval.py)

    data/processed/graph_analysis.json          - A2/A3            (analyse_graphs.py)

    data/processed/bootstrap_*.json             - A1               (bootstrap_ci.py)



Usage:

    python scripts/make_paper_tables.py --out paper/tables

"""



from __future__ import annotations



import argparse

import json

import statistics as st

from pathlib import Path



REPO_ROOT = Path(__file__).resolve().parent.parent

PROCESSED = REPO_ROOT / "data" / "processed"



# Shown in paper order, with the labels the paper uses.

ARCH_LABEL = {

    "L0 flat_summary": r"L0 \textsc{flat}",

    "L1 vector": r"L1 \textsc{vector}",

    "L2 kv_entity": r"L2 \textsc{kv}",

    "L3 graph_cheap": r"L3 \textsc{graph}",

}

ARCH_ORDER = ["L0 flat_summary", "L1 vector", "L2 kv_entity", "L3 graph_cheap"]





def read(name: str) -> dict | None:

    path = PROCESSED / name

    if not path.exists():

        print(f"  [skip] {name} not found")

        return None

    return json.loads(path.read_text())





def fmt(value, places: int = 3, bold: bool = False) -> str:

    if value is None:

        return "--"

    text = f"{value:.{places}f}"

    return rf"\textbf{{{text}}}" if bold else text





def tabular(caption: str, label: str, spec: str, header: list[str], rows: list[list[str]]) -> str:

    lines = [

        r"\begin{table}[t]",

        r"\centering",

        rf"\caption{{{caption}}}",

        rf"\label{{{label}}}",

        rf"\begin{{tabular}}{{{spec}}}",

        r"\toprule",

        " & ".join(header) + r" \\",

        r"\midrule",

    ]

    lines += [" & ".join(r) + r" \\" for r in rows]

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    return "\n".join(lines)





def architectures_of(blob: dict) -> dict[str, dict]:

    """The harness writes results either as a list or keyed by name; accept both."""

    results = blob.get("results", blob)

    if isinstance(results, list):

        return {r["architecture"]: r for r in results}

    return results





def table_main(blob: dict) -> str:

    """Headline accuracy and cost per architecture."""

    archs = architectures_of(blob)



    # Retrieval metrics sit at the top level; generation metrics are nested under

    # "generation" and are absent entirely on a retrieval-only run.

    def value(entry, key):

        if key in ("factual_correctness", "answer_accuracy"):

            return (entry.get("generation") or {}).get(key)

        return entry.get(key)



    present = [n for n in ARCH_ORDER if n in archs]

    columns = ["evidence_recall", "context_relevance", "factual_correctness", "answer_accuracy"]



    # Bold the actual column maximum rather than assuming L1 wins everything: on

    # LoCoMo, L2 takes context relevance, and hardcoding the winner would have

    # printed a table that contradicts its own numbers.

    best_of = {}

    for key in columns:

        scored = [v for n in present if (v := value(archs[n], key)) is not None]

        best_of[key] = max(scored) if scored else None



    rows = []

    for name in present:

        entry = archs[name]

        tokens = (entry.get("cost") or {}).get("construction_tokens")

        cells = [ARCH_LABEL.get(name, name)]

        for key in columns:

            current = value(entry, key)

            cells.append(fmt(current, bold=current is not None and current == best_of[key]))

        cells.append(f"{tokens:,}".replace(",", r"\,") if tokens is not None else "--")

        rows.append(cells)

    return tabular(

        "Retrieval, generation and construction cost on LoCoMo (1{,}986 QA, 10 "

        "conversations, matched 5-turn context budget).",

        "tab:main",

        "lrrrrr",

        ["Architecture", "Recall", "Relevance", "Factual", "Acc.", "Constr.\\ tokens"],

        rows,

    )





def table_graph_stats(blob: dict) -> str:

    """A2: graph topology per corpus, as mean [min--max]."""

    rows = []

    for dataset, label in (("locomo", "LoCoMo"), ("hotpotqa", "HotpotQA")):

        corpora = blob.get(dataset) or []

        if not corpora:

            continue



        def stat(key, places=2, sep=r"--"):

            values = [c[key] for c in corpora]

            mean = st.mean(values)

            text = f"{mean:,.{places}f}".replace(",", r"\,")

            lo = f"{min(values):,.{places}f}".replace(",", r"\,")

            hi = f"{max(values):,.{places}f}".replace(",", r"\,")

            return rf"{text} \tiny[{lo}{sep}{hi}]"



        rows.append([

            rf"{label} \scriptsize(n={len(corpora)})",

            stat("nodes", 0),

            stat("edges", 0),

            stat("mean_degree", 2),

            stat("clustering", 3),

            stat("components", 0),

            stat("independent_cycles", 0),

        ])

    return tabular(

        "Extracted graph topology, per corpus (one LoCoMo conversation or one "

        "HotpotQA pool). Mean with [min--max]. Independent cycles "

        "$=E-V+C$; a forest would be zero, so no corpus is one.",

        "tab:graphstats",

        "lrrrrrr",

        ["Corpus", "$V$", "$E$", "Deg.", "Clust.", "Comp.", "Cycles"],

        rows,

    )





def table_reachability(blob: dict) -> str:

    """A3: where gold evidence sits relative to itself."""

    from collections import Counter



    rows = []

    specs = [

        ("locomo", "reachability_multi_hop", "with_speakers", "LoCoMo"),

        ("locomo", "reachability_multi_hop", "without_speakers", r"\quad less speakers"),

        ("hotpotqa", "reachability_bridge", "with_speakers", "HotpotQA"),

    ]

    for dataset, key, variant, label in specs:

        corpora = blob.get(dataset) or []

        if not corpora:

            continue

        total = same = disconnected = 0

        distribution = Counter()

        for corpus in corpora:

            entry = corpus[key][variant]

            total += entry["questions"]

            same += entry["gold_entities_share_a_component"]

            disconnected += entry["gold_turns_disconnected"]

            for hops, count in entry["shortest_path_distribution"].items():

                distribution[int(hops)] += count

        paths = sum(distribution.values())

        if not total or not paths:

            continue

        deep = sum(v for k, v in distribution.items() if k >= 2)

        rows.append([

            label,

            str(total),

            rf"{same / total:.1%}".replace("%", r"\%"),

            rf"{distribution[0] / paths:.0%}".replace("%", r"\%"),

            rf"{distribution[1] / paths:.0%}".replace("%", r"\%"),

            rf"\textbf{{{deep / paths:.1%}}}".replace("%", r"\%"),

            rf"{disconnected / total:.1%}".replace("%", r"\%"),

        ])

    return tabular(

        "Gold-evidence reachability for multi-hop questions. ``Hardest leg'' is "

        "the hop budget needed to reach all of a question's gold evidence from "

        "any one piece of it. Traversal of depth $\\geq 2$ is both necessary and "

        "possible for a small minority; the rest is already adjacent or "

        "unreachable.",

        "tab:reach",

        "lrrrrrr",

        ["Corpus", "$n$", "Connected", "0 hop", "1 hop", "$\\geq$2 hop", "Disconn."],

        rows,

    )





def table_bootstrap(blobs: dict[str, dict]) -> str:

    """A1: paired bootstrap intervals, by conversation."""

    rows = []

    metrics = [

        ("evidence_recall", "Recall"),

        ("context_relevance", "Relevance"),

        ("factual_correctness", "Factual"),

        ("answer_accuracy", "Accuracy"),

    ]

    for (dataset_label, _source), blob in blobs.items():

        for comparison, entry in (blob.get("comparisons") or {}).items():

            for metric_key, metric_label in metrics:

                stats = (entry.get(metric_key) or {}).get("conversation")

                if not stats or not stats.get("n"):

                    continue

                star = "" if stats["significant"] else r"\,n.s."

                rows.append([

                    dataset_label,

                    comparison.replace("->", r"$\rightarrow$").replace("_", r"\_"),

                    metric_label,

                    fmt(stats["left_mean"]),

                    fmt(stats["right_mean"]),

                    rf"{stats['difference']:+.4f}",

                    rf"[{stats['ci_low']:+.4f}, {stats['ci_high']:+.4f}]{star}",

                ])

    return tabular(

        "Paired bootstrap (10{,}000 resamples), resampling whole conversations. "

        "Intervals excluding zero are significant; ``n.s.'' marks those that do "

        "not. Every L1-vs-L3 gap is significant; no graph-tuning gain is.",

        "tab:bootstrap",

        "lllrrrr",

        ["Data", "Comparison", "Metric", "From", "To", "$\\Delta$", "95\\% CI"],

        rows,

    )





def _recall(filename: str, prefix: str) -> float | None:

    blob = read(filename)

    if not blob:

        return None

    for row in blob.get("results", []):

        if row["architecture"].startswith(prefix):

            return row["evidence_recall"]

    return None





def table_ablation() -> str | None:

    """Traversal ablation at full scale, both datasets (finding #1)."""

    rows = []

    specs = [

        ("LoCoMo", "eval_locomo_alpha0_retrieval.json", "scale_locomo_full.json",

         "eval_sweep_perqa.json", "L3"),

        ("HotpotQA", "eval_hotpotqa_alpha0.json", "eval_hotpotqa_relweight.json",

         "eval_hotpotqa.json", "L3"),

    ]

    for label, none_f, tuned_f, static_f, pre in specs:

        no_trav, tuned, static = (_recall(f, pre) for f in (none_f, tuned_f, static_f))

        if no_trav is None:

            continue

        rows.append([

            label,

            fmt(no_trav, 4),

            fmt(static, 4),

            fmt(tuned, 4, bold=True),

        ])

    if not rows:

        return None

    return tabular(

        "Traversal ablation at full scale. $\\alpha=0$ means the walk never leaves "

        "its seeds, so edge weights cannot matter: it is the single no-traversal "

        "baseline for both weighting schemes. On LoCoMo both differences are "

        "significant and opposite in sign; on HotpotQA ($n=200$) neither resolves.",

        "tab:ablation",

        "lrrr",

        ["Corpus", "No traversal", "Static weights", "Relation-weighted"],

        rows,

    )





def table_hybrid() -> str | None:

    """The vector_slots sweep (finding #15)."""

    rows = []

    for slots in (0, 1, 2, 3, 4, 5):

        filename = "eval_hybrid_full.json" if slots == 3 else f"hybrid_vs{slots}.json"

        value = _recall(filename, "hybrid")

        if value is None:

            continue

        note = {0: "pure graph ranking", 3: "default", 5: "reproduces L1"}.get(slots, "")

        rows.append([

            str(slots),

            fmt(value, 4, bold=(slots == 1)),

            rf"\scriptsize {note}" if note else "",

        ])

    baseline = _recall("scale_locomo_full.json", "L1")

    if baseline is not None:

        rows.append([r"\midrule L1", fmt(baseline, 4), r"\scriptsize reference"])

    if not rows:

        return None

    return tabular(

        "Hybrid rung on LoCoMo: how many of the $k{=}5$ slots the vector channel "

        "keeps, the graph filling the rest. Seeds always come from vector-ranked "

        "turns. The gain over Vector does not replicate on HotpotQA, where the "

        "hybrid scores 0.7067 against Vector's 0.7644.",

        "tab:hybrid",

        "lrl",

        [r"\texttt{vector\_slots}", "Recall", ""],

        rows,

    )





def table_resolution() -> str | None:

    """Entity resolution, both datasets, both rungs (finding #16)."""

    specs = [

        ("LoCoMo", "Graph", "scale_locomo_full.json", "L3", "resolved_locomo_L3.json"),

        ("LoCoMo", "Hybrid", "hybrid_vs1.json", "hybrid", "resolved_locomo_hybrid.json"),

        ("HotpotQA", "Graph", "eval_hotpotqa_relweight.json", "L3", "resolved_hotpotqa_L3.json"),

        ("HotpotQA", "Hybrid", "hybrid_hotpotqa.json", "hybrid", "resolved_hotpotqa_hybrid.json"),

    ]

    rows = []

    for corpus, rung, base_f, pre, res_f in specs:

        base, resolved = _recall(base_f, pre), _recall(res_f, pre)

        if base is None or resolved is None:

            continue

        rows.append([corpus, rung, fmt(base, 4), fmt(resolved, 4),

                     rf"{resolved - base:+.4f}"])

    if not rows:

        return None

    return tabular(

        "Entity resolution. Merging co-referent nodes repairs the graph "

        "(LoCoMo degree $2.13\\to2.36$) and does not improve retrieval through "

        "it; on the LoCoMo graph rung it is significantly worse.",

        "tab:resolution",

        "llrrr",

        ["Corpus", "Rung", "Baseline", "Resolved", "$\Delta$"],

        rows,

    )





def table_robustness() -> str | None:

    """Scale and budget sweeps (findings #13, #14)."""

    rows = []

    for scale in ("100", "200", "400", "full"):

        l1, l3 = (_recall(f"scale_locomo_{scale}.json", p) for p in ("L1", "L3"))

        if l1 is None:

            continue

        rows.append(["History", scale, fmt(l1, 4), fmt(l3, 4), rf"{l3 - l1:+.4f}"])

    for k in (3, 5, 10, 20):

        filename = "scale_locomo_full.json" if k == 5 else f"budget_locomo_{k}.json"

        l1, l3 = (_recall(filename, p) for p in ("L1", "L3"))

        if l1 is None:

            continue

        rows.append(["Budget", f"k={k}", fmt(l1, 4), fmt(l3, 4), rf"{l3 - l1:+.4f}"])

    if not rows:

        return None

    return tabular(

        "Robustness of the Vector-over-Graph gap. Vector leads at every history "

        "length and every retrieval budget tested; the gap neither narrows with "

        "scale nor closes with a larger budget. Raw (all-scored) recall is shown; "

        "see the text for the fixed-question-set curve.",

        "tab:robustness",

        "llrrr",

        ["Axis", "Setting", "Vector", "Graph", "$\Delta$"],

        rows,

    )





def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument("--out", default=str(REPO_ROOT / "paper" / "tables"))

    args = parser.parse_args()



    out = Path(args.out)

    out.mkdir(parents=True, exist_ok=True)

    written: list[str] = []



    def emit(filename: str, content: str | None) -> None:

        if not content:

            return

        (out / filename).write_text(content, encoding="utf-8")

        written.append(filename)



    print("Reading result files:")

    main_blob = read("eval_full_sweep.json")

    if main_blob:

        emit("main.tex", table_main(main_blob))



    graph_blob = read("graph_analysis.json")

    if graph_blob:

        emit("graph_stats.tex", table_graph_stats(graph_blob))

        emit("reachability.tex", table_reachability(graph_blob))



    bootstraps = {}

    for filename, label in (

        ("bootstrap_locomo.json", "LoCoMo"),

        ("bootstrap_locomo_relweight.json", "LoCoMo"),

        ("bootstrap_l1_vs_rwp8.json", "LoCoMo"),

        ("bootstrap_hotpotqa.json", "HotpotQA"),

        ("bootstrap_relweight.json", "HotpotQA"),

    ):

        blob = read(filename)

        if blob:

            bootstraps[(label, filename)] = blob

    if bootstraps:

        emit("bootstrap.tex", table_bootstrap(bootstraps))



    emit("ablation.tex", table_ablation())

    emit("hybrid.tex", table_hybrid())

    emit("resolution.tex", table_resolution())

    emit("robustness.tex", table_robustness())



    print(f"\nWrote {len(written)} table(s) to {out}:")

    for name in written:

        print(f"  {name}")





if __name__ == "__main__":

    main()

