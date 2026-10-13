#!/usr/bin/env python
"""Run the retrieval eval over LoCoMo for the architectures built so far (L0, L1).

Both architectures are given the *same context budget* (`--context-turns`):
L0's recency window and L1's top-k both return that many turns, so a difference
in score is a difference in retrieval strategy rather than in how much context
each was allowed to hand downstream.

Examples:
    # quick smoke run over two conversations
    python scripts/run_eval.py --limit 2

    # full run (L1 embeds ~5.9k turns on CPU — expect several minutes)
    python scripts/run_eval.py

    # one scale point from configs/default.yaml
    python scripts/run_eval.py --max-turns 150
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from gms.architectures.flat import FlatRecencyMemory  # noqa: E402
from gms.architectures.hybrid import HybridMemory  # noqa: E402
from gms.architectures.graph import (  # noqa: E402
    _PAGERANK_ALPHA,
    _RELATION_WEIGHT_POWER,
    GraphMemory,
)
from gms.architectures.kv_entity import KVEntityMemory  # noqa: E402
from gms.architectures.vector import VectorMemory  # noqa: E402
from gms.data.hotpotqa import HOTPOTQA_CATEGORIES, load_hotpotqa  # noqa: E402
from gms.data.locomo import LOCOMO_QA_CATEGORIES, load_locomo  # noqa: E402
from gms.eval.harness import EvalResult, evaluate  # noqa: E402
from gms.utils.env import load_env_file  # noqa: E402

# Raw category labels are dataset-specific (ints for LoCoMo, strings for
# HotpotQA) and deliberately not remapped; one lookup covers both for display.
CATEGORY_NAMES = {**LOCOMO_QA_CATEGORIES, **HOTPOTQA_CATEGORIES}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dataset",
        default="locomo",
        choices=["locomo", "hotpotqa"],
        help="locomo = conversational memory; hotpotqa = the entity-diverse control",
    )
    parser.add_argument("--data", default=None, help="dataset path (defaults per --dataset)")
    parser.add_argument(
        "--questions-per-pool",
        type=int,
        default=20,
        help="hotpotqa only: questions pooled into one corpus, so retrieval is not trivially easy",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="evaluate only the first N CONVERSATIONS (for hotpotqa: N pools of "
             "--questions-per-pool questions each, so --limit 10 is 200 questions, not 200). "
             "Not to be confused with load_hotpotqa's own `limit`, which counts questions.",
    )
    parser.add_argument("--context-turns", type=int, default=5, help="context budget shared by every architecture")
    parser.add_argument("--max-turns", type=int, default=None, help="scale point: truncate each conversation to N turns")
    parser.add_argument(
        "--architectures",
        nargs="+",
        default=["L0", "L1"],
        choices=["L0", "L1", "L2", "L3", "hybrid"],
        help="which architectures to run (L2/L3/hybrid need GMS_LLM_BASE_URL/GMS_LLM_MODEL set)",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="also generate and score answers (one LLM call per QA item, per architecture)",
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        help="judge factual correctness, enabling Answer Accuracy (implies --generate)",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="model to judge with (default: GMS_JUDGE_MODEL, else the generation model — "
             "which is self-grading and warns)",
    )
    parser.add_argument(
        "--cache",
        nargs="?",
        const=str(REPO_ROOT / "data" / "processed" / "llm_cache.jsonl"),
        default=None,
        help="reuse identical LLM calls from a disk cache (extraction is deterministic, so a "
             "re-run costs nothing). Cached runs report accurate tokens but meaningless seconds.",
    )
    parser.add_argument(
        "--relation-weight-power",
        type=float,
        default=None,
        help="L3: re-weight edges by how well their relations match the query, raised to this "
             "power. Defaults to the architecture's own default (8.0); pass 0.0 for static "
             "triple counts, which measure ~0.025 recall worse at full scale.",
    )
    parser.add_argument(
        "--pagerank-alpha",
        type=float,
        default=None,
        help="L3: PageRank damping. 0.0 means the walk never leaves the seeds, i.e. no traversal "
             "at all — the correct baseline for any graph variant, since comparing against a "
             "worse graph configuration measures recovery from a self-inflicted floor.",
    )
    parser.add_argument(
        "--synonym-threshold",
        type=float,
        default=None,
        help="L3/hybrid: cosine above which HippoRAG-style synonym *edges* are added between "
             "similar nodes (they are linked, not merged — contrast --resolve-entities). 0.0 "
             "disables. Only ever measured on one conversation; see finding #5.",
    )
    parser.add_argument(
        "--resolve-entities",
        action="store_true",
        help="L3/hybrid: merge co-referent nodes (`burton` -> `tim burton`) before querying. Off "
             "by default because every published result predates it. Raises connectivity but "
             "lowers node specificity, so the net effect is a measurement, not a given.",
    )
    parser.add_argument(
        "--vector-slots",
        type=int,
        default=None,
        help="hybrid: how many of the --context-turns slots the vector channel keeps; the graph "
             "fills the rest. Default is half, rounded up. Equal to --context-turns reproduces L1 "
             "exactly; 0 is a pure vector-seeded graph.",
    )
    parser.add_argument(
        "--allow-large-run",
        action="store_true",
        help="permit a HotpotQA run larger than 50 pools, which the --limit guard otherwise refuses.",
    )
    parser.add_argument(
        "--dump-per-qa",
        action="store_true",
        help="also write one row per QA item per architecture (JSONL next to --out). Aggregates "
             "cannot be re-sliced afterwards; this makes any later analysis a re-count rather "
             "than a re-run.",
    )
    parser.add_argument("--out", default=str(REPO_ROOT / "data" / "processed" / "eval_retrieval.json"))
    return parser.parse_args()


def report(result: EvalResult) -> None:
    print(f"\n=== {result.architecture} ===")
    print(f"  conversations      : {result.conversations}")
    print(f"  turns ingested     : {result.turns_ingested:,}")
    print(f"  QA evaluated       : {result.qa_evaluated:,}")
    if result.qa_skipped_out_of_scale:
        print(f"  QA skipped (scale) : {result.qa_skipped_out_of_scale:,}  (evidence past the truncation)")
    if result.qa_skipped_no_evidence:
        print(f"  QA skipped (no ev) : {result.qa_skipped_no_evidence:,}  (no resolvable evidence)")
    if result.unresolvable_evidence_refs:
        print(f"  bad evidence refs  : {result.unresolvable_evidence_refs:,}  (malformed/dangling in source data)")

    print(f"\n  evidence recall    : {result.mean_evidence_recall():.3f}")
    print(f"  relevance (share)  : {result.relevance_share():.3f}  "
          f"(of a {result.mean_relevance_ceiling():.3f} ceiling — the comparable form)")
    print(f"  context relevance  : {result.mean_context_relevance():.3f}  (raw; rewards "
          f"returning fewer units)")
    print(f"  turns returned     : {result.mean_turns_returned():.2f} per query  "
          f"(budget was {result.architecture.split()[0]}'s own units, not turns)")

    cost = result.cost
    if cost.failures:
        share = 100 * cost.failures / result.turns_ingested if result.turns_ingested else 0
        print(f"\n  [WARN] {cost.failures:,} extraction failures ({share:.1f}% of turns) — "
              f"those turns contributed nothing to memory.")

    print("\n  cost")
    print(f"    construction     : {cost.construction_tokens:,} tokens, {cost.construction_seconds:.1f}s")
    print(f"    retrieval        : {cost.retrieval_tokens:,} tokens, {cost.retrieval_seconds:.1f}s over {cost.queries:,} queries")
    print(f"    storage entries  : {cost.storage_entries:,}")

    if result.generation_tokens:
        print("\n  generation")
        print(f"    ROUGE-L          : {result.mean_rouge_l():.3f}   "
              f"(over {len(result.answerable)} answerable questions)")
        print(f"    semantic sim     : {result.mean_semantic_similarity():.3f}")
        print(f"    refusal accuracy : {result.refusal_accuracy():.3f}   "
              f"(over {len(result.unanswerable)} unanswerable questions)")
        print(f"    false refusals   : {result.false_refusal_rate():.3f}   "
              f"(answerable questions wrongly declined)")
        print(f"    cost             : {result.generation_tokens:,} tokens, "
              f"{result.generation_seconds:.1f}s")

        accuracy = result.mean_answer_accuracy()
        if accuracy is not None:
            print(f"\n    factual correct. : {result.mean_factual_correctness():.3f}  (LLM-judged)")
            print(f"    ANSWER ACCURACY  : {accuracy:.3f}  "
                  f"(0.75*factual + 0.25*semantic)")
            print(f"    judge cost       : {result.judge_tokens:,} tokens, {result.judge_seconds:.1f}s")
            if result.judge_unparseable:
                print(f"    [WARN] {result.judge_unparseable} unreadable verdicts, excluded "
                      "rather than scored as incorrect.")
        else:
            print("\n    Answer Accuracy  : not computed (needs --judge; the formula is "
                  "three-quarters\n                       judged correctness, so the semantic "
                  "term alone is not it)")
        if result.false_refusal_rate() > 0.5:
            print("    [WARN] The model declines most answerable questions — refusal accuracy"
                  "\n           is meaningless until that is fixed.")

    by_category = sorted(result.by_category().items(), key=lambda kv: str(kv[0]))

    print("\n  by raw QA category — retrieval")
    print(f"    {'cat':<4} {'name':<22} {'n':>6} {'recall':>8} {'relevance':>10}")
    for category, scores in by_category:
        name = CATEGORY_NAMES.get(category, "?")
        print(
            f"    {str(category):<4} {name:<22} {scores['n']:>6.0f} "
            f"{scores['evidence_recall']:>8.3f} {scores['context_relevance']:>10.3f}"
        )

    if not result.generation_tokens:
        return

    def num(value, width=9):
        return f"{value:>{width}.3f}" if value is not None else f"{'—':>{width}}"

    print("\n  by raw QA category — generation")
    print("    (answerable questions are scored on answer quality, unanswerable ones on refusal)")
    print(f"    {'cat':<4} {'name':<22} {'n_ans':>6} {'ROUGE-L':>9} {'factual':>9} "
          f"{'ACCURACY':>9} {'false-ref':>10} {'n_unans':>8} {'refus-acc':>10}")
    for category, scores in by_category:
        name = CATEGORY_NAMES.get(category, "?")
        print(
            f"    {str(category):<4} {name:<22} {scores['n_answerable']:>6.0f} "
            f"{num(scores['rouge_l'])} {num(scores['factual_correctness'])} "
            f"{num(scores['answer_accuracy'])} {num(scores['false_refusal_rate'], 10)} "
            f"{scores['n_unanswerable']:>8.0f} {num(scores['refusal_accuracy'], 10)}"
        )


def main() -> None:
    args = parse_args()
    load_env_file()
    args.generate = args.generate or args.judge  # judging has nothing to judge without it

    if args.dataset == "hotpotqa":
        data = args.data or str(REPO_ROOT / "data" / "raw" / "hotpotqa" / "validation_distractor.parquet")
        conversations = load_hotpotqa(data, questions_per_pool=args.questions_per_pool)
    else:
        data = args.data or str(REPO_ROOT / "data" / "raw" / "locomo" / "locomo10.json")
        conversations = load_locomo(data)
    if args.limit is not None:
        conversations = conversations[: args.limit]

    # `--limit` counts conversations, but `load_hotpotqa`'s own `limit` counts
    # questions, and for HotpotQA a "conversation" is a pool of 20 of them. A
    # number that looks like a question count (`--limit 200`) therefore selects
    # 200 pools = 4,000 questions = 165k turns, a ~30h run, and the only visible
    # difference from the intended 10-pool run is one line of startup output.
    # This has now gone wrong twice, so refuse rather than warn.
    if args.dataset == "hotpotqa" and args.limit is not None:
        questions = sum(len(c.qa_items) for c in conversations)
        if args.limit > 50 and not args.allow_large_run:
            raise SystemExit(
                f"--limit {args.limit} selects {args.limit} POOLS "
                f"({questions:,} questions, {sum(c.num_turns() for c in conversations):,} turns), "
                f"which is almost certainly not what you meant. "
                f"For N questions pass --limit {max(1, args.limit // args.questions_per_pool)} "
                f"(pools of {args.questions_per_pool}). The published control runs used --limit 10 "
                f"= 200 questions. Pass --allow-large-run to override."
            )

    # Print the number that actually determines cost, not just the conversation
    # count: a structured rung issues one LLM call per turn, so turns are the
    # budget. An unlimited HotpotQA load is 306k turns — ~55 hours per structured
    # architecture — and reads indistinguishably from a small run if only the
    # conversation count is shown.
    total_turns = sum(c.num_turns() for c in conversations)
    total_qa = sum(len(c.qa_items) for c in conversations)
    structured = [a for a in args.architectures if a in {"L2", "L3", "hybrid"}]

    print(f"Evaluating {len(conversations)} conversation(s): {total_turns:,} turns, {total_qa:,} QA"
          f", context budget {args.context_turns}", end="")
    print(f", truncated to {args.max_turns} turns" if args.max_turns else "")

    if structured:
        calls = total_turns * len(structured)
        print(f"  {', '.join(structured)} will issue ~{calls:,} extraction calls "
              f"(~{calls * 0.65 / 3600:.1f}h at 0.65s/call, minus cache hits)")
        if calls > 50_000:
            print("  [WARN] That is a very long run. Use --limit to cap conversations.")

    results: list[EvalResult] = []

    # One client for extraction and generation alike, so the model stays fixed
    # across architectures — that is a deliberate experimental control.
    needs_llm = args.generate or {"L2", "L3", "hybrid"} & set(args.architectures)
    llm = None
    model_name = None  # captured before wrapping: the cache hides the inner client's attributes
    caches = []
    if needs_llm:
        from gms.llm import CachingLLMClient, VLLMClient

        llm = VLLMClient.from_env()
        model_name = llm.model
        if args.cache:
            llm = CachingLLMClient(llm, args.cache)
            caches.append(("generation/extraction", llm))
            print(f"\nLLM cache: {args.cache}")

    # Built once and shared across conversations: the memory is per-conversation,
    # the loaded model is not. The same embedder instance also keeps L1 and L2
    # on identical embeddings, so their scores differ by architecture only.
    # Generation needs it too, for the semantic-similarity term.
    embedder = None
    if args.generate or {"L1", "L2", "L3", "hybrid"} & set(args.architectures):
        from gms.architectures.embeddings import BGEEmbedder, CachingEmbedder

        print("\nLoading embedding model...")
        embedder = BGEEmbedder()
        if args.cache:
            # Embedding, not the LLM, is what a resumed run was re-paying: a full
            # HotpotQA pass embeds ~8k turns plus ~30k triples and nodes.
            embedder = CachingEmbedder(embedder, Path(args.cache).parent / "embedding_cache")
            caches.append(("embeddings", embedder))

    judge = None
    if args.judge:
        from gms.llm import VLLMClient

        judge_model = args.judge_model or os.environ.get("GMS_JUDGE_MODEL") or model_name
        # Passed explicitly so the judge does not inherit GMS_LLM_ENABLE_THINKING:
        # the toggle is per-model, and a model without a thinking mode 400s on it.
        judge_thinking = os.environ.get("GMS_JUDGE_ENABLE_THINKING")
        judge = VLLMClient.from_env(
            model=judge_model,
            enable_thinking=(
                None if judge_thinking is None else judge_thinking.lower() not in {"0", "false", "no"}
            ),
        )
        if args.cache:
            from gms.llm import CachingLLMClient

            judge = CachingLLMClient(judge, args.cache)
            caches.append(("judge", judge))
        if judge_model == model_name:
            print(f"\n[WARN] Judging with the generation model ({judge_model!r}). Self-grading"
                  "\n       inflates Answer Accuracy — a model recognises its own phrasing."
                  "\n       Relative comparisons between architectures survive this; absolute"
                  "\n       numbers are provisional. Set --judge-model to a different model.")
        else:
            print(f"\nJudging with {judge_model!r} (generation model is {model_name!r}).")

    gen_kwargs = {"llm": llm, "embedder": embedder, "judge": judge} if args.generate else {}
    if args.generate:
        print(f"Generating answers with {model_name!r} — one call per QA item, per architecture.")

    if "L0" in args.architectures:
        results.append(
            evaluate(
                "L0 flat_summary",
                lambda: FlatRecencyMemory(window=args.context_turns),
                conversations,
                max_turns=args.max_turns,
                **gen_kwargs,
            )
        )

    if "L1" in args.architectures:
        results.append(
            evaluate(
                "L1 vector",
                lambda: VectorMemory(embedder=embedder, top_k=args.context_turns),
                conversations,
                max_turns=args.max_turns,
                **gen_kwargs,
            )
        )

    if "L2" in args.architectures:
        print(f"Using extraction model {model_name!r} — one call per turn, so this is the slow one.")
        results.append(
            evaluate(
                "L2 kv_entity",
                lambda: KVEntityMemory(llm=llm, embedder=embedder, top_k=args.context_turns),
                conversations,
                max_turns=args.max_turns,
                **gen_kwargs,
            )
        )

    if "L3" in args.architectures:
        # Same client as L2's extraction: the extraction model is fixed
        # across every structured rung so structure is the only variable.
        graphs = []

        def build_graph():
            memory = GraphMemory(
                llm=llm,
                embedder=embedder,
                top_k=args.context_turns,
                **({} if args.relation_weight_power is None
                    else {"relation_weight_power": args.relation_weight_power}),
                **({} if args.pagerank_alpha is None
                    else {"alpha": args.pagerank_alpha}),
                **({"resolve_entities": True} if args.resolve_entities else {}),
                **({} if args.synonym_threshold is None
                    else {"synonym_threshold": args.synonym_threshold}),
            )
            graphs.append(memory)
            return memory

        print(f"Using extraction model {model_name!r} for L3 triples — one call per turn.")
        results.append(
            evaluate(
                "L3 graph_cheap",
                build_graph,
                conversations,
                max_turns=args.max_turns,
                **gen_kwargs,
            )
        )

        # Graph quality is in the metric stack and only L3/L3' can report it.
        if graphs:
            print("\n  graph quality (mean over conversations)")
            keys = ["nodes", "edges", "avg_degree", "clustering"]
            stats = [g.graph_stats() for g in graphs]
            for key in keys:
                mean_value = sum(s[key] for s in stats) / len(stats)
                print(f"    {key:<16} : {mean_value:.2f}")

    if "hybrid" in args.architectures:
        # Same extraction client again — the hybrid is L3's graph plus a vector
        # channel, so letting it extract with a different model would confound
        # the very comparison it exists to make.
        def build_hybrid():
            return HybridMemory(
                llm=llm,
                embedder=embedder,
                top_k=args.context_turns,
                **({} if args.vector_slots is None else {"vector_slots": args.vector_slots}),
                **({} if args.relation_weight_power is None
                    else {"relation_weight_power": args.relation_weight_power}),
                **({} if args.pagerank_alpha is None
                    else {"alpha": args.pagerank_alpha}),
                **({"resolve_entities": True} if args.resolve_entities else {}),
                **({} if args.synonym_threshold is None
                    else {"synonym_threshold": args.synonym_threshold}),
            )

        print(f"Using extraction model {model_name!r} for hybrid triples — one call per turn.")
        results.append(
            evaluate(
                "hybrid vector_seeded_graph",
                build_hybrid,
                conversations,
                max_turns=args.max_turns,
                **gen_kwargs,
            )
        )

    for result in results:
        report(result)

    if caches:
        print()
        for label, cache in caches:
            total = cache.hits + cache.misses
            if total:
                print(f"[CACHE] {label:<22} {cache.hits:,}/{total:,} served from disk")
        if any(c.hits for _, c in caches):
            print("        Token counts are the originals; elapsed seconds are not a cost measurement.")

    if len(results) > 1:
        print("\n=== comparison ===")
        print(f"  {'architecture':<18} {'recall':>8} {'relev(share)':>13} {'turns/query':>12} {'constr. tokens':>16}")
        for result in results:
            print(
                f"  {result.architecture:<18} {result.mean_evidence_recall():>8.3f} "
                f"{result.relevance_share():>13.3f} {result.mean_turns_returned():>12.2f} "
                f"{result.cost.construction_tokens:>16,}"
            )
        spread = {round(r.mean_turns_returned(), 1) for r in results}
        if len(spread) > 1:
            print("\n  [WARN] Architectures surfaced different numbers of turns per query, so"
                  "\n         recall is not directly comparable — budgets are matched in"
                  "\n         units, not rendered tokens. See the README on matched budgets.")

    payload = {
        # Every parameter that changes the numbers, recorded with them: a result
        # file that does not state its own configuration cannot be compared
        # against another run.
        "config": {
            "dataset": args.dataset,
            "conversations": len(conversations),
            "questions": sum(len(c.qa_items) for c in conversations),
            "turns": sum(c.num_turns() for c in conversations),
            "context_turns": args.context_turns,
            "max_turns": args.max_turns,
            "architectures": args.architectures,
            "relation_weight_power": (
                _RELATION_WEIGHT_POWER if args.relation_weight_power is None
                else args.relation_weight_power
            ),
            "pagerank_alpha": (
                _PAGERANK_ALPHA if args.pagerank_alpha is None else args.pagerank_alpha
            ),
            "vector_slots": args.vector_slots,
            "resolve_entities": args.resolve_entities,
            "synonym_threshold": args.synonym_threshold,
            "judge_model": judge_model if args.judge else None,
            "generation_model": model_name,
        },
        "results": [
            {
                "architecture": r.architecture,
                "conversations": r.conversations,
                "turns_ingested": r.turns_ingested,
                "qa_evaluated": r.qa_evaluated,
                "qa_skipped_out_of_scale": r.qa_skipped_out_of_scale,
                "qa_skipped_no_evidence": r.qa_skipped_no_evidence,
                "unresolvable_evidence_refs": r.unresolvable_evidence_refs,
                "evidence_recall": r.mean_evidence_recall(),
                "context_relevance": r.mean_context_relevance(),
                "relevance_share": r.relevance_share(),
                "relevance_ceiling": r.mean_relevance_ceiling(),
                "mean_turns_returned": r.mean_turns_returned(),
                "generation": {
                    "rouge_l": r.mean_rouge_l(),
                    "semantic_similarity": r.mean_semantic_similarity(),
                    "refusal_accuracy": r.refusal_accuracy(),
                    "false_refusal_rate": r.false_refusal_rate(),
                    "tokens": r.generation_tokens,
                    "seconds": r.generation_seconds,
                    "factual_correctness": r.mean_factual_correctness() if r.judge_tokens else None,
                    "answer_accuracy": r.mean_answer_accuracy(),
                    "judge_tokens": r.judge_tokens,
                    "judge_seconds": r.judge_seconds,
                    "judge_unparseable": r.judge_unparseable,
                } if r.generation_tokens else None,
                "cost": asdict(r.cost),
                "by_category": {str(k): v for k, v in r.by_category().items()},
            }
            for r in results
        ],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nWritten to {out_path}")

    if args.dump_per_qa:
        rows_path = out_path.with_suffix(".per_qa.jsonl")
        with rows_path.open("w", encoding="utf-8") as handle:
            for result in results:
                for row in result.per_qa:
                    handle.write(
                        json.dumps({"architecture": result.architecture, **asdict(row)}) + "\n"
                    )
        total = sum(len(r.per_qa) for r in results)
        print(f"Per-QA rows ({total:,}) written to {rows_path}")


if __name__ == "__main__":
    main()
