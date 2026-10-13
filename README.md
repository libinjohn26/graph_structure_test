# Minimal Viable Graph Memory

A controlled comparison of memory architectures for LLM agents: when does
graph-structured memory actually beat simpler alternatives, and what does it
cost?

Five architectures are implemented behind one interface and scored by one
harness, so the only thing that varies between them is how memory is built and
retrieved. The extraction model is held fixed across every structured rung,
which makes the comparison about *structure* rather than about extraction
quality.

| rung | name | construction | retrieval |
|---|---|---|---|
| L0 | `flat_summary` | none | recency window |
| L1 | `vector` | embed each turn | top-k cosine |
| L2 | `kv_entity` | LLM extracts entity → attributes, no edges | entity-name match, vector fallback |
| L3 | `graph_cheap` | LLM-extracted triples | personalized PageRank (HippoRAG2-style) |
| — | `hybrid` | L3's graph **plus** L1's turn embeddings | vector-retrieved turns seed the walk |

Results are reported in the accompanying paper, not here.

## Setup

```bash
python -m venv .venv && . .venv/Scripts/activate   # or bin/activate on POSIX
pip install -r requirements.txt

cp .env.example .env        # then fill in GMS_LLM_API_KEY
bash scripts/download_locomo.sh
bash scripts/download_hotpotqa.sh

python scripts/check_setup.py   # env, data, loaders
python scripts/check_llm.py     # server, model, extraction, judge
```

Datasets are not redistributed — the download scripts fetch
[LoCoMo](https://github.com/snap-research/locomo) (Maharana et al., ACL 2024)
and [HotpotQA](https://hotpotqa.github.io/) (Yang et al., EMNLP 2018) from
source.

The backbone is any OpenAI-compatible endpoint (built against vLLM). Embeddings
are `BAAI/bge-large-en-v1.5` via sentence-transformers, downloaded on first use.

**Run `check_llm.py` immediately before any long job.** It times a completion and
warns when the server is degraded, which is the failure mode that wastes hours.

## Running an evaluation

```bash
# retrieval + cost only, no backbone needed beyond extraction
python scripts/run_eval.py --architectures L0 L1 L2 L3 --cache --dump-per-qa

# add generated answers and LLM-judged factual correctness
python scripts/run_eval.py --architectures L1 L3 --judge --cache --dump-per-qa

# the entity-diverse control
python scripts/run_eval.py --dataset hotpotqa --limit 10 \
    --architectures L0 L1 L3 --judge --cache --dump-per-qa
```

Useful flags:

| flag | what it does |
|---|---|
| `--cache` | replay identical LLM calls from disk. Extraction, generation and judging all run at `temperature=0` with a fixed seed, so replay is exact |
| `--dump-per-qa` | write one row per question per architecture. **Use it on any long run** — aggregates cannot be re-sliced afterwards |
| `--context-turns` | context budget, shared by every architecture so none competes on volume |
| `--max-turns` | truncate conversations, for the scale axis |
| `--relation-weight-power` | L3/hybrid: weight edges by how well their relations match the query |
| `--pagerank-alpha` | `0.0` disables traversal — the baseline every graph variant must be compared against |
| `--resolve-entities` | merge co-referent nodes before querying |
| `--vector-slots` | hybrid: how many budget slots the vector channel keeps |

`--limit` counts **conversations**, not questions. For HotpotQA a conversation
is a pool of `--questions-per-pool` (20) questions, so `--limit 10` is a
200-question run. The script refuses a HotpotQA `--limit` above 50 rather than
silently starting a multi-day job.

## Analysis

These need no LLM at all — they work from the extraction cache and the per-question dumps.

```bash
python scripts/analyse_graphs.py                   # topology + gold-evidence reachability
python scripts/analyse_graphs.py --resolve-entities # the same, with co-referent nodes merged
python scripts/analyse_scale.py --dataset locomo   # recall against conversation length
python scripts/bootstrap_ci.py --per-qa <dump> --out <report>
python scripts/make_paper_tables.py --out paper/tables
```

`bootstrap_ci.py` is paired, 10,000 resamples, and reports intervals two ways:
resampling **questions**, and resampling whole **conversations**. Prefer the
second when generalising — questions inside one conversation share a graph and
an evidence pool, so they are not independent.

## Layout

```
src/gms/
  data/           dataset loaders → one shared schema (Turn/Session/QAItem/Conversation)
  architectures/  the five rungs, all subclassing MemoryArchitecture
  eval/           harness, metrics, generation, judge
scripts/          setup checks, the eval driver, analyses
tests/            219 tests, fully offline — model-backed pieces are faked
configs/          default.yaml: models, scale points, parameter grid
```

## Conventions worth knowing before changing anything

- **Everything downstream of a loader sees only the shared schema** in
  `data/schema.py`, never a dataset's native JSON. Swapping in a new corpus
  cannot break anything else.
- **`retrieve` returns a `Retrieval`**: the context the generator sees, plus
  `source_turn_ids` naming the turns it came from. Metrics score those ids, not
  the text — which is what lets L2's attributes and L3's triples be scored on
  the same footing as verbatim turns. **Populate the ids or the architecture
  scores zero on every retrieval metric.** Report each turn once.
- **Render turns with `format_turn`, match and embed on `turn_content`.**
  `format_turn` carries the session timestamp, which temporal questions need;
  `turn_content` omits it, because every turn in a session shares that string
  and embedding it only dilutes the match.
- **Compare at a matched context budget.** Unmatched budgets measure how much
  context each rung was allowed, not how well it chose.
- **Compare every graph variant against `--pagerank-alpha 0`**, never against a
  worse graph configuration. Unweighted traversal does not beat no-traversal, so
  "weighted edges beat unweighted edges" measures recovery from a self-inflicted
  floor.
- **An empty extraction is not a failure.** Greetings and backchannel state no
  facts — about a fifth of LoCoMo turns. The parsers return `None` for an
  unusable response and `[]` for "valid, nothing stated"; only the first counts
  as a failure.
- **`add_turn` and `retrieve` time their whole operation** into `CostLog`, not
  just the model call. Traversal, not the LLM, dominates graph retrieval cost.
- **Read tokens, not seconds.** Identical work produced byte-identical token
  counts and ~2.3x different wall-clock; cached runs replay tokens faithfully
  and latency not at all.

Run `python scripts/check_setup.py` after anything touching data loading or
dependencies, and `check_llm.py` before anything long.

## Metrics

Defined in `src/gms/eval/metrics.py` and `judge.py`, aggregated in `harness.py`.
For one question, **G** is the set of gold evidence turn ids and **R** the set
the architecture reports.

| metric | definition | averaged over |
|---|---|---|
| Evidence Recall | `|G ∩ R| / |G|` | all scored questions |
| Context Relevance | `|G ∩ R| / |R|` | all scored questions |
| `relevance_share` | relevance ÷ `min(|G|,|R|)/|R|` | questions with a non-zero ceiling |
| ROUGE-L | F-measure over the LCS of word tokens | answerable only |
| Answer Accuracy | `0.75·factual + 0.25·semantic` | answerable, judged only |
| Refusal accuracy | correct refusals ÷ unanswerable | unanswerable only |
| False-refusal rate | refusals ÷ answerable | answerable only |

All are macro-averages over questions, not pooled counts. Three behaviours that
change the numbers and are easy to miss:

- **A refusal to an answerable question scores 0 and never reaches the judge.**
  Judges grade a refusal CORRECT against any reference, because it contradicts
  nothing. Routing refusals through the judge inflated one rung's Answer
  Accuracy from 0.40 to 0.65 before this guard existed.
- **Answer Accuracy floors near 0.11, not 0**, because the semantic term
  inherits the embedding cosine floor. Prefer factual correctness when comparing
  architectures.
- **`evidence_recall = 0` does not mean the context was insufficient.** LoCoMo
  annotates one evidence turn while facts recur across a conversation, so
  retrieval can surface a different turn carrying the same fact and still score
  zero. Recall understates context sufficiency, and retrieval gains do not
  automatically become accuracy gains.

## Tests

```bash
python -m pytest tests/ -q
```

219 tests, no network and no model downloads: embedders and LLM clients are
injected, and the fakes are deterministic across processes (a fake built on
Python's salted `hash()` makes tests pass or fail by luck of the seed).
