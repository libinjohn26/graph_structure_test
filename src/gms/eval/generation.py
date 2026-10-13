"""Answer generation: turn retrieved context into an answer to score.

The prompt is shaped by two measured properties of LoCoMo rather than by taste:

- **Gold answers are short** — a median of 3 words (`scripts/explore_locomo.py`).
  ROUGE-L against a three-word reference collapses if the model replies in
  sentences, so brevity is instructed explicitly. A verbose model would look
  inaccurate when it was merely chatty.
- **22% of questions are unanswerable by design** (LoCoMo's adversarial
  category, which omits the answer field entirely). The benchmark expects the
  agent to *say so*, so the prompt gives refusal an explicit, detectable form
  instead of leaving the model to phrase it however it likes.

Refusal is detected by sentinel only, never by fuzzy matching on phrases like
"the context does not mention". Fuzzy matching would paper over a model that
ignores the instruction; a sentinel makes compliance measurable, which is why
`EvalResult` reports refusal rates rather than assuming them.
"""

from __future__ import annotations

from gms.llm import LLMClient, LLMResponse

REFUSAL_SENTINEL = "NOT_ANSWERABLE"

ANSWER_SYSTEM = (
    "You answer questions about a conversation using only the context provided. "
    "Answer in as few words as possible — a name, a date, or a short phrase, never a sentence. "
    f"If the context does not contain the answer, reply with exactly {REFUSAL_SENTINEL} "
    "and nothing else."
)

ANSWER_PROMPT = """Context from the conversation:
{context}

Question: {question}
Answer:"""


def build_prompt(context: str, question: str) -> str:
    return ANSWER_PROMPT.format(context=context or "(no context retrieved)", question=question)


def is_refusal(answer: str) -> bool:
    """Whether the model declined to answer, by the sentinel it was told to use."""
    return REFUSAL_SENTINEL.lower() in answer.strip().lower()


def generate_answer(llm: LLMClient, context: str, question: str) -> LLMResponse:
    return llm.complete(build_prompt(context, question), system=ANSWER_SYSTEM)
