# Explanation quality: LLM alone vs LLM + RAG

**LLM-as-judge ratings (not human ratings).** Two judge models scored each blinded item 1 to 5.

Judges: A = `openai/gpt-oss-120b`, B = `openai/gpt-oss-20b`. Items: 4.

## Mean scores (both judges)

| condition | faithfulness | clarity | usefulness | overall |
|---|---|---|---|---|
| llm_only | 5.00 | 4.75 | 5.00 | 4.92 |
| llm_rag | 5.00 | 5.00 | 5.00 | 5.00 |

## Per judge (overall)

| judge | llm_only | llm_rag |
|---|---|---|
| judge_A | 5.00 | 5.00 |
| judge_B | 4.83 | 5.00 |

## RAG vs alone, per form

RAG better: 1, tie: 1, RAG worse: 0

## Agreement between the two judges

- exact agreement: 0.92
- within one point: 1.00
- Cohen's kappa: 0.00 (slight)
- quadratic-weighted kappa: 0.00 (slight)
- pairs compared: 12
