# Retrieval eval: which embedding model for Russian business documents

Corpus: `data/corpus/` (11 synthetic files, 68 chunks). A hit@k means a chunk from a gold (file, section) is in the top k. Qdrant in memory, fastembed on CPU. Reproduce: `uv run python scripts/eval_retrieval.py`.

## Claims from the planted-error set (wording close to the corpus) — 99 queries

| dense model | mode | recall@1 | recall@3 | recall@5 | MRR | ms/query |
|---|---|---|---|---|---|---|
| multilingual-e5-large | dense | 85% | 99% | 99% | 0.91 | 63.9 |
| — (BM25 only, Russian stemmer) | sparse | 87% | 97% | 98% | 0.92 | 1.2 |
| multilingual-e5-large | hybrid | 87% | 99% | 100% | 0.93 | 67.9 |

## Paraphrased colloquial questions (wording avoids the corpus) — 45 queries

| dense model | mode | recall@1 | recall@3 | recall@5 | MRR | ms/query |
|---|---|---|---|---|---|---|
| multilingual-e5-large | dense | 80% | 96% | 98% | 0.88 | 57.7 |
| — (BM25 only, Russian stemmer) | sparse | 22% | 47% | 56% | 0.35 | 1.1 |
| multilingual-e5-large | hybrid | 60% | 96% | 98% | 0.76 | 60.1 |

Index time per model (embedding the corpus on CPU): multilingual-e5-large 18.3 s

Best hybrid result on the paraphrased set: **intfloat/multilingual-e5-large**.

## Misses of multilingual-e5-large (hybrid, paraphrased, rank > 3)

- `p11` rank 5; top-1 was 01_tarify_rf.md / Тариф «Команда»
- `p38` rank >5; top-1 was 07_podderzhka.md / База знаний и статус сервиса
