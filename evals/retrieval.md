# Retrieval eval: which embedding model for Russian business documents

Corpus: `data/corpus/` (11 synthetic files, 68 chunks). A hit@k means a chunk from a gold (file, section) is in the top k. Qdrant in memory, fastembed on CPU. Reproduce: `uv run python scripts/eval_retrieval.py`.

## Claims from the planted-error set (wording close to the corpus) — 66 queries

| dense model | mode | recall@1 | recall@3 | recall@5 | MRR | ms/query |
|---|---|---|---|---|---|---|
| bge-small-en-v1.5 | dense | 47% | 61% | 70% | 0.56 | 11.2 |
| — (BM25 only, Russian stemmer) | sparse | 91% | 100% | 100% | 0.95 | 1.2 |
| bge-small-en-v1.5 | hybrid | 74% | 97% | 100% | 0.85 | 15.3 |
| paraphrase-multilingual-MiniLM-L12-v2 | dense | 74% | 89% | 92% | 0.82 | 7.8 |
| paraphrase-multilingual-MiniLM-L12-v2 | hybrid | 88% | 98% | 98% | 0.93 | 10.2 |
| paraphrase-multilingual-mpnet-base-v2 | dense | 79% | 94% | 96% | 0.86 | 18.8 |
| paraphrase-multilingual-mpnet-base-v2 | hybrid | 86% | 98% | 100% | 0.92 | 22.5 |
| multilingual-e5-large | dense | 89% | 98% | 98% | 0.93 | 71.9 |
| multilingual-e5-large | hybrid | 92% | 100% | 100% | 0.96 | 73.5 |

## Paraphrased colloquial questions (wording avoids the corpus) — 45 queries

| dense model | mode | recall@1 | recall@3 | recall@5 | MRR | ms/query |
|---|---|---|---|---|---|---|
| bge-small-en-v1.5 | dense | 7% | 11% | 20% | 0.11 | 12.3 |
| — (BM25 only, Russian stemmer) | sparse | 22% | 47% | 56% | 0.35 | 1.5 |
| bge-small-en-v1.5 | hybrid | 16% | 36% | 44% | 0.26 | 14.0 |
| paraphrase-multilingual-MiniLM-L12-v2 | dense | 29% | 60% | 78% | 0.47 | 5.9 |
| paraphrase-multilingual-MiniLM-L12-v2 | hybrid | 33% | 62% | 78% | 0.48 | 9.5 |
| paraphrase-multilingual-mpnet-base-v2 | dense | 51% | 69% | 82% | 0.61 | 16.5 |
| paraphrase-multilingual-mpnet-base-v2 | hybrid | 49% | 62% | 73% | 0.57 | 19.6 |
| multilingual-e5-large | dense | 80% | 96% | 98% | 0.88 | 59.0 |
| multilingual-e5-large | hybrid | 60% | 96% | 98% | 0.76 | 59.0 |

Index time per model (embedding the corpus on CPU): bge-small-en-v1.5 5.4 s, paraphrase-multilingual-MiniLM-L12-v2 2.3 s, paraphrase-multilingual-mpnet-base-v2 6.8 s, multilingual-e5-large 19.0 s

Best hybrid result on the paraphrased set: **intfloat/multilingual-e5-large**.

## Misses of multilingual-e5-large (hybrid, paraphrased, rank > 3)

- `p11` rank 5; top-1 was 01_tarify_rf.md / Тариф «Команда»
- `p38` rank >5; top-1 was 07_podderzhka.md / База знаний и статус сервиса
