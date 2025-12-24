# Minor Research: SBERT for RTE

A simple, reproducible baseline for **Recognizing Textual Entailment (RTE)** using **Sentence-BERT** embeddings.
Given a pair of sentences *(premise, hypothesis)*, the model predicts whether the premise **entails** the hypothesis.

---

## What’s inside

- Sentence-pair classification pipeline (train → evaluate → save outputs)
- Minimal CLI (train/test paths + output directory)
- Reproducible experiment folder (`--out_dir`) for logs/metrics/artifacts

> Tip: this repository is designed for coursework / minor research reporting—keep it small and easy to rerun.

---

## Quick start

### 1) Setup

```bash
pip install -r requirements.txt
```
### 2) Run training + evaluation
```bash
python run.py --train rte_train.txt --test rte_test.txt --out_dir outputs
```
