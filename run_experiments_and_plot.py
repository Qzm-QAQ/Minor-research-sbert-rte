#!/usr/bin/env python3
"""
RTE (Recognizing Textual Entailment) experiments:
1) Sentence-BERT embeddings as features + SVM (4 kernels) with 3 pair-vector modes
2) Bag-of-Words baseline (NLTK tokenize + POS filter + stemming) + SVM (4 kernels)

Outputs:
- results.csv
- results_sbert_heatmap.png
- results_bow_bar.png
- best_confusion_matrix.png
"""
import argparse
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt

# --- optional imports (installed via requirements) ---
# Sentence-Transformers
from sentence_transformers import SentenceTransformer

# NLTK baseline
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


def load_rte_tsv(path: Path):
    """
    Robust TSV loader that does NOT apply CSV quote parsing.
    Format: index<TAB>sentence1<TAB>sentence2<TAB>label
    """
    rows = []
    with path.open("r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t", 3)  # max 4 fields
            if len(parts) != 4:
                # skip malformed lines
                continue
            idx, s1, s2, lab = parts
            lab = lab.strip()
            if lab not in ("entailment", "not_entailment"):
                continue
            rows.append((idx, s1, s2, lab))
    return rows


def y_from_label(lab: str) -> int:
    return 1 if lab == "entailment" else 0


def ensure_nltk_resources():
    """
    Download NLTK resources if missing.
    (Needs internet the first time on a new machine.)
    """
    # Tokenizer
    try:
        word_tokenize("test.")
    except LookupError:
        nltk.download("punkt")
        # some environments require this additionally
        try:
            nltk.download("punkt_tab")
        except Exception:
            pass

    # POS tagger (different NLTK versions may require different packages)
    try:
        nltk.pos_tag(["test"])
    except LookupError:
        nltk.download("averaged_perceptron_tagger")
        try:
            nltk.download("averaged_perceptron_tagger_eng")
        except Exception:
            pass


# -------------------------
# SBERT features
# -------------------------
def sbert_encode_sentences(model: SentenceTransformer, sentences, batch_size=32):
    return model.encode(
        sentences,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )


def build_sbert_pair_vectors(pairs, model: SentenceTransformer, mode: str):
    """
    mode in {"concat", "mean", "max"}
    """
    # cache per unique sentence
    uniq = {}
    order = []
    for _, s1, s2, _ in pairs:
        if s1 not in uniq:
            uniq[s1] = None
            order.append(s1)
        if s2 not in uniq:
            uniq[s2] = None
            order.append(s2)

    embs = sbert_encode_sentences(model, order)
    for sent, emb in zip(order, embs):
        uniq[sent] = emb

    X = []
    y = []
    for _, s1, s2, lab in pairs:
        v1 = uniq[s1]
        v2 = uniq[s2]
        if mode == "concat":
            vec = np.concatenate([v1, v2], axis=0)
        elif mode == "mean":
            vec = (v1 + v2) / 2.0
        elif mode == "max":
            vec = np.maximum(v1, v2)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        X.append(vec)
        y.append(y_from_label(lab))

    return np.vstack(X).astype(np.float32), np.array(y, dtype=np.int64)


# -------------------------
# BOW baseline features
# -------------------------
KEEP_PREFIX = ("NN", "VB", "JJ", "RB")
STEMMER = PorterStemmer()


def preprocess_words(sentence: str):
    # tokenization
    tokens = word_tokenize(sentence)
    # POS tagging
    tagged = nltk.pos_tag(tokens)
    # filter: nouns/verbs/adjectives/adverbs
    kept = [w.lower() for (w, t) in tagged if t.startswith(KEEP_PREFIX)]
    # stemming
    stems = [STEMMER.stem(w) for w in kept]
    return stems


def pair_to_feature_list(s1: str, s2: str):
    feats = [f"{w}/s1" for w in preprocess_words(s1)]
    feats += [f"{w}/s2" for w in preprocess_words(s2)]
    return feats


def build_bow_dictionary(train_pairs, test_pairs):
    """
    Assignment requirement: build the feature dictionary using ALL features
    appearing in both train and test.
    """
    vocab = {}
    all_pairs = train_pairs + test_pairs
    all_feat_lists = []
    for _, s1, s2, _ in all_pairs:
        feats = pair_to_feature_list(s1, s2)
        all_feat_lists.append(feats)
        for f in feats:
            if f not in vocab:
                vocab[f] = len(vocab)
    return vocab, all_feat_lists


def vectorize_from_vocab(pairs, vocab, feat_lists):
    rows, cols, data = [], [], []
    y = []
    for i, (pair, feats) in enumerate(zip(pairs, feat_lists)):
        _, _, _, lab = pair
        y.append(y_from_label(lab))
        cnt = Counter(feats)
        for f, c in cnt.items():
            j = vocab.get(f)
            if j is None:
                continue
            rows.append(i)
            cols.append(j)
            data.append(float(c))

    # use a sparse matrix (scikit-learn accepts sparse input)
    from scipy.sparse import csr_matrix
    X = csr_matrix((data, (rows, cols)), shape=(len(pairs), len(vocab)), dtype=np.float32)
    return X, np.array(y, dtype=np.int64)


# -------------------------
# Train/Eval
# -------------------------
def train_eval_svm(X_train, y_train, X_test, y_test, kernel: str):
    clf = SVC(kernel=kernel)  # default params required by assignment
    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    fit_sec = time.perf_counter() - t0
    pred = clf.predict(X_test)
    acc = accuracy_score(y_test, pred)
    return acc, fit_sec, pred


def plot_sbert_heatmap(df: pd.DataFrame, out_path: Path):
    sub = df[(df["feature"] == "sbert")]
    modes = ["concat", "mean", "max"]
    kernels = ["rbf", "linear", "poly", "sigmoid"]
    mat = np.zeros((len(modes), len(kernels)), dtype=float)
    for i, m in enumerate(modes):
        for j, k in enumerate(kernels):
            v = sub[(sub["pair_mode"] == m) & (sub["kernel"] == k)]["accuracy"].values
            mat[i, j] = float(v[0]) if len(v) else np.nan

    plt.figure()
    plt.imshow(mat, aspect="auto")
    plt.xticks(range(len(kernels)), kernels)
    plt.yticks(range(len(modes)), modes)
    plt.colorbar()
    plt.title("SBERT + SVM accuracy (test)")
    plt.xlabel("kernel")
    plt.ylabel("pair vector mode")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_bow_bar(df: pd.DataFrame, out_path: Path):
    sub = df[(df["feature"] == "bow")].sort_values("kernel")
    plt.figure()
    plt.bar(sub["kernel"].tolist(), sub["accuracy"].tolist())
    plt.title("BOW baseline + SVM accuracy (test)")
    plt.xlabel("kernel")
    plt.ylabel("accuracy")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_confusion(y_true, y_pred, out_path: Path, title: str):
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])  # 1: entailment, 0: not
    plt.figure()
    plt.imshow(cm, aspect="auto")
    plt.xticks([0, 1], ["entailment", "not_entailment"])
    plt.yticks([0, 1], ["entailment", "not_entailment"])
    plt.colorbar()
    plt.title(title)
    plt.xlabel("pred")
    plt.ylabel("true")
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="rte_train.txt", help="path to rte_train.txt")
    ap.add_argument("--test", default="rte_test.txt", help="path to rte_test.txt")
    ap.add_argument("--sbert_model", default="distilbert-base-nli-mean-tokens",
                    help="SentenceTransformer model name")
    ap.add_argument("--out_dir", default="outputs", help="output directory")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_pairs = load_rte_tsv(Path(args.train))
    test_pairs = load_rte_tsv(Path(args.test))

    # dataset sanity check
    train_labels = Counter([p[3] for p in train_pairs])
    test_labels = Counter([p[3] for p in test_pairs])
    print(f"Train size: {len(train_pairs)}  label dist: {dict(train_labels)}")
    print(f"Test  size: {len(test_pairs)}  label dist: {dict(test_labels)}")

    kernels = ["rbf", "linear", "poly", "sigmoid"]
    results = []

    # --- SBERT experiments ---
    print("\n=== SBERT + SVM ===")
    model = SentenceTransformer(args.sbert_model)
    best = {"accuracy": -1.0}

    for mode in ["concat", "mean", "max"]:
        X_train, y_train = build_sbert_pair_vectors(train_pairs, model, mode=mode)
        X_test, y_test = build_sbert_pair_vectors(test_pairs, model, mode=mode)
        for k in kernels:
            acc, fit_sec, pred = train_eval_svm(X_train, y_train, X_test, y_test, kernel=k)
            print(f"SBERT mode={mode:6s} kernel={k:7s} acc={acc:.4f} fit_sec={fit_sec:.2f}")
            results.append({
                "feature": "sbert",
                "pair_mode": mode,
                "kernel": k,
                "accuracy": acc,
                "fit_sec": fit_sec,
            })
            if acc > best["accuracy"]:
                best = {
                    "feature": "sbert",
                    "pair_mode": mode,
                    "kernel": k,
                    "accuracy": acc,
                    "y_true": y_test,
                    "y_pred": pred,
                }

    # --- BOW baseline ---
    print("\n=== BOW baseline + SVM ===")
    ensure_nltk_resources()
    vocab, all_feat_lists = build_bow_dictionary(train_pairs, test_pairs)
    train_feats = all_feat_lists[: len(train_pairs)]
    test_feats = all_feat_lists[len(train_pairs):]
    Xtr, ytr = vectorize_from_vocab(train_pairs, vocab, train_feats)
    Xte, yte = vectorize_from_vocab(test_pairs, vocab, test_feats)

    for k in kernels:
        acc, fit_sec, pred = train_eval_svm(Xtr, ytr, Xte, yte, kernel=k)
        print(f"BOW  kernel={k:7s} acc={acc:.4f} fit_sec={fit_sec:.2f}")
        results.append({
            "feature": "bow",
            "pair_mode": "-",
            "kernel": k,
            "accuracy": acc,
            "fit_sec": fit_sec,
        })
        if acc > best["accuracy"]:
            best = {
                "feature": "bow",
                "pair_mode": "-",
                "kernel": k,
                "accuracy": acc,
                "y_true": yte,
                "y_pred": pred,
            }

    df = pd.DataFrame(results)
    csv_path = out_dir / "results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # plots
    plot_sbert_heatmap(df, out_dir / "results_sbert_heatmap.png")
    plot_bow_bar(df, out_dir / "results_bow_bar.png")
    plot_confusion(best["y_true"], best["y_pred"], out_dir / "best_confusion_matrix.png",
                   title=f"Best model: {best['feature']} {best['pair_mode']} {best['kernel']}  acc={best['accuracy']:.4f}")

    print("Saved plots in:", out_dir.resolve())


if __name__ == "__main__":
    main()
