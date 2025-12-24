#!/usr/bin/env python3
"""
RTE (Recognizing Textual Entailment) experiments:

Main experiments (assignment-required):
1) Sentence-BERT embeddings as features + SVM (4 kernels) with 3 pair-vector modes
2) Bag-of-Words baseline (NLTK tokenize + POS filter + stemming) + SVM (4 kernels)

Required outputs:
- results.csv
- results_sbert_heatmap.png
- results_bow_bar.png
- best_confusion_matrix.png

Additional visualizations (do NOT change the main experiment logic):
- training_time_comparison.png      (compare fit_sec across settings)
- best_feature_pca.png              (PCA 2D visualization on best feature vectors)
- best_confusion_matrix_norm.png    (normalized confusion matrix)
- sgd_hinge_training_curve.png      (illustrative "training loss curve" via SGDClassifier hinge loss)
"""

import argparse
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.utils import shuffle as sk_shuffle
from sklearn.linear_model import SGDClassifier


# Sentence-Transformers
from sentence_transformers import SentenceTransformer

# NLTK baseline
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


# =========================================================
# 0) Data Loader
# =========================================================
def load_rte_tsv(path: Path):
    """
    Robust TSV loader.

    - Does NOT assume a header.
    - Handles extra TABs inside sentence fields by splitting label from the right.
    - Expected format:
        index<TAB>sentence1<TAB>sentence2<TAB>label
      where label in {"entailment", "not_entailment"}.

    Notes:
    - If a sentence contains extra tabs, we still recover fields by:
        (a) rsplit last tab => label
        (b) split remaining head into idx, s1, s2 (s2 may include tabs if present)

    Returns:
      list of tuples: (idx, s1, s2, lab)
    """
    rows = []
    bad = []

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for ln, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line.strip():
                continue

            # Split label from the RIGHT to survive tabs inside sentences
            try:
                head, lab = line.rsplit("\t", 1)
            except ValueError:
                bad.append((ln, "rsplit_fail", line))
                continue

            # Normalize label a bit (safe; does not change correct labels)
            lab_norm = lab.strip().lower().replace("-", "_").replace(" ", "_")

            # Now parse head: need at least idx + s1 + s2
            parts = head.split("\t")
            if len(parts) < 3:
                bad.append((ln, f"too_few_fields={len(parts)}", line))
                continue

            idx = parts[0]
            s1 = parts[1]
            # s2 may include extra tabs; join the rest
            s2 = "\t".join(parts[2:])

            # Skip possible header line
            if idx.strip().lower() == "index" or lab_norm == "label":
                continue

            if lab_norm not in ("entailment", "not_entailment"):
                bad.append((ln, f"bad_label={lab!r}", line))
                continue

            rows.append((idx, s1, s2, lab_norm))

    if bad:
        print(f"[WARN] {path}: {len(bad)} malformed lines (show up to 5):")
        for ln, why, txt in bad[:5]:
            print(f"  line {ln} [{why}] :: {txt[:140]}")

    return rows


def y_from_label(lab: str) -> int:
    """entailment -> 1, not_entailment -> 0"""
    return 1 if lab == "entailment" else 0


# =========================================================
# 1) NLTK resources
# =========================================================
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
        # Some environments require this additionally
        try:
            nltk.download("punkt_tab")
        except Exception:
            pass

    # POS tagger
    try:
        nltk.pos_tag(["test"])
    except LookupError:
        nltk.download("averaged_perceptron_tagger")
        # Some environments require this additionally
        try:
            nltk.download("averaged_perceptron_tagger_eng")
        except Exception:
            pass


# =========================================================
# 2) SBERT Features
# =========================================================
def sbert_encode_sentences(model: SentenceTransformer, sentences, batch_size=32):
    """
    Encode sentences into embeddings.
    normalize_embeddings=False to keep original behavior.
    """
    return model.encode(
        sentences,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )


def build_sbert_pair_vectors(pairs, model: SentenceTransformer, mode: str):
    """
    Build vector representation for sentence pair (s1, s2).

    mode in {"concat", "mean", "max"} as required by the assignment:
    (1) concat: [vec(s1); vec(s2)] -> dim 1536
    (2) mean:   (vec(s1)+vec(s2))/2 -> dim 768
    (3) max:    element-wise max    -> dim 768
    """
    # Cache per unique sentence to avoid recomputing within this call
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


# =========================================================
# 3) BOW baseline Features (NLTK)
# =========================================================
KEEP_PREFIX = ("NN", "VB", "JJ", "RB")  # noun/verb/adj/adv
STEMMER = PorterStemmer()

# Negation words are important for entailment; keep them even if POS filter removes them.
NEG_KEEP = {"no", "not", "n't", "never"}


def preprocess_words(sentence: str):
    """
    Baseline preprocessing:
    - tokenize
    - POS tagging
    - keep content words: noun/verb/adj/adv
    - additionally keep negation words
    - stemming
    """
    tokens = word_tokenize(sentence)
    tagged = nltk.pos_tag(tokens)

    kept = [
        w.lower()
        for (w, t) in tagged
        if t.startswith(KEEP_PREFIX) or w.lower() in NEG_KEEP
    ]

    stems = [STEMMER.stem(w) for w in kept]
    return stems


def pair_to_feature_list(s1: str, s2: str):
    """
    Construct feature list for a pair:
    - words from s1 with suffix /s1
    - words from s2 with suffix /s2
    """
    feats = [f"{w}/s1" for w in preprocess_words(s1)]
    feats += [f"{w}/s2" for w in preprocess_words(s2)]
    return feats


def build_bow_dictionary(train_pairs, test_pairs):
    """
    Assignment requirement:
    build vocabulary using ALL features in both training and test data.
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
    """
    Convert feature lists to sparse count vectors using the given vocab.
    Each entry is the occurrence count of that feature.
    """
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

    from scipy.sparse import csr_matrix
    X = csr_matrix(
        (data, (rows, cols)),
        shape=(len(pairs), len(vocab)),
        dtype=np.float32
    )
    return X, np.array(y, dtype=np.int64)


# =========================================================
# 4) Train / Eval (SVM)
# =========================================================
def train_eval_svm(X_train, y_train, X_test, y_test, kernel: str):
    """
    Train SVC with default parameters (as required), evaluate accuracy.
    Returns: (accuracy, fit_seconds, y_pred)
    """
    clf = SVC(kernel=kernel)  # default params
    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    fit_sec = time.perf_counter() - t0

    pred = clf.predict(X_test)
    acc = accuracy_score(y_test, pred)
    return acc, fit_sec, pred


# =========================================================
# 5) Plotting utilities (Matplotlib only)
# =========================================================
def plot_sbert_heatmap(df: pd.DataFrame, out_path: Path):
    """Heatmap: SBERT accuracy by pair_mode x kernel"""
    sub = df[df["feature"] == "sbert"].copy()

    modes = ["concat", "mean", "max"]
    kernels = ["rbf", "linear", "poly", "sigmoid"]

    mat = np.full((len(modes), len(kernels)), np.nan, dtype=float)
    for i, m in enumerate(modes):
        for j, k in enumerate(kernels):
            v = sub[(sub["pair_mode"] == m) & (sub["kernel"] == k)]["accuracy"].values
            if len(v):
                mat[i, j] = float(v[0])

    plt.figure(figsize=(7.2, 3.8))
    im = plt.imshow(mat, aspect="auto")
    plt.xticks(range(len(kernels)), kernels)
    plt.yticks(range(len(modes)), modes)
    plt.colorbar(im)
    plt.title("SBERT + SVM accuracy (test)")
    plt.xlabel("kernel")
    plt.ylabel("pair vector mode")

    # annotate numbers
    for (i, j), val in np.ndenumerate(mat):
        if not np.isnan(val):
            plt.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_bow_bar(df: pd.DataFrame, out_path: Path):
    """Bar plot: BOW accuracy by kernel"""
    sub = df[df["feature"] == "bow"].sort_values("kernel")
    plt.figure(figsize=(6.2, 3.6))
    plt.bar(sub["kernel"].tolist(), sub["accuracy"].tolist())
    plt.title("BOW baseline + SVM accuracy (test)")
    plt.xlabel("kernel")
    plt.ylabel("accuracy")
    plt.ylim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_confusion(y_true, y_pred, out_path: Path, title: str):
    """Raw confusion matrix"""
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])  # 1: entailment, 0: not
    plt.figure(figsize=(5.2, 4.4))
    im = plt.imshow(cm, aspect="auto")
    plt.xticks([0, 1], ["entailment", "not_entailment"])
    plt.yticks([0, 1], ["entailment", "not_entailment"])
    plt.colorbar(im)
    plt.title(title)
    plt.xlabel("pred")
    plt.ylabel("true")
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_confusion_normalized(y_true, y_pred, out_path: Path, title: str):
    """Normalized confusion matrix (row-normalized)"""
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    cm = cm.astype(np.float64)
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sum, out=np.zeros_like(cm), where=row_sum != 0)

    plt.figure(figsize=(5.2, 4.4))
    im = plt.imshow(cm_norm, aspect="auto", vmin=0.0, vmax=1.0)
    plt.xticks([0, 1], ["entailment", "not_entailment"])
    plt.yticks([0, 1], ["entailment", "not_entailment"])
    plt.colorbar(im)
    plt.title(title)
    plt.xlabel("pred")
    plt.ylabel("true")
    for (i, j), v in np.ndenumerate(cm_norm):
        plt.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_training_time_comparison(df: pd.DataFrame, out_path: Path):
    """
    Compare training time (fit_sec) across feature/kernel settings.
    We aggregate SBERT settings by (pair_mode, kernel) and plot them;
    plus BOW by kernel.
    """
    plt.figure(figsize=(8.6, 4.2))

    # Prepare categories
    sub_sbert = df[df["feature"] == "sbert"].copy()
    sub_bow = df[df["feature"] == "bow"].copy()

    # x labels: SBERT-mode-kernel + BOW-kernel
    labels = []
    times = []

    # SBERT
    for _, r in sub_sbert.iterrows():
        labels.append(f"sbert-{r['pair_mode']}-{r['kernel']}")
        times.append(float(r["fit_sec"]))

    # BOW
    for _, r in sub_bow.iterrows():
        labels.append(f"bow-{r['kernel']}")
        times.append(float(r["fit_sec"]))

    x = np.arange(len(labels))
    plt.bar(x, times)
    plt.xticks(x, labels, rotation=60, ha="right")
    plt.ylabel("fit time (sec)")
    plt.title("Training time comparison (SVC fit_sec)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_best_feature_pca(best_feature_name: str, X_train, y_train, out_path: Path, max_points=1500):
    """
    PCA visualization (2D) on the training feature vectors.
    This helps show whether entailment/not_entailment are separable.

    Note: For BOW (sparse), PCA on sparse is possible if converted to dense,
          but dense may be huge. Here we only do PCA for SBERT features by default.
    """
    # Subsample for speed/clarity
    n = min(len(y_train), max_points)
    idx = np.arange(len(y_train))
    np.random.seed(0)
    np.random.shuffle(idx)
    idx = idx[:n]

    Xs = X_train[idx]
    ys = y_train[idx]

    # PCA needs dense
    if not isinstance(Xs, np.ndarray):
        Xs = Xs.toarray()

    pca = PCA(n_components=2, random_state=0)
    Z = pca.fit_transform(Xs)

    plt.figure(figsize=(6.2, 5.6))
    # y=1 entailment, y=0 not_entailment
    for cls, name in [(1, "entailment"), (0, "not_entailment")]:
        mask = (ys == cls)
        plt.scatter(Z[mask, 0], Z[mask, 1], s=12, alpha=0.7, label=name)

    plt.title(f"PCA 2D of best feature vectors ({best_feature_name})")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# =========================================================
# 6) "Training loss curve" (illustrative) via SGDClassifier
# =========================================================
def hinge_loss(y_pm1: np.ndarray, scores: np.ndarray) -> float:
    """
    Hinge loss: mean(max(0, 1 - y * score)), where y in {-1, +1}
    """
    return float(np.maximum(0.0, 1.0 - y_pm1 * scores).mean())


def run_sgd_hinge_training_curve(
    X_train, y_train,
    out_path: Path,
    epochs: int = 25,
    random_state: int = 0,
):
    """
    SVC does not expose iterative training loss.
    To plot a "training curve", we run SGDClassifier with hinge loss (linear SVM)
    and record hinge loss / training accuracy per epoch.

    IMPORTANT:
    - This does NOT replace the main SVC experiment.
    - It's only used to visualize an optimization process ("training curve").
    """
    # Convert labels {0,1} -> {-1,+1} for hinge loss computation
    y_pm1 = np.where(y_train == 1, 1.0, -1.0).astype(np.float64)

    # SGDClassifier expects dense or sparse OK
    clf = SGDClassifier(
        loss="hinge",
        random_state=random_state,
        max_iter=1,          # we will control epochs manually
        tol=None,
        learning_rate="optimal",
        early_stopping=False,
    )

    losses = []
    accs = []

    # We need classes for partial_fit on first call
    classes = np.array([0, 1], dtype=np.int64)

    X, y = X_train, y_train
    for ep in range(1, epochs + 1):
        # Shuffle each epoch for SGD stability
        X, y = sk_shuffle(X, y, random_state=random_state + ep)

        if ep == 1:
            clf.partial_fit(X, y, classes=classes)
        else:
            clf.partial_fit(X, y)

        # decision_function works for hinge
        scores = clf.decision_function(X_train)
        # decision_function returns shape (n,) for binary
        loss = hinge_loss(y_pm1, scores)
        pred = (scores >= 0).astype(np.int64)
        acc = float((pred == y_train).mean())

        losses.append(loss)
        accs.append(acc)

    # Plot training curve
    plt.figure(figsize=(6.6, 4.2))
    xs = np.arange(1, epochs + 1)
    plt.plot(xs, losses, marker="o", linewidth=1.5, label="hinge loss (train)")
    plt.plot(xs, accs, marker="x", linewidth=1.5, label="accuracy (train)")
    plt.xlabel("epoch")
    plt.title("SGDClassifier (hinge) training curve (illustrative)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# =========================================================
# 7) Main
# =========================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="rte_train.txt", help="path to rte_train.txt")
    ap.add_argument("--test", default="rte_test.txt", help="path to rte_test.txt")
    ap.add_argument("--sbert_model", default="distilbert-base-nli-mean-tokens",
                    help="SentenceTransformer model name")
    ap.add_argument("--out_dir", default="outputs", help="output directory")
    # extra controls (do not change main results)
    ap.add_argument("--pca_points", type=int, default=1500, help="max points for PCA plot")
    ap.add_argument("--sgd_epochs", type=int, default=25, help="epochs for SGD hinge curve")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    train_pairs = load_rte_tsv(Path(args.train))
    test_pairs = load_rte_tsv(Path(args.test))

    # Keep your current dataset checks (as you requested: other logic unchanged)
    assert len(train_pairs) == 2490, f"train size mismatch: {len(train_pairs)}"
    assert len(test_pairs) == 277, f"test size mismatch: {len(test_pairs)}"

    # Dataset sanity print
    train_labels = Counter([p[3] for p in train_pairs])
    test_labels = Counter([p[3] for p in test_pairs])
    print(f"Train size: {len(train_pairs)}  label dist: {dict(train_labels)}")
    print(f"Test  size: {len(test_pairs)}  label dist: {dict(test_labels)}")

    kernels = ["rbf", "linear", "poly", "sigmoid"]
    results = []

    # -----------------------------------------------------
    # SBERT experiments (required)
    # -----------------------------------------------------
    print("\n=== SBERT + SVM ===")
    model = SentenceTransformer(args.sbert_model)
    best = {"accuracy": -1.0}

    # We'll store best-feature train vectors for PCA/SGD curve later
    best_X_train = None
    best_y_train = None
    best_feature_name = None

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
                # Save best feature vectors for extra plots
                best_X_train = X_train
                best_y_train = y_train
                best_feature_name = f"sbert-{mode}-{k}"

    # -----------------------------------------------------
    # BOW baseline experiments (required)
    # -----------------------------------------------------
    print("\n=== BOW baseline + SVM ===")
    ensure_nltk_resources()

    vocab, all_feat_lists = build_bow_dictionary(train_pairs, test_pairs)
    train_feats = all_feat_lists[: len(train_pairs)]
    test_feats = all_feat_lists[len(train_pairs):]
    print(f"[INFO] BOW vocab size: {len(vocab)}")

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
            # (Optional) For BOW best, PCA/SGD might be heavy; we keep SBERT best by default.

    # -----------------------------------------------------
    # Save results CSV (required)
    # -----------------------------------------------------
    df = pd.DataFrame(results)
    csv_path = out_dir / "results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # -----------------------------------------------------
    # Required plots
    # -----------------------------------------------------
    plot_sbert_heatmap(df, out_dir / "results_sbert_heatmap.png")
    plot_bow_bar(df, out_dir / "results_bow_bar.png")
    plot_confusion(
        best["y_true"], best["y_pred"],
        out_dir / "best_confusion_matrix.png",
        title=f"Best model: {best['feature']} {best['pair_mode']} {best['kernel']}  acc={best['accuracy']:.4f}"
    )

    # -----------------------------------------------------
    # Additional plots (do not change main results)
    # -----------------------------------------------------
    plot_confusion_normalized(
        best["y_true"], best["y_pred"],
        out_dir / "best_confusion_matrix_norm.png",
        title=f"Normalized CM: {best['feature']} {best['pair_mode']} {best['kernel']}"
    )

    plot_training_time_comparison(df, out_dir / "training_time_comparison.png")

    # PCA visualization for best feature vectors (usually SBERT best)
    if best_X_train is not None and best_y_train is not None and best_feature_name is not None:
        plot_best_feature_pca(
            best_feature_name,
            best_X_train, best_y_train,
            out_dir / "best_feature_pca.png",
            max_points=args.pca_points
        )

        # Training curve (loss) via SGD hinge (illustrative)
        run_sgd_hinge_training_curve(
            best_X_train, best_y_train,
            out_dir / "sgd_hinge_training_curve.png",
            epochs=args.sgd_epochs
        )

    print("Saved plots in:", out_dir.resolve())


if __name__ == "__main__":
    main()
