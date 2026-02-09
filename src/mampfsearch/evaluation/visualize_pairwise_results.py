import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd


# --- Global configuration (edit these) ------------------------------------
# Path to the pairwise results CSV/JSON (edit as needed)
INPUT_PATH = (
    "/home/fbuerckel/Mampfsearch/Mampfsearch/"
    "src/mampfsearch/evaluation/Results/18100a-lecture-21-multicam_360p_16_9/gpt-oss_20b_20260208_143908/pairwise_results.csv"
)
# Directory to save output images (None => save alongside input)
OUTPUT_DIR = (
    "/home/fbuerckel/Mampfsearch/Mampfsearch/"
    "src/mampfsearch/evaluation/Results/18100a-lecture-21-multicam_360p_16_9/gpt-oss_20b_20260208_143908/"
)
# Whether to show plots interactively
SHOW = False


def _load_pairwise_results(path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load pairwise results from either the streamed CSV or the JSON output."""

    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(str(input_path))

    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(input_path)
        meta: Dict[str, Any] = {"input": str(input_path), "format": "csv"}
        return df, meta

    if suffix == ".json":
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        matches = data.get("matches") or []
        df = pd.json_normalize(matches)
        meta = data.get("metadata") or {}
        meta.update({"input": str(input_path), "format": "json"})
        return df, meta

    raise ValueError(f"Unsupported input format: {suffix} (expected .csv or .json)")


def _infer_names(df: pd.DataFrame, meta: Dict[str, Any]) -> Tuple[str, str]:
    name_a = str(meta.get("name_a") or "A")
    name_b = str(meta.get("name_b") or "B")

    # If metadata isn't present (CSV mode), infer from winner column values.
    if name_a == "A" and name_b == "B" and "winner" in df.columns:
        winners = (
            df["winner"]
            .dropna()
            .astype(str)
            .str.strip()
            .loc[lambda s: (s != "") & (s.str.upper() != "TIE")]
        )
        uniq = sorted(set(winners.tolist()))
        if len(uniq) >= 2:
            name_a, name_b = uniq[0], uniq[1]
        elif len(uniq) == 1:
            name_a = uniq[0]

    return name_a, name_b


def _outcome_series(df: pd.DataFrame) -> pd.Series:
    """Return normalized outcome per row: winner name, 'TIE', or 'INVALID'."""

    winner = df.get("winner")
    error = df.get("error")

    winner_s = (
        winner.astype("string").fillna("").str.strip() if winner is not None else None
    )
    error_s = (
        error.astype("string").fillna("").str.strip() if error is not None else None
    )

    is_invalid = None
    if error_s is not None:
        is_invalid = error_s != ""
    elif winner_s is not None:
        is_invalid = winner_s == ""
    else:
        is_invalid = pd.Series([True] * len(df), index=df.index)

    outcome = pd.Series([""] * len(df), index=df.index, dtype="string")
    if winner_s is not None:
        # Preserve winner values (including 'TIE'); overwrite invalids later.
        outcome = winner_s.copy()

    outcome = outcome.where(~is_invalid, other="INVALID")
    outcome = outcome.replace({"": "INVALID"})
    outcome = outcome.apply(lambda x: x.upper() if x != "INVALID" else x)
    outcome = outcome.replace({"NONE": "INVALID", "NAN": "INVALID"})
    return outcome


def main() -> None:
    # Use top-level globals for configuration: INPUT_PATH, OUTPUT_DIR, SHOW
    if not INPUT_PATH:
        print(
            "Please set INPUT_PATH at the top of the file to the pairwise results path."
        )
        raise SystemExit(2)

    input_path_str = INPUT_PATH
    output_dir_str = OUTPUT_DIR
    show = SHOW

    df, meta = _load_pairwise_results(input_path_str)
    if df.empty:
        raise ValueError("Input contains no matches/rows.")

    input_path = Path(input_path_str)
    output_dir = Path(output_dir_str) if output_dir_str else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Lazy import so users still get useful 'no pandas' errors earlier.
    import matplotlib.pyplot as plt

    name_a, name_b = _infer_names(df, meta)

    # Normalize outcomes and compute counts.
    outcome = _outcome_series(df)

    # Resolve names in a case-insensitive way.
    # In some CSVs winner may already be resolved to 'multi'/'unstructured'.
    def _normalize_winner(x: str) -> str:
        if x == "INVALID":
            return x
        if x.upper() == "TIE":
            return "TIE"
        if x == name_a or x.upper() == name_a.upper():
            return name_a
        if x == name_b or x.upper() == name_b.upper():
            return name_b
        return x

    outcome_resolved = outcome.astype(str).apply(_normalize_winner)

    counts = outcome_resolved.value_counts(dropna=False)
    wins_a = int(counts.get(name_a, 0))
    wins_b = int(counts.get(name_b, 0))
    ties = int(counts.get("TIE", 0))
    invalid = int(counts.get("INVALID", 0))
    total = int(len(df))

    title = input_path.stem

    # ---- Plot 1: Overall outcomes ----
    fig1, ax1 = plt.subplots(figsize=(7, 4))
    categories = [name_a, name_b, "TIE", "INVALID"]
    values = [wins_a, wins_b, ties, invalid]
    ax1.bar(categories, values)
    ax1.set_title(f"Pairwise outcomes: {title}")
    ax1.set_ylabel("Count")
    ax1.set_ylim(0, max(values) + 1)

    for i, v in enumerate(values):
        ax1.text(i, v + 0.2, str(v), ha="center", va="bottom", fontsize=9)

    fig1.tight_layout()
    out1 = output_dir / f"{input_path.stem}_outcomes.svg"
    fig1.savefig(out1, dpi=200)

    # ---- Plot 2: Cumulative wins over pairs ----
    # Create a best-effort x-axis: 'pair' column if present, else row index.
    if "pair" in df.columns:
        x_fallback = pd.Series(range(1, total + 1), index=df.index, dtype="int")
        x = pd.to_numeric(df["pair"], errors="coerce").fillna(x_fallback)
    else:
        x = pd.RangeIndex(1, total + 1)

    is_a = outcome_resolved == name_a
    is_b = outcome_resolved == name_b
    is_tie = outcome_resolved == "TIE"
    is_invalid = outcome_resolved == "INVALID"

    cum_a = is_a.cumsum()
    cum_b = is_b.cumsum()
    cum_invalid = is_invalid.cumsum()

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(x, cum_a, label=f"{name_a} wins")
    ax2.plot(x, cum_b, label=f"{name_b} wins")
    ax2.plot(x, cum_invalid, label="Invalid", linestyle="--")
    ax2.set_title(f"Cumulative counts over pairs: {title}")
    ax2.set_xlabel("Pair")
    ax2.set_ylabel("Cumulative count")
    ax2.legend(loc="upper left")
    fig2.tight_layout()
    out2 = output_dir / f"{input_path.stem}_cumulative.png"
    fig2.savefig(out2, dpi=200)

    # ---- Plot 3: Top invalid reasons (if present) ----
    invalid_reason_col = None
    for candidate in ("invalid_reason", "invalid_reasoning", "invalid_reason_detail"):
        if candidate in df.columns:
            invalid_reason_col = candidate
            break

    out3: Optional[Path] = None
    if invalid_reason_col and invalid > 0:
        reasons = (
            df.loc[is_invalid, invalid_reason_col]
            .astype("string")
            .fillna("")
            .str.strip()
        )
        reasons = reasons.replace({"": "(empty)"})
        top = reasons.value_counts().head(10)

        fig3, ax3 = plt.subplots(figsize=(10, 4))
        ax3.barh(top.index[::-1], top.values[::-1])
        ax3.set_title(f"Top invalid reasons (n={invalid}): {title}")
        ax3.set_xlabel("Count")
        fig3.tight_layout()
        out3 = output_dir / f"{input_path.stem}_invalid_reasons.png"
        fig3.savefig(out3, dpi=200)

    # Console summary
    invalid_rate = (invalid / total) if total else 0.0
    print("=== Pairwise summary ===")
    print(f"input: {input_path}")
    print(f"rows: {total}")
    print(f"{name_a} wins: {wins_a}")
    print(f"{name_b} wins: {wins_b}")
    print(f"ties: {ties}")
    print(f"invalid: {invalid} ({invalid_rate:.1%})")
    print("saved:")
    print(f"  {out1}")
    print(f"  {out2}")
    if out3:
        print(f"  {out3}")

    if show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
