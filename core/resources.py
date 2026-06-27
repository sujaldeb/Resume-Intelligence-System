"""
core/resources.py

Resource loading layer for the Resume Intelligence Platform.
All runtime artifacts and the sentence-transformer model are loaded here.
No other module touches the filesystem for artifacts.

All downstream pipeline functions receive the resources dict as an argument.
This module has no dependency on Streamlit beyond the cache decorator,
which degrades gracefully when Streamlit is not present.
"""

import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import (
    ARTIFACT_PATHS,
    BENCHMARK_SCORE_METRICS,
    EMBEDDING_DIM,
    EMBEDDING_MODEL_NAME,
    REQUIRED_EMBEDDING_KEYS,
    REQUIRED_ESCO_COLS,
    REQUIRED_JOB_DESC_COLS,
    REQUIRED_SCORING_KEYS,
    SUPPORTED_DOMAINS,
)

# Streamlit import — optional so this module works outside Streamlit contexts

try:
    import streamlit as st
    _cache_resource = st.cache_resource
except ImportError:
    # identity decorator for non-Streamlit execution (tests, notebooks, scripts)
    def _cache_resource(fn=None, **kwargs):
        if fn is not None:
            return fn
        def decorator(f):
            return f
        return decorator


# Private loaders

def _load_pickle(path: Path) -> object:
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


# Validators

def _validate_embedding(artifact: dict) -> None:
    missing = REQUIRED_EMBEDDING_KEYS - set(artifact.keys())
    if missing:
        raise RuntimeError(
            f"runtime_embedding_artifacts.pkl missing keys: {missing}"
        )

    emb = artifact["job_embeddings"]
    if not isinstance(emb, np.ndarray) or emb.ndim != 2 or emb.shape[1] != EMBEDDING_DIM:
        raise RuntimeError(
            f"job_embeddings must be a 2D ndarray with {EMBEDDING_DIM} columns. "
            f"Got shape {getattr(emb, 'shape', type(emb))}."
        )

    missing_domains = SUPPORTED_DOMAINS - set(artifact["domain_job_indices"].keys())
    if missing_domains:
        raise RuntimeError(
            f"runtime_embedding_artifacts.pkl domain_job_indices missing "
            f"supported domains: {missing_domains}"
        )

    rescaling = artifact["similarity_rescaling"]
    for key in ("pop_min", "pop_max"):
        if key not in rescaling:
            raise RuntimeError(
                f"runtime_embedding_artifacts.pkl similarity_rescaling missing key: {key}"
            )
    if rescaling["pop_min"] >= rescaling["pop_max"]:
        raise RuntimeError(
            "runtime_embedding_artifacts.pkl similarity_rescaling: "
            "pop_min must be less than pop_max"
        )


def _validate_benchmark(artifact: dict) -> None:
    missing = SUPPORTED_DOMAINS - set(artifact.keys())
    if missing:
        raise RuntimeError(
            f"runtime_benchmark_lookup.json missing supported domains: {missing}"
        )

    for domain, data in artifact.items():
        if not isinstance(data, dict):
            raise RuntimeError(
                f"runtime_benchmark_lookup.json domain '{domain}' must be a dict"
            )
        if "score_arrays" not in data:
            raise RuntimeError(
                f"runtime_benchmark_lookup.json domain '{domain}' missing key: score_arrays"
            )
        for metric in BENCHMARK_SCORE_METRICS:
            if metric not in data["score_arrays"]:
                raise RuntimeError(
                    f"runtime_benchmark_lookup.json domain '{domain}' missing "
                    f"metric '{metric}' in score_arrays"
                )
            arr = data["score_arrays"][metric]
            if not isinstance(arr, list) or len(arr) == 0:
                raise RuntimeError(
                    f"runtime_benchmark_lookup.json domain '{domain}' metric "
                    f"'{metric}' must be a non-empty list"
                )


def _validate_scoring(artifact: dict) -> None:
    missing = REQUIRED_SCORING_KEYS - set(artifact.keys())
    if missing:
        raise RuntimeError(
            f"ats_scoring_artifacts.json missing keys: {missing}"
        )

    if not isinstance(artifact["flag_cols"], list) or len(artifact["flag_cols"]) == 0:
        raise RuntimeError(
            "ats_scoring_artifacts.json flag_cols must be a non-empty list"
        )
    if not isinstance(artifact["flag_weights"], dict) or len(artifact["flag_weights"]) == 0:
        raise RuntimeError(
            "ats_scoring_artifacts.json flag_weights must be a non-empty dict"
        )

    flag_cols_set    = set(artifact["flag_cols"])
    flag_weight_keys = set(artifact["flag_weights"].keys())
    if flag_cols_set != flag_weight_keys:
        raise RuntimeError(
            "ats_scoring_artifacts.json flag_cols and flag_weights keys do not match. "
            f"In flag_cols only: {flag_cols_set - flag_weight_keys}. "
            f"In flag_weights only: {flag_weight_keys - flag_cols_set}."
        )


def _validate_job_desc(df: pd.DataFrame) -> None:
    missing = REQUIRED_JOB_DESC_COLS - set(df.columns)
    if missing:
        raise RuntimeError(
            f"curated_job_descriptions.csv missing columns: {missing}"
        )
    if df.empty:
        raise RuntimeError("curated_job_descriptions.csv contains no rows")

    if df["job_id"].isnull().any():
        raise RuntimeError(
            "curated_job_descriptions.csv contains null values in job_id"
        )
    n_dupes = df["job_id"].duplicated().sum()
    if n_dupes > 0:
        raise RuntimeError(
            f"curated_job_descriptions.csv contains {n_dupes} duplicate job_id values"
        )

    missing_domains = SUPPORTED_DOMAINS - set(df["domain"].dropna().unique())
    if missing_domains:
        raise RuntimeError(
            f"curated_job_descriptions.csv missing supported domains: {missing_domains}"
        )


def _validate_esco(df: pd.DataFrame) -> None:
    missing = REQUIRED_ESCO_COLS - set(df.columns)
    if missing:
        raise RuntimeError(
            f"esco_skill_mapping.csv missing columns: {missing}"
        )
    if df.empty:
        raise RuntimeError("esco_skill_mapping.csv contains no rows")

    if df["canonical_token"].isnull().any():
        raise RuntimeError(
            "esco_skill_mapping.csv contains null values in canonical_token"
        )
    if df["canonical_token"].duplicated().any():
        n_dupes = df["canonical_token"].duplicated().sum()
        raise RuntimeError(
            f"esco_skill_mapping.csv contains {n_dupes} duplicate canonical_token values"
        )

    valid_categories = {"esco_mapped", "platform_tool", "no_esco_match"}
    unknown = set(df["token_category"].unique()) - valid_categories
    if unknown:
        raise RuntimeError(
            f"esco_skill_mapping.csv contains unknown token_category values: {unknown}"
        )


# Cross-artifact consistency check

def _validate_domain_consistency(embedding: dict, benchmark: dict, job_desc: pd.DataFrame) -> None:
    """
    Confirms that the three artifacts agree on which domains are present.
    Catches cases where artifacts were generated at different pipeline runs.
    """
    embedding_domains = set(embedding["domain_job_indices"].keys())
    benchmark_domains = set(benchmark.keys())
    job_desc_domains  = set(job_desc["domain"].dropna().unique())

    if embedding_domains != benchmark_domains:
        raise RuntimeError(
            f"Domain mismatch between embedding artifact and benchmark lookup. "
            f"Embedding only: {embedding_domains - benchmark_domains}. "
            f"Benchmark only: {benchmark_domains - embedding_domains}."
        )
    if not SUPPORTED_DOMAINS.issubset(job_desc_domains):
        raise RuntimeError(
            f"curated_job_descriptions.csv does not cover all domains present "
            f"in embedding artifact. Missing: {embedding_domains - job_desc_domains}."
        )


# Derived structure builders

def _build_normalization_lookup(esco_df: pd.DataFrame) -> dict:
    """
    Maps each canonical token to its normalized label.
    esco_mapped tokens use the ESCO preferred label.
    All other tokens retain their canonical form.
    """
    return {
        row["canonical_token"]: (
            row["esco_preferred_label"]
            if row["token_category"] == "esco_mapped"
            else row["canonical_token"]
        )
        for _, row in esco_df.iterrows()
    }


def _build_job_desc_index(job_desc: pd.DataFrame) -> pd.DataFrame:
    """
    Returns job_desc indexed by job_id for O(1) lookup during semantic matching.
    job_id is coerced to int to match the integer keys in job_id_to_idx.
    """
    df = job_desc.copy()
    df["job_id"] = df["job_id"].astype(int)
    return df.set_index("job_id")


# Public interface

@_cache_resource(show_spinner=False)
def load_resources() -> dict:
    """
    Loads and validates all runtime artifacts and the sentence-transformer model.
    Cached for the lifetime of the Streamlit instance.
    All downstream pipeline functions receive this dict as an argument.

    Raises RuntimeError with a specific message on any startup failure.
    Returns a resources dict on success.
    """
    # importing here avoids a hard dependency at module import time
    from sentence_transformers import SentenceTransformer

    resources  = {}
    load_times = {}

    # fail early if any file is missing before attempting any load
    missing_files = [
        path.name
        for path in ARTIFACT_PATHS.values()
        if not path.exists()
    ]
    if missing_files:
        raise RuntimeError(
            "Application startup failed. Missing artifact files:\n"
            + "\n".join(f"  - {f}" for f in missing_files)
        )

    loaders = {
        "embedding": _load_pickle,
        "benchmark": _load_json,
        "scoring":   _load_json,
        "job_desc":  _load_csv,
        "esco":      _load_csv,
    }

    validators = {
        "embedding": _validate_embedding,
        "benchmark": _validate_benchmark,
        "scoring":   _validate_scoring,
        "job_desc":  _validate_job_desc,
        "esco":      _validate_esco,
    }

    # load and validate each artifact independently
    for key in ("embedding", "benchmark", "scoring", "job_desc", "esco"):
        t0       = time.perf_counter()
        artifact = loaders[key](ARTIFACT_PATHS[key])
        validators[key](artifact)
        resources[key]  = artifact
        load_times[key] = round(time.perf_counter() - t0, 3)

    # cross-artifact domain consistency
    _validate_domain_consistency(
        resources["embedding"],
        resources["benchmark"],
        resources["job_desc"],
    )

    # model — loaded last, heaviest operation
    t0 = time.perf_counter()
    resources["model"] = SentenceTransformer(EMBEDDING_MODEL_NAME)
    load_times["model"] = round(time.perf_counter() - t0, 3)

    # derived structures — built once, reused across all pipeline calls
    resources["normalization_lookup"] = _build_normalization_lookup(resources["esco"])
    resources["canonical_tokens"]     = resources["esco"]["canonical_token"].tolist()
    resources["job_desc_index"]        = _build_job_desc_index(resources["job_desc"])

    # inverted job index: embedding row index -> job_id
    # pre-computed here so run_semantic_matching does not rebuild it on every call
    resources["idx_to_job_id"] = {
        idx: job_id
        for job_id, idx in resources["embedding"]["job_id_to_idx"].items()
    }

    resources["_load_times"] = load_times
    resources["_healthy"]    = True

    return resources


def get_load_report(resources: dict) -> str:
    """Returns a formatted startup diagnostics string for logging."""
    lines = ["Resource load report:"]
    for key, t in resources["_load_times"].items():
        lines.append(f"  {key:<12} {t:.3f}s")
    total = sum(resources["_load_times"].values())
    lines.append(f"  {'total':<12} {round(total, 3)}s")
    lines.append(f"  {'status':<12} healthy")
    return "\n".join(lines)