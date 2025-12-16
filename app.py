import re
import csv
from io import StringIO
from urllib.parse import urlparse

import pandas as pd
import streamlit as st

# -----------------------------
# CONFIG / CONSTANTS
# -----------------------------

WORKING_COLUMNS = [
    # For edges
    "Date",
    "Time",
    "Author Handle",
    "Opening Text",
    "Hit Sentence",
    "Links",
    "Source Domain",
    # For node enrichment
    "Language",
    "Country",          # optional; will be ignored if missing
    "Engagement",
    "Reach",
    "Estimated Views",
    "Hashtags",
    "Keyphrases",
    "Document Tags",
]

HANDLE_REGEX = r"(@[A-Za-z0-9_]+)"
X_DOMAINS = {"x.com", "twitter.com", "mobile.twitter.com"}


# -----------------------------
# UTILITIES: LOADING MELTWATER EXPORTS (ROBUST)
# -----------------------------

def load_and_clean_meltwater_file(file_like) -> pd.DataFrame:
    """
    Robust reader for Meltwater exports that often contain:
      - inconsistent field counts
      - broken/unbalanced quotes
      - mixed delimiters (TSV is common)
      - a junk first line before the real header

    Strategy:
      1) Read raw bytes/text and drop first line (junk/query line)
      2) Try TSV with QUOTE_NONE (treat quotes as literal)
      3) Fallback to comma CSV with same settings
      4) Force strings; never let pandas infer types
      5) Skip malformed rows deterministically
    """
    try:
        file_like.seek(0)
    except Exception:
        pass

    raw = file_like.read()
    if isinstance(raw, bytes):
        text = raw.decode("latin1", errors="replace")
    else:
        text = str(raw)

    # Drop the first line (Meltwater often includes a query/junk line)
    lines = text.splitlines()
    if len(lines) >= 2:
        text = "\n".join(lines[1:])

    def _read_with_sep(sep: str) -> pd.DataFrame:
        return pd.read_csv(
            StringIO(text),
            sep=sep,
            engine="python",
            dtype=str,
            keep_default_na=False,
            on_bad_lines="skip",
            quoting=csv.QUOTE_NONE,
        )

    # Try TSV first, then comma CSV
    try:
        df = _read_with_sep("\t")
        # If it parsed into a single column, delimiter likely wasn't tab
        if df.shape[1] <= 1:
            df = _read_with_sep(",")
    except Exception:
        df = _read_with_sep(",")

    df.columns = df.columns.astype(str).str.strip()
    return df


def load_and_combine_meltwater_streamlit(uploaded_files) -> pd.DataFrame:
    """
    Load multiple Meltwater files from Streamlit's uploader,
    clean each with the robust loader above,
    then concatenate into a single combined DataFrame.
    """
    dfs = []
    for f in uploaded_files:
        df_clean = load_and_clean_meltwater_file(f)
        if not df_clean.empty:
            dfs.append(df_clean)

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


# -----------------------------
# UTILITIES: NORMALIZATION + WORKING TABLE
# -----------------------------

def normalize_author_handle(series: pd.Series) -> pd.Series:
    """
    Normalize author handles:
      - force string
      - strip whitespace
      - ensure leading '@' if non-empty and missing
    """
    s = series.fillna("").astype(str).str.strip()
    s = s.apply(lambda x: x if (x == "" or x.startswith("@")) else f"@{x}")
    return s


def get_working_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce a raw Meltwater table to the working schema:
      - keep only columns that exist from WORKING_COLUMNS
      - normalize Author Handle and Source Domain
      - ensure text columns are non-null strings
    """
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    out.columns = out.columns.astype(str).str.strip()

    keep = [c for c in WORKING_COLUMNS if c in out.columns]
    out = out[keep].copy()

    if "Author Handle" in out.columns:
        out["Author Handle"] = normalize_author_handle(out["Author Handle"])

    if "Source Domain" in out.columns:
        out["Source Domain"] = (
            out["Source Domain"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
            .str.replace(r"^www\.", "", regex=True)
        )

    for col in ["Opening Text", "Hit Sentence", "Links"]:
        if col in out.columns:
            out[col] = out[col].fillna("").astype(str)

    return out


# -----------------------------
# EDGE EXTRACTION: TEXT MENTIONS
# -----------------------------

def extract_handles_from_text(series: pd.Series) -> pd.DataFrame:
    """
    Given a Series of text, return a DataFrame with:
      - row_index: original row index
      - handle: extracted handle (@something)
    Uses vectorized extractall for performance.
    """
    text = series.fillna("").astype(str)
    extracted = text.str.extractall(HANDLE_REGEX)
    if extracted.empty:
        return pd.DataFrame(columns=["row_index", "handle"])

    extracted = extracted.reset_index()
    extracted = extracted.rename(columns={"level_0": "row_index", 0: "handle"})
    return extracted[["row_index", "handle"]]


def build_text_mention_edges(working: pd.DataFrame) -> pd.DataFrame:
    """
    Build edges from text mentions:
      source = Author Handle
      target = @handle found in Opening Text or Hit Sentence
      edge_type = 'text_mention'
    """
    if "Author Handle" not in working.columns:
        return pd.DataFrame(columns=["source", "target", "edge_type"])

    text_cols = [c for c in ["Opening Text", "Hit Sentence"] if c in working.columns]
    if not text_cols:
        return pd.DataFrame(columns=["source", "target", "edge_type"])

    text_series = working[text_cols].fillna("").agg(" ".join, axis=1)
    extracted = extract_handles_from_text(text_series)
    if extracted.empty:
        return pd.DataFrame(columns=["source", "target", "edge_type"])

    src_map = working["Author Handle"]
    extracted["source"] = extracted["row_index"].map(src_map)
    extracted["target"] = extracted["handle"]

    extracted["source"] = extracted["source"].fillna("").astype(str).str.strip()
    extracted["target"] = extracted["target"].fillna("").astype(str).str.strip()

    # Drop self-loops and empty endpoints
    mask_valid = (extracted["source"] != "") & (extracted["target"] != "")
    mask_not_self = extracted["source"].str.lower() != extracted["target"].str.lower()
    edges = extracted.loc[mask_valid & mask_not_self, ["source", "target"]].copy()
    edges["edge_type"] = "text_mention"

    return edges.drop_duplicates().reset_index(drop=True)


# -----------------------------
# EDGE EXTRACTION: LINKS (X ACCOUNTS + DOMAINS)
# -----------------------------

def parse_links_cell(links_value) -> set:
    """
    Given the raw 'Links' cell from Meltwater, return a set of targets:
      - @username for X/Twitter URLs
      - domain.com for all other URLs
    """
    targets = set()
    if pd.isna(links_value):
        return targets

    text = str(links_value).strip()
    if not text:
        return targets

    # crude split on whitespace / commas / semicolons
    parts = re.split(r"[\s,;]+", text)
    parts = [p for p in parts if p]

    for url in parts:
        # If scheme missing, urlparse treats it as path and netloc becomes empty
        if not re.match(r"^https?://", url, flags=re.I):
            url = "https://" + url

        parsed = urlparse(url)
        domain = (parsed.netloc or "").lower()
        if not domain:
            continue

        if domain.startswith("www."):
            domain = domain[4:]

        path = (parsed.path or "").strip("/")
        if domain in X_DOMAINS:
            # X account is first path segment (ignore 'intent', 'share', etc.)
            segments = [s for s in path.split("/") if s]
            if segments:
                username = segments[0]
                # Skip obvious non-user paths
                if username.lower() not in {"intent", "share", "home", "i"}:
                    targets.add("@" + username)
        else:
            targets.add(domain)

    return targets


def build_link_edges(working: pd.DataFrame) -> pd.DataFrame:
    """
    Build edges from Links:
      source = Author Handle
      target = @username (for X) or domain.com (for other URLs)
      edge_type = 'x_link' or 'domain_link'
    """
    if "Author Handle" not in working.columns or "Links" not in working.columns:
        return pd.DataFrame(columns=["source", "target", "edge_type"])

    rows = []
    for _, row in working.iterrows():
        src = row.get("Author Handle", "")
        src = "" if pd.isna(src) else str(src).strip()
        if not src:
            continue

        targets = parse_links_cell(row.get("Links", ""))
        for tgt in targets:
            edge_type = "x_link" if str(tgt).startswith("@") else "domain_link"
            rows.append({"source": src, "target": tgt, "edge_type": edge_type})

    if not rows:
        return pd.DataFrame(columns=["source", "target", "edge_type"])

    edges = pd.DataFrame(rows)
    return edges.drop_duplicates().reset_index(drop=True)


# -----------------------------
# EDGE LIST ASSEMBLY
# -----------------------------

def build_edge_list(working: pd.DataFrame) -> pd.DataFrame:
    """
    Build the full directed edge list from the working table.
    Combines:
      - text mentions
      - link-derived edges
    Aggregates to compute weights.
    """
    text_edges = build_text_mention_edges(working)
    link_edges = build_link_edges(working)

    edges = pd.concat([text_edges, link_edges], ignore_index=True)
    if edges.empty:
        return pd.DataFrame(columns=["source", "target", "edge_type", "weight"])

    edges["weight"] = 1
    edges = (
        edges.groupby(["source", "target", "edge_type"], as_index=False)["weight"]
        .sum()
    )
    return edges


# -----------------------------
# NODE TABLE (SHARED HANDLE + DOMAIN SET)
# -----------------------------

def aggregate_author_stats(working: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate stats for author handles from the working table.
    """
    if "Author Handle" not in working.columns:
        return pd.DataFrame(columns=[
            "id", "num_posts", "total_engagement", "total_reach",
            "estimated_views", "language", "country"
        ])

    df = working.copy()
    df["Author Handle"] = normalize_author_handle(df["Author Handle"])

    for col in ["Engagement", "Reach", "Estimated Views"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = 0.0

    grouped = df.groupby("Author Handle")

    num_posts = grouped.size().rename("num_posts")
    total_eng = grouped["Engagement"].sum(min_count=1).rename("total_engagement")
    total_reach = grouped["Reach"].sum(min_count=1).rename("total_reach")
    est_views = grouped["Estimated Views"].sum(min_count=1).rename("estimated_views")

    def mode_or_nan(s):
        m = s.mode(dropna=True)
        return m.iloc[0] if not m.empty else pd.NA

    language = (
        grouped["Language"].apply(mode_or_nan).rename("language")
        if "Language" in df.columns else pd.Series(dtype="object")
    )
    country = (
        grouped["Country"].apply(mode_or_nan).rename("country")
        if "Country" in df.columns else pd.Series(dtype="object")
    )

    stats = pd.concat(
        [num_posts, total_eng, total_reach, est_views, language, country],
        axis=1
    ).reset_index()

    return stats.rename(columns={"Author Handle": "id"})


def aggregate_domain_stats(working: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate stats for domains using the Source Domain column.
    """
    if "Source Domain" not in working.columns:
        return pd.DataFrame(columns=[
            "id", "num_posts", "total_engagement", "total_reach",
            "estimated_views", "language", "country"
        ])

    df = working.copy()
    df["Source Domain"] = df["Source Domain"].fillna("").astype(str).str.strip().str.lower()
    df["Source Domain"] = df["Source Domain"].str.replace(r"^www\.", "", regex=True)

    for col in ["Engagement", "Reach", "Estimated Views"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = 0.0

    grouped = df.groupby("Source Domain")

    num_posts = grouped.size().rename("num_posts")
    total_eng = grouped["Engagement"].sum(min_count=1).rename("total_engagement")
    total_reach = grouped["Reach"].sum(min_count=1).rename("total_reach")
    est_views = grouped["Estimated Views"].sum(min_count=1).rename("estimated_views")

    def mode_or_nan(s):
        m = s.mode(dropna=True)
        return m.iloc[0] if not m.empty else pd.NA

    language = (
        grouped["Language"].apply(mode_or_nan).rename("language")
        if "Language" in df.columns else pd.Series(dtype="object")
    )
    country = (
        grouped["Country"].apply(mode_or_nan).rename("country")
        if "Country" in df.columns else pd.Series(dtype="object")
    )

    stats = pd.concat(
        [num_posts, total_eng, total_reach, est_views, language, country],
        axis=1
    ).reset_index()

    return stats.rename(columns={"Source Domain": "id"})


def build_node_table(working: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    """
    Build a shared node table for handles and domains.

    - Start from all unique node ids appearing in edges (sources + targets).
    - Classify node_type: 'handle' if startswith '@', otherwise 'domain'.
    - Join in aggregated stats from working table for authors and domains.
    """
    if edges.empty:
        return pd.DataFrame(columns=[
            "id", "node_type", "num_posts", "total_engagement",
            "total_reach", "estimated_views", "language", "country"
        ])

    node_ids = pd.unique(edges[["source", "target"]].values.ravel("K"))
    nodes = pd.DataFrame({"id": node_ids})

    nodes["node_type"] = nodes["id"].apply(
        lambda x: "handle" if str(x).startswith("@") else "domain"
    )

    author_stats = aggregate_author_stats(working)
    domain_stats = aggregate_domain_stats(working)

    nodes = nodes.merge(author_stats, on="id", how="left", suffixes=("", "_author"))
    nodes = nodes.merge(domain_stats, on="id", how="left", suffixes=("", "_domain"))

    # Coalesce author & domain stats
    for col in ["num_posts", "total_engagement", "total_reach",
                "estimated_views", "language", "country"]:
        col_author = col
        col_domain = f"{col}_domain"
        if col_author in nodes.columns and col_domain in nodes.columns:
            nodes[col] = nodes[col_author].combine_first(nodes[col_domain])
        elif col_domain in nodes.columns and col_author not in nodes.columns:
            nodes[col] = nodes[col_domain]

    keep_cols = [
        "id", "node_type", "num_posts", "total_engagement",
        "total_reach", "estimated_views", "language", "country"
    ]
    nodes = nodes[keep_cols]

    for col in ["num_posts", "total_engagement", "total_reach", "estimated_views"]:
        if col in nodes.columns:
            nodes[col] = pd.to_numeric(nodes[col], errors="coerce").fillna(0).astype(int)

    return nodes


# -----------------------------
# STREAMLIT APP
# -----------------------------

def main():
    st.title("Meltwater → Cosmograph Network Builder")
    st.write(
        """
Upload one or more Meltwater exports (CSV/TSV).  
This app will:
- Load + combine files
- Build a **directed edge list** (mentions + links)
- Build a **node table** with handles & domains in one shared node set
Then you can download both CSVs and load them into Cosmograph.
        """
    )

    uploaded_files = st.file_uploader(
        "Upload Meltwater export files",
        type=["csv", "tsv", "txt"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        st.success(f"{len(uploaded_files)} file(s) uploaded.")

    if st.button("Generate Network (Edges + Nodes)"):
        if not uploaded_files:
            st.warning("Please upload at least one file.")
            return

        try:
            with st.spinner("Loading and cleaning Meltwater data..."):
                combined = load_and_combine_meltwater_streamlit(uploaded_files)

            if combined.empty:
                st.warning("No data loaded. Check that your files are not empty.")
                return

            with st.spinner("Reducing to working schema..."):
                working = get_working_table(combined)

            st.write("Loaded rows:", len(combined))
            st.write("Loaded columns:", list(combined.columns))
            st.write("Working rows:", len(working))
            st.write("Working columns:", list(working.columns))

            with st.spinner("Building edge list..."):
                edges = build_edge_list(working)

            if edges.empty:
                st.warning(
                    "No edges found. Confirm your export includes 'Author Handle' and either:\n"
                    "- Opening Text / Hit Sentence with @handles, and/or\n"
                    "- Links with URLs"
                )
                return

            with st.spinner("Building node table..."):
                nodes = build_node_table(working, edges)

            st.success(f"Built {len(edges)} edges and {len(nodes)} nodes.")

            st.subheader("Edge list preview")
            st.dataframe(edges.head(100))

            st.subheader("Node table preview")
            st.dataframe(nodes.head(100))

            edges_csv = edges.to_csv(index=False).encode("utf-8")
            nodes_csv = nodes.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download edge list CSV",
                data=edges_csv,
                file_name="edges.csv",
                mime="text/csv",
            )
            st.download_button(
                label="Download node table CSV",
                data=nodes_csv,
                file_name="nodes.csv",
                mime="text/csv",
            )

        except Exception as e:
            st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
