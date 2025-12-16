import re
from urllib.parse import urlparse

import pandas as pd
import streamlit as st

# -----------------------------
# CONFIG / CONSTANTS
# -----------------------------

# Columns we actually care about for network construction
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

HANDLE_REGEX = r'(@[A-Za-z0-9_]+)'
X_DOMAINS = {"x.com", "twitter.com", "mobile.twitter.com"}


# -----------------------------
# UTILITIES: COLUMN NAME CLEANING
# -----------------------------

def clean_column_names(cols) -> list:
    """
    Clean weird BOM / encoding artifacts and whitespace from column names.
    Turns things like 'ÿþDate' into 'Date'.
    """
    cleaned = []
    for c in cols:
        c = str(c)
        # remove UTF BOM and 'ÿþ', strip spaces
        c_clean = c.replace("\ufeff", "").lstrip("ÿþ").strip()
        cleaned.append(c_clean)
    return cleaned


# -----------------------------
# UTILITIES: LOADING MELTWATER CSVs
# -----------------------------

def load_and_clean_meltwater_file(file_like) -> pd.DataFrame:
    """
    Read a single Meltwater CSV uploaded to Streamlit.

    Assumes:
      - Row 0 is a junk/query line
      - Row 1 is the real header row (Date, Time, Document ID, URL, ...)

    Uses:
      - latin1 encoding
      - python engine
      - sep=None so pandas can infer the delimiter (comma, semicolon, etc.)
    """
    df = pd.read_csv(
        file_like,
        encoding="latin1",
        engine="python",
        on_bad_lines="warn",   # or "skip" if you want to silently drop bad lines
        header=1,              # second row is header
        sep=None,              # let pandas sniff the delimiter
    )

    # Clean up column names (strip BOM / weird chars / spaces)
    df.columns = clean_column_names(df.columns)
    return df


def load_and_combine_meltwater_streamlit(uploaded_files) -> pd.DataFrame:
    """
    Load multiple Meltwater CSV files from Streamlit's uploader,
    clean each with the Meltwater-specific loader above,
    then concatenate into a single combined DataFrame.
    """
    dfs = []
    for f in uploaded_files:
        df_clean = load_and_clean_meltwater_file(f)
        dfs.append(df_clean)

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    return combined


# -----------------------------
# UTILITIES: NORMALIZATION / WORKING SCHEMA
# -----------------------------

def normalize_author_handle(series: pd.Series) -> pd.Series:
    """
    Ensure handles have a leading '@' and are trimmed of whitespace.
    """
    s = series.astype(str).str.strip()
    s = "@" + s.str.lstrip("@")
    return s


def get_working_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    From a full Meltwater table, select only the columns we care about.
    Columns missing from a particular file are simply ignored.
    """
    if df.empty:
        return df

    # Normalize column names again, just in case
    df = df.copy()
    df.columns = clean_column_names(df.columns)

    available_cols = [c for c in WORKING_COLUMNS if c in df.columns]
    working = df[available_cols].copy()

    if "Author Handle" in working.columns:
        working["Author Handle"] = normalize_author_handle(working["Author Handle"])

    return working


# -----------------------------
# EDGE EXTRACTION: TEXT MENTIONS
# -----------------------------

def extract_handles_from_text(series: pd.Series) -> pd.DataFrame:
    """
    Given a Series of text, return a DataFrame with:
    - row_index: original row index
    - handle: extracted handle (@something)
    Using vectorized extractall for performance.
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

    # Map row_index -> source handle
    src_map = working["Author Handle"]
    extracted["source"] = extracted["row_index"].map(src_map)
    extracted["target"] = extracted["handle"]

    # Drop self-loops
    mask_not_self = extracted["source"].str.lower() != extracted["target"].str.lower()
    edges = extracted.loc[mask_not_self, ["source", "target"]].copy()
    edges["edge_type"] = "text_mention"

    # Drop duplicates
    edges = edges.drop_duplicates().reset_index(drop=True)
    return edges


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
        parsed = urlparse(url)
        domain = parsed.netloc.lower()

        if not domain:
            continue

        if domain.startswith("www."):
            domain = domain[4:]

        path = parsed.path.strip("/")

        # X / Twitter account
        if domain in X_DOMAINS:
            segments = path.split("/")
            if segments:
                username = segments[0]
                if username:
                    targets.add("@" + username)
        else:
            # Treat as domain node
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
        src = row["Author Handle"]
        if pd.isna(src):
            continue
        src = str(src).strip()
        if not src:
            continue

        targets = parse_links_cell(row["Links"])
        for tgt in targets:
            if tgt.startswith("@"):
                edge_type = "x_link"
            else:
                edge_type = "domain_link"
            rows.append({"source": src, "target": tgt, "edge_type": edge_type})

    if not rows:
        return pd.DataFrame(columns=["source", "target", "edge_type"])

    edges = pd.DataFrame(rows)
    edges = edges.drop_duplicates().reset_index(drop=True)
    return edges


# -----------------------------
# EDGE LIST ASSEMBLY
# -----------------------------

def build_edge_list(working: pd.DataFrame) -> pd.DataFrame:
    """
    Build the full directed edge list from the WORKING table.
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
    Aggregate stats for author handles from the WORKING table.
    """
    if "Author Handle" not in working.columns:
        return pd.DataFrame(columns=[
            "id", "num_posts", "total_engagement", "total_reach",
            "estimated_views", "language", "country"
        ])

    df = working.copy()
    df["Author Handle"] = normalize_author_handle(df["Author Handle"])

    # Basic numeric fields
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

    stats = stats.rename(columns={"Author Handle": "id"})
    return stats


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
    df["Source Domain"] = df["Source Domain"].astype(str).str.strip().str.lower()
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

    stats = stats.rename(columns={"Source Domain": "id"})
    return stats


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

    # Basic type classification
    nodes["node_type"] = nodes["id"].apply(
        lambda x: "handle" if str(x).startswith("@") else "domain"
    )

    # Aggregated stats
    author_stats = aggregate_author_stats(working)
    domain_stats = aggregate_domain_stats(working)

    # Merge stats (handles first, then domains)
    nodes = nodes.merge(author_stats, on="id", how="left", suffixes=("", "_author"))
    nodes = nodes.merge(domain_stats, on="id", how="left", suffixes=("", "_domain"))

    # Coalesce author & domain stats where appropriate
    for col in ["num_posts", "total_engagement", "total_reach",
                "estimated_views", "language", "country"]:
        col_author = col
        col_domain = f"{col}_domain"
        if col_author in nodes.columns and col_domain in nodes.columns:
            nodes[col] = nodes[col_author].combine_first(nodes[col_domain])
        elif col_domain in nodes.columns and col_author not in nodes.columns:
            nodes[col] = nodes[col_domain]

    # Keep only the final columns of interest
    keep_cols = [
        "id", "node_type", "num_posts", "total_engagement",
        "total_reach", "estimated_views", "language", "country"
    ]
    nodes = nodes[keep_cols]

    # Fill numeric NaNs with 0 and cast to int where appropriate
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
        Upload one or more Meltwater CSV exports.  
        This app will:
        - Clean the Meltwater headers and combine files  
        - Build a **directed edge list** (mentions + links)  
        - Build a **node table** with handles & domains in one shared node set  
        You can then download both CSVs and load them into Cosmograph.
        """
    )

    uploaded_files = st.file_uploader(
        "Upload Meltwater CSV files",
        type=["csv"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        st.success(f"{len(uploaded_files)} file(s) uploaded.")

    if st.button("Generate Network (Edges + Nodes)"):
        if not uploaded_files:
            st.warning("Please upload at least one Meltwater CSV file.")
            return

        try:
            with st.spinner("Loading and cleaning Meltwater data..."):
                combined = load_and_combine_meltwater_streamlit(uploaded_files)

            if combined.empty:
                st.warning("No data loaded. Check that your CSV files are not empty.")
                return

            # Debug: show loaded columns
            st.write("Loaded rows:", combined.shape[0])
            st.write("Loaded columns:", list(combined.columns))

            with st.spinner("Reducing to working schema..."):
                working = get_working_table(combined)

            # Debug info: which columns did we actually keep?
            st.write("Working rows:", working.shape[0])
            st.write("Working columns:", list(working.columns))

            if working.empty or "Author Handle" not in working.columns:
                st.warning(
                    "Your export loaded, but required Meltwater columns were not found.\n"
                    "Make sure the file has 'Author Handle', and that the header row is the "
                    "standard Meltwater header."
                )
                return

            with st.spinner("Building edge list..."):
                edges = build_edge_list(working)

            if edges.empty:
                st.warning(
                    "No edges found. Check that your data has 'Author Handle', "
                    "text fields (Opening Text / Hit Sentence) with @handles, "
                    "and/or Links with URLs."
                )
                return

            with st.spinner("Building node table..."):
                nodes = build_node_table(working, edges)

            st.success(f"Built {len(edges)} edges and {len(nodes)} nodes.")

            st.subheader("Edge list preview")
            st.dataframe(edges.head(100))

            st.subheader("Node table preview")
            st.dataframe(nodes.head(100))

            # Download buttons
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
