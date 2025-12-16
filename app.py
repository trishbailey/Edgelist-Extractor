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
    "Date",
    "Time",
    "Author Handle",
    "Opening Text",
    "Hit Sentence",
    "Links",
    "Source Domain",
    "Language",
    "Country",
    "Engagement",
    "Reach",
    "Estimated Views",
    "Hashtags",
    "Keyphrases",
    "Document Tags",
]

# Columns we need to build edges
REQUIRED_EDGE_COLUMNS = {"Author Handle"}
TEXT_EDGE_COLUMNS = {"Opening Text", "Hit Sentence"}
LINK_EDGE_COLUMNS = {"Links"}

HANDLE_REGEX = r"(@[A-Za-z0-9_]+)"
X_DOMAINS = {"x.com", "twitter.com", "mobile.twitter.com"}


# -----------------------------
# UTILITIES: LOADING MELTWATER EXPORTS (ROBUST + HEADER DETECTION)
# -----------------------------

def _decode_uploaded_file(file_like) -> str:
    try:
        file_like.seek(0)
    except Exception:
        pass

    raw = file_like.read()
    if isinstance(raw, bytes):
        return raw.decode("latin1", errors="replace")
    return str(raw)


def _find_header_line_index(lines) -> int | None:
    """
    Find the line index that looks like the real Meltwater header.
    We look for a line containing several expected column names.
    """
    # Be tolerant: check lowercase, and allow either comma or tab separated header lines.
    header_markers = [
        "date",
        "time",
        "author handle",
    ]
    # Search first ~50 lines (more than enough for Meltwater preamble)
    max_scan = min(len(lines), 80)
    for i in range(max_scan):
        hay = lines[i].strip().lower()
        if not hay:
            continue
        # Must contain all core markers
        if all(m in hay for m in header_markers):
            return i
    return None


def load_and_clean_meltwater_file(file_like) -> pd.DataFrame:
    """
    Robust reader that:
      1) decodes raw text
      2) finds the true header row by scanning for 'Date', 'Time', 'Author Handle'
      3) reads from that header row onward
      4) tries TSV first, then CSV
      5) treats quotes literally and skips malformed lines
    """
    text = _decode_uploaded_file(file_like)
    lines = text.splitlines()

    header_idx = _find_header_line_index(lines)
    if header_idx is None:
        # Could not find a header line; return empty with a helpful signal
        return pd.DataFrame()

    # Rebuild text starting at the header row
    trimmed_text = "\n".join(lines[header_idx:])

    def _read_with_sep(sep: str) -> pd.DataFrame:
        return pd.read_csv(
            StringIO(trimmed_text),
            sep=sep,
            header=0,
            engine="python",
            dtype=str,
            keep_default_na=False,
            on_bad_lines="skip",
            quoting=csv.QUOTE_NONE,
        )

    # Try TSV then CSV
    try:
        df = _read_with_sep("\t")
        # If it parsed as 1 column, delimiter probably isn't tab
        if df.shape[1] <= 1:
            df = _read_with_sep(",")
    except Exception:
        df = _read_with_sep(",")

    # Clean column names
    df.columns = df.columns.astype(str).str.strip()
    return df


def load_and_combine_meltwater_streamlit(uploaded_files) -> pd.DataFrame:
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
    s = series.fillna("").astype(str).str.strip()
    s = s.apply(lambda x: x if (x == "" or x.startswith("@")) else f"@{x}")
    return s


def get_working_table(df: pd.DataFrame) -> pd.DataFrame:
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
    text = series.fillna("").astype(str)
    extracted = text.str.extractall(HANDLE_REGEX)
    if extracted.empty:
        return pd.DataFrame(columns=["row_index", "handle"])

    extracted = extracted.reset_index()
    extracted = extracted.rename(columns={"level_0": "row_index", 0: "handle"})
    return extracted[["row_index", "handle"]]


def build_text_mention_edges(working: pd.DataFrame) -> pd.DataFrame:
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

    mask_valid = (extracted["source"] != "") & (extracted["target"] != "")
    mask_not_self = extracted["source"].str.lower() != extracted["target"].str.lower()

    edges = extracted.loc[mask_valid & mask_not_self, ["source", "target"]].copy()
    edges["edge_type"] = "text_mention"
    return edges.drop_duplicates().reset_index(drop=True)


# -----------------------------
# EDGE EXTRACTION: LINKS (X ACCOUNTS + DOMAINS)
# -----------------------------

def parse_links_cell(links_value) -> set:
    targets = set()
    if pd.isna(links_value):
        return targets

    text = str(links_value).strip()
    if not text:
        return targets

    parts = re.split(r"[\s,;]+", text)
    parts = [p for p in parts if p]

    for url in parts:
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
            segments = [s for s in path.split("/") if s]
            if segments:
                username = segments[0]
                if username.lower() not in {"intent", "share", "home", "i"}:
                    targets.add("@" + username)
        else:
            targets.add(domain)

    return targets


def build_link_edges(working: pd.DataFrame) -> pd.DataFrame:
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

    return pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)


# -----------------------------
# EDGE LIST ASSEMBLY
# -----------------------------

def build_edge_list(working: pd.DataFrame) -> pd.DataFrame:
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
# NODE TABLE
# -----------------------------

def aggregate_author_stats(working: pd.DataFrame) -> pd.DataFrame:
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

    stats = pd.concat([num_posts, total_eng, total_reach, est_views, language, country], axis=1).reset_index()
    return stats.rename(columns={"Author Handle": "id"})


def aggregate_domain_stats(working: pd.DataFrame) -> pd.DataFrame:
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

    stats = pd.concat([num_posts, total_eng, total_reach, est_views, language, country], axis=1).reset_index()
    return stats.rename(columns={"Source Domain": "id"})


def build_node_table(working: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    if edges.empty:
        return pd.DataFrame(columns=[
            "id", "node_type", "num_posts", "total_engagement",
            "total_reach", "estimated_views", "language", "country"
        ])

    node_ids = pd.unique(edges[["source", "target"]].values.ravel("K"))
    nodes = pd.DataFrame({"id": node_ids})
    nodes["node_type"] = nodes["id"].apply(lambda x: "handle" if str(x).startswith("@") else "domain")

    author_stats = aggregate_author_stats(working)
    domain_stats = aggregate_domain_stats(working)

    nodes = nodes.merge(author_stats, on="id", how="left")
    nodes = nodes.merge(domain_stats, on="id", how="left", suffixes=("", "_domain"))

    for col in ["num_posts", "total_engagement", "total_reach", "estimated_views", "language", "country"]:
        dom = f"{col}_domain"
        if dom in nodes.columns:
            nodes[col] = nodes[col].combine_first(nodes[dom])

    keep_cols = ["id", "node_type", "num_posts", "total_engagement", "total_reach", "estimated_views", "language", "country"]
    nodes = nodes[keep_cols]

    for col in ["num_posts", "total_engagement", "total_reach", "estimated_views"]:
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
- Build a directed edge list (mentions + links)
- Build a node table with handles & domains
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
                st.error(
                    "No data loaded. The loader could not find a valid header row.\n\n"
                    "This usually means the export's header line doesn't contain 'Date', 'Time', and 'Author Handle' "
                    "as expected, or the file is not a Meltwater export."
                )
                return

            st.write("Loaded rows:", len(combined))
            st.write("Loaded columns:", list(combined.columns))

            with st.spinner("Reducing to working schema..."):
                working = get_working_table(combined)

            st.write("Working rows:", len(working))
            st.write("Working columns:", list(working.columns))

            if working.empty or "Author Handle" not in working.columns:
                st.error(
                    "Working table is empty or missing 'Author Handle'.\n\n"
                    "Your file is loading, but its headers do not match the expected Meltwater schema. "
                    "Scroll up to see 'Loaded columns' and compare them to the required names."
                )
                return

            with st.spinner("Building edge list..."):
                edges = build_edge_list(working)

            if edges.empty:
                st.warning(
                    "No edges found. Confirm your export includes:\n"
                    "- 'Author Handle' AND\n"
                    "- either 'Opening Text'/'Hit Sentence' with @handles OR 'Links' with URLs."
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
