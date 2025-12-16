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
    "Country",          # optional; ignored if missing
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
# UTILITIES: FILE DECODE + HEADER-PREAMBLE HANDLING
# -----------------------------

def _decode_uploaded_file(file_like) -> str:
    """Decode Streamlit UploadedFile into text without hard failures."""
    try:
        file_like.seek(0)
    except Exception:
        pass

    raw = file_like.read()
    if isinstance(raw, bytes):
        return raw.decode("latin1", errors="replace")
    return str(raw)


def _strip_leading_empty_lines(lines: list[str]) -> list[str]:
    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1
    return lines[i:]


def _drop_meltwater_query_line(lines: list[str]) -> list[str]:
    """
    Your Meltwater export pattern:
      - line 1: query/title only (1 field; few or no delimiters)
      - line 2: real header (46 fields; many delimiters)
    We drop the first line if it doesn't look like a header row.
    """
    lines = _strip_leading_empty_lines(lines)
    if len(lines) < 2:
        return lines

    first = lines[0]
    second = lines[1]

    # Count delimiters to infer if line is a real table row/header or a single-cell title.
    # Header line will have many commas or tabs.
    first_commas = first.count(",")
    first_tabs = first.count("\t")
    second_commas = second.count(",")
    second_tabs = second.count("\t")

    first_delims = max(first_commas, first_tabs)
    second_delims = max(second_commas, second_tabs)

    # If first line has very few delimiters but second has many, drop the first.
    if first_delims <= 2 and second_delims >= 10:
        return lines[1:]

    return lines


def _choose_sep_from_header(header_line: str) -> str:
    """Pick delimiter based on header line."""
    if header_line.count("\t") > header_line.count(","):
        return "\t"
    return ","


# -----------------------------
# UTILITIES: LOADING MELTWATER EXPORTS (ROBUST)
# -----------------------------

def load_and_clean_meltwater_file(file_like) -> pd.DataFrame:
    """
    Robust loader for your Meltwater export:
      - Drops the first single-cell query/title line
      - Uses the true header row
      - Tries TSV/CSV
      - Skips malformed rows and treats quotes as literal
    """
    text = _decode_uploaded_file(file_like)
    lines = text.splitlines()
    lines = _drop_meltwater_query_line(lines)

    if not lines:
        return pd.DataFrame()

    # Now line 0 should be the real header row
    header_line = lines[0]
    body_text = "\n".join(lines)

    # Choose primary separator based on header line
    primary_sep = _choose_sep_from_header(header_line)
    fallback_sep = "," if primary_sep == "\t" else "\t"

    def _read(sep: str) -> pd.DataFrame:
        return pd.read_csv(
            StringIO(body_text),
            sep=sep,
            header=0,
            engine="python",
            dtype=str,
            keep_default_na=False,
            on_bad_lines="skip",
            quoting=csv.QUOTE_NONE,
        )

    try:
        df = _read(primary_sep)
        # If it collapsed into 1 column, the separator guess was wrong
        if df.shape[1] <= 1:
            df = _read(fallback_sep)
    except Exception:
        try:
            df = _read(fallback_sep)
        except Exception:
            return pd.DataFrame()

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
    # Meltwater sometimes exports without '@' (rare), so enforce it safely
    s = s.apply(lambda x: x if (x == "" or x.startswith("@")) else f"@{x}")
    return s


def get_working_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce raw Meltwater table to the working schema your edge/node code expects.
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
        # Add scheme if missing so urlparse can populate netloc
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
    edges = edges.groupby(["source", "target", "edge_type"], as_index=False)["weight"].sum()
    return edges


# -----------------------------
# NODE TABLE (SHARED HANDLE + DOMAIN SET)
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

    language = grouped["Language"].apply(mode_or_nan).rename("language") if "Language" in df.columns else pd.Series(dtype="object")
    country = grouped["Country"].apply(mode_or_nan).rename("country") if "Country" in df.columns else pd.Series(dtype="object")

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

    language = grouped["Language"].apply(mode_or_nan).rename("language") if "Language" in df.columns else pd.Series(dtype="object")
    country = grouped["Country"].apply(mode_or_nan).rename("country") if "Country" in df.columns else pd.Series(dtype="object")

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
                st.error(
                    "No data loaded.\n\n"
                    "This usually means the file isn't being parsed as a delimited table, "
                    "or the header/preamble format differs from the standard Meltwater export."
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
                    "Your export loaded, but required Meltwater columns were not found."
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
