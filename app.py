"""
Meltwater → Cosmograph Network Builder
=======================================
A Streamlit app that:
1. Loads multiple Meltwater CSV exports
2. Combines them into a single dataset
3. Extracts edges (mentions + links)
4. Builds a node table with aggregated stats
5. Exports CSVs for Cosmograph visualization
"""

import re
from io import StringIO
from urllib.parse import urlparse

import pandas as pd
import streamlit as st

# -----------------------------
# CONFIG / CONSTANTS
# -----------------------------

# Expected Meltwater column names (canonical)
EXPECTED_COLUMNS = [
    "Date", "Time", "Document ID", "URL", "Input Name", "Keywords",
    "Information Type", "Source Type", "Source Name", "Source Domain",
    "Content Type", "Author Name", "Author Handle", "Title", "Opening Text",
    "Hit Sentence", "Image", "Hashtags", "Links", "Country", "Region",
    "State", "City", "Language", "Sentiment", "Keyphrases", "Reach",
    "Global Reach", "National Reach", "Local Reach", "AVE", "Social Echo",
    "Editorial Echo", "Engagement", "Shares", "Quotes", "Likes", "Replies",
    "Reposts", "Comments", "Reactions", "Views", "Estimated Views",
    "Document Tags", "Custom Categories", "Custom Fields"
]

# Columns we need for network construction
WORKING_COLUMNS = [
    "Date", "Time", "Author Handle", "Opening Text", "Hit Sentence",
    "Links", "Source Domain", "Language", "Country", "Engagement",
    "Reach", "Estimated Views", "Hashtags", "Keyphrases", "Document Tags"
]

HANDLE_REGEX = r'(@[A-Za-z0-9_]+)'
X_DOMAINS = {"x.com", "twitter.com", "mobile.twitter.com"}


# -----------------------------
# UTILITIES: COLUMN NAME CLEANING
# -----------------------------

def clean_column_name(col: str) -> str:
    """Clean a single column name of BOM, encoding artifacts, and whitespace."""
    col = str(col)
    # Remove UTF-8 BOM
    col = col.replace("\ufeff", "")
    # Remove common encoding artifacts
    col = col.lstrip("ÿþ")
    # Strip whitespace
    col = col.strip()
    return col


def clean_column_names(cols) -> list:
    """Clean a list of column names."""
    return [clean_column_name(c) for c in cols]


def find_header_row(lines: list[str], max_rows: int = 5) -> int:
    """
    Find the header row by looking for known Meltwater column names.
    Returns the 0-based index of the header row.
    """
    # Key columns that must be present in the header
    key_columns = {"Date", "Author Handle", "URL", "Source Domain"}
    
    for i, line in enumerate(lines[:max_rows]):
        # Check if this line contains multiple key column names
        matches = sum(1 for col in key_columns if col in line)
        if matches >= 3:
            return i
    
    # Default: assume row 1 is header (row 0 is junk)
    return 1


# -----------------------------
# UTILITIES: LOADING MELTWATER CSVs
# -----------------------------

def load_meltwater_file(file_like) -> pd.DataFrame:
    """
    Robustly load a single Meltwater CSV file.
    
    Strategy:
    1. Read raw bytes and decode with multiple encoding attempts
    2. Find the actual header row by looking for known column names
    3. Parse with explicit delimiter (tab for UTF-16, comma for others)
    4. Clean column names
    """
    # Read raw bytes
    raw_bytes = file_like.read()
    file_like.seek(0)  # Reset for potential re-read
    
    # Detect UTF-16 by looking for BOM or null bytes pattern
    is_utf16 = False
    if raw_bytes.startswith(b'\xff\xfe') or raw_bytes.startswith(b'\xfe\xff'):
        is_utf16 = True
    elif b'\x00' in raw_bytes[:100]:
        # Null bytes in first 100 bytes strongly suggests UTF-16
        is_utf16 = True
    
    # Try multiple encodings, prioritizing UTF-16 if detected
    content = None
    if is_utf16:
        encodings = ['utf-16', 'utf-16-le', 'utf-16-be', 'utf-8-sig', 'utf-8', 'latin1', 'cp1252']
    else:
        encodings = ['utf-8-sig', 'utf-8', 'latin1', 'cp1252', 'utf-16', 'utf-16-le']
    
    for encoding in encodings:
        try:
            content = raw_bytes.decode(encoding)
            # Verify we got readable content (not garbage)
            if '\x00' not in content[:200]:  # No null bytes in decoded content
                break
            else:
                content = None  # Reset and try next encoding
        except (UnicodeDecodeError, LookupError):
            continue
    
    if content is None:
        # Last resort: decode with replacement
        content = raw_bytes.decode('utf-8', errors='replace')
    
    # Split into lines to find header row
    lines = content.split('\n')
    header_row_idx = find_header_row(lines)
    
    # Detect delimiter: tab or comma
    # Check the header row for tabs vs commas
    header_line = lines[header_row_idx] if header_row_idx < len(lines) else ""
    tab_count = header_line.count('\t')
    comma_count = header_line.count(',')
    delimiter = '\t' if tab_count > comma_count else ','
    
    # Parse CSV from string
    try:
        df = pd.read_csv(
            StringIO(content),
            sep=delimiter,              # Auto-detected delimiter
            header=header_row_idx,      # Dynamic header row detection
            on_bad_lines='skip',        # Skip malformed lines
            quoting=1,                  # QUOTE_ALL - handles embedded delimiters
            dtype=str,                  # Read everything as string initially
            low_memory=False
        )
    except Exception as e:
        st.warning(f"CSV parse error, trying alternative parser: {e}")
        # Fallback: try with python engine
        df = pd.read_csv(
            StringIO(content),
            sep=delimiter,
            header=header_row_idx,
            on_bad_lines='skip',
            engine='python',
            dtype=str
        )
    
    # Clean column names
    df.columns = clean_column_names(df.columns)
    
    return df


def validate_meltwater_columns(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """
    Validate that the DataFrame has expected Meltwater columns.
    Returns (is_valid, list_of_missing_required_columns).
    """
    required = {"Date", "Author Handle"}
    optional_but_useful = {"Opening Text", "Hit Sentence", "Links", "Source Domain"}
    
    actual_cols = set(df.columns)
    missing_required = required - actual_cols
    missing_useful = optional_but_useful - actual_cols
    
    is_valid = len(missing_required) == 0
    
    all_missing = list(missing_required) + list(missing_useful)
    return is_valid, all_missing


def load_and_combine_files(uploaded_files) -> pd.DataFrame:
    """
    Load multiple Meltwater CSV files and combine them.
    """
    dfs = []
    
    for f in uploaded_files:
        try:
            df = load_meltwater_file(f)
            
            # Validate columns
            is_valid, missing = validate_meltwater_columns(df)
            
            if not is_valid:
                st.warning(
                    f"⚠️ File '{f.name}' missing required columns: {missing}. "
                    f"Found columns: {list(df.columns)[:10]}..."
                )
                continue
            
            if missing:
                st.info(f"ℹ️ File '{f.name}' missing optional columns: {missing}")
            
            dfs.append(df)
            
        except Exception as e:
            st.error(f"❌ Failed to load '{f.name}': {e}")
            continue
    
    if not dfs:
        return pd.DataFrame()
    
    # Combine all dataframes
    combined = pd.concat(dfs, ignore_index=True)
    return combined


# -----------------------------
# UTILITIES: WORKING TABLE
# -----------------------------

def normalize_author_handle(series: pd.Series) -> pd.Series:
    """Ensure handles have a leading '@' and are trimmed."""
    s = series.astype(str).str.strip()
    # Remove any leading @ first, then add one
    s = "@" + s.str.lstrip("@")
    # Handle nan/@nan cases
    s = s.replace("@nan", pd.NA)
    s = s.replace("@", pd.NA)
    return s


def get_working_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract only the columns we need for network construction.
    """
    if df.empty:
        return df
    
    # Get available working columns
    available_cols = [c for c in WORKING_COLUMNS if c in df.columns]
    
    if not available_cols:
        return pd.DataFrame()
    
    working = df[available_cols].copy()
    
    # Normalize Author Handle
    if "Author Handle" in working.columns:
        working["Author Handle"] = normalize_author_handle(working["Author Handle"])
    
    return working


# -----------------------------
# EDGE EXTRACTION: TEXT MENTIONS
# -----------------------------

def extract_handles_from_text(series: pd.Series) -> pd.DataFrame:
    """
    Extract @handles from a text series.
    Returns DataFrame with row_index and handle columns.
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
    
    # Combine text columns
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
    
    # Remove rows with missing source
    extracted = extracted.dropna(subset=["source"])
    
    # Normalize target handles
    extracted["target"] = extracted["target"].str.lower()
    extracted["source_lower"] = extracted["source"].str.lower()
    
    # Drop self-loops
    mask_not_self = extracted["source_lower"] != extracted["target"]
    edges = extracted.loc[mask_not_self, ["source", "target"]].copy()
    edges["edge_type"] = "text_mention"
    
    # Normalize source to match target format (lowercase)
    edges["target"] = "@" + edges["target"].str.lstrip("@")
    
    # Drop duplicates
    edges = edges.drop_duplicates().reset_index(drop=True)
    return edges


# -----------------------------
# EDGE EXTRACTION: LINKS
# -----------------------------

def parse_links_cell(links_value) -> set:
    """
    Parse the 'Links' cell and extract targets:
      - @username for X/Twitter URLs
      - domain.com for all other URLs
    """
    targets = set()
    
    if pd.isna(links_value):
        return targets
    
    text = str(links_value).strip()
    if not text:
        return targets
    
    # Meltwater separates links with semicolons inside quotes
    # Remove surrounding quotes first
    text = text.strip('"')
    
    # Split on common delimiters
    parts = re.split(r'[;\s,"]+', text)
    parts = [p.strip() for p in parts if p.strip()]
    
    for url in parts:
        # Skip if doesn't look like a URL
        if not url.startswith(('http://', 'https://')):
            continue
        
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            
            if not domain:
                continue
            
            # Remove www. prefix
            if domain.startswith("www."):
                domain = domain[4:]
            
            path = parsed.path.strip("/")
            
            # X / Twitter account extraction
            if domain in X_DOMAINS:
                segments = path.split("/")
                if segments and segments[0]:
                    username = segments[0]
                    # Skip non-user paths
                    if username not in {"search", "hashtag", "i", "intent"}:
                        targets.add("@" + username.lower())
            else:
                # Add domain as node
                targets.add(domain)
        except Exception:
            continue
    
    return targets


def build_link_edges(working: pd.DataFrame) -> pd.DataFrame:
    """
    Build edges from Links column:
      source = Author Handle
      target = @username (for X) or domain.com (for other URLs)
      edge_type = 'x_link' or 'domain_link'
    """
    if "Author Handle" not in working.columns or "Links" not in working.columns:
        return pd.DataFrame(columns=["source", "target", "edge_type"])
    
    rows = []
    
    for idx, row in working.iterrows():
        src = row["Author Handle"]
        if pd.isna(src):
            continue
        src = str(src).strip()
        if not src or src == "@nan":
            continue
        
        targets = parse_links_cell(row.get("Links"))
        src_lower = src.lower()
        
        for tgt in targets:
            # Skip self-loops
            if tgt.startswith("@") and tgt.lower() == src_lower:
                continue
            
            edge_type = "x_link" if tgt.startswith("@") else "domain_link"
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
    Build the complete directed edge list.
    Combines text mentions and link-derived edges.
    """
    text_edges = build_text_mention_edges(working)
    link_edges = build_link_edges(working)
    
    edges = pd.concat([text_edges, link_edges], ignore_index=True)
    
    if edges.empty:
        return pd.DataFrame(columns=["source", "target", "edge_type", "weight"])
    
    # Aggregate and compute weights
    edges["weight"] = 1
    edges = edges.groupby(["source", "target", "edge_type"], as_index=False)["weight"].sum()
    
    return edges


# -----------------------------
# NODE TABLE CONSTRUCTION
# -----------------------------

def aggregate_author_stats(working: pd.DataFrame) -> pd.DataFrame:
    """Aggregate statistics for author handles."""
    if "Author Handle" not in working.columns:
        return pd.DataFrame(columns=[
            "id", "num_posts", "total_engagement", "total_reach",
            "estimated_views", "language", "country"
        ])
    
    df = working.copy()
    
    # Convert numeric columns
    for col in ["Engagement", "Reach", "Estimated Views"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = 0.0
    
    # Drop rows with invalid Author Handle
    df = df[df["Author Handle"].notna()]
    df = df[df["Author Handle"] != "@nan"]
    df = df[df["Author Handle"] != "@"]
    
    if df.empty:
        return pd.DataFrame(columns=[
            "id", "num_posts", "total_engagement", "total_reach",
            "estimated_views", "language", "country"
        ])
    
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
    """Aggregate statistics for source domains."""
    if "Source Domain" not in working.columns:
        return pd.DataFrame(columns=[
            "id", "num_posts", "total_engagement", "total_reach",
            "estimated_views", "language", "country"
        ])
    
    df = working.copy()
    df["Source Domain"] = df["Source Domain"].astype(str).str.strip().str.lower()
    df["Source Domain"] = df["Source Domain"].str.replace(r"^www\.", "", regex=True)
    
    # Drop invalid domains
    df = df[df["Source Domain"] != "nan"]
    df = df[df["Source Domain"] != ""]
    
    for col in ["Engagement", "Reach", "Estimated Views"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = 0.0
    
    if df.empty:
        return pd.DataFrame(columns=[
            "id", "num_posts", "total_engagement", "total_reach",
            "estimated_views", "language", "country"
        ])
    
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
    Build a unified node table for both handles and domains.
    """
    if edges.empty:
        return pd.DataFrame(columns=[
            "id", "node_type", "num_posts", "total_engagement",
            "total_reach", "estimated_views", "language", "country"
        ])
    
    # Get all unique node IDs from edges
    node_ids = pd.unique(edges[["source", "target"]].values.ravel("K"))
    nodes = pd.DataFrame({"id": node_ids})
    
    # Classify node type
    nodes["node_type"] = nodes["id"].apply(
        lambda x: "handle" if str(x).startswith("@") else "domain"
    )
    
    # Get aggregated stats
    author_stats = aggregate_author_stats(working)
    domain_stats = aggregate_domain_stats(working)
    
    # Merge author stats
    nodes = nodes.merge(author_stats, on="id", how="left", suffixes=("", "_author"))
    
    # Merge domain stats
    nodes = nodes.merge(domain_stats, on="id", how="left", suffixes=("", "_domain"))
    
    # Coalesce author & domain stats
    for col in ["num_posts", "total_engagement", "total_reach",
                "estimated_views", "language", "country"]:
        col_domain = f"{col}_domain"
        if col in nodes.columns and col_domain in nodes.columns:
            nodes[col] = nodes[col].combine_first(nodes[col_domain])
        elif col_domain in nodes.columns and col not in nodes.columns:
            nodes[col] = nodes[col_domain]
    
    # Keep only final columns
    keep_cols = [
        "id", "node_type", "num_posts", "total_engagement",
        "total_reach", "estimated_views", "language", "country"
    ]
    
    # Ensure all columns exist
    for col in keep_cols:
        if col not in nodes.columns:
            nodes[col] = pd.NA
    
    nodes = nodes[keep_cols]
    
    # Fill numeric NaNs with 0
    for col in ["num_posts", "total_engagement", "total_reach", "estimated_views"]:
        nodes[col] = pd.to_numeric(nodes[col], errors="coerce").fillna(0).astype(int)
    
    return nodes


# -----------------------------
# STREAMLIT APP
# -----------------------------

def main():
    st.set_page_config(
        page_title="Meltwater → Network Builder",
        page_icon="🌐",
        layout="wide"
    )
    
    st.title("🌐 Meltwater → Cosmograph Network Builder")
    st.write("""
    Upload one or more Meltwater CSV exports. This app will:
    - **Combine** multiple files into a single dataset
    - **Extract edges** from mentions and links
    - **Build a node table** with handles and domains
    - **Export CSVs** for Cosmograph visualization
    """)
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload Meltwater CSV files",
        type=["csv"],
        accept_multiple_files=True,
        help="Select one or more CSV files exported from Meltwater"
    )
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} file(s) uploaded")
    
    # Generate button
    if st.button("🚀 Generate Network", type="primary"):
        if not uploaded_files:
            st.warning("Please upload at least one Meltwater CSV file.")
            return
        
        try:
            # Step 1: Load and combine
            with st.spinner("Loading and combining Meltwater data..."):
                combined = load_and_combine_files(uploaded_files)
            
            if combined.empty:
                st.error("No valid data loaded. Check that your files have the expected Meltwater format.")
                return
            
            st.info(f"📊 Loaded **{len(combined):,}** rows with **{len(combined.columns)}** columns")
            
            # Debug: show columns
            with st.expander("View loaded columns"):
                st.write(list(combined.columns))
            
            # Step 2: Get working table
            with st.spinner("Extracting working columns..."):
                working = get_working_table(combined)
            
            if working.empty:
                st.error("Could not extract working columns. Check your CSV format.")
                return
            
            if "Author Handle" not in working.columns:
                st.error(
                    "Required column 'Author Handle' not found. "
                    "Make sure your Meltwater export includes this column."
                )
                st.write("Available columns:", list(combined.columns)[:20])
                return
            
            # Debug: show working columns
            with st.expander("View working columns"):
                st.write(f"Working rows: {len(working):,}")
                st.write(f"Working columns: {list(working.columns)}")
            
            # Step 3: Build edges
            with st.spinner("Building edge list..."):
                edges = build_edge_list(working)
            
            if edges.empty:
                st.warning(
                    "No edges found. Check that your data has:\n"
                    "- Author Handle column\n"
                    "- Text fields (Opening Text / Hit Sentence) with @mentions\n"
                    "- Links column with URLs"
                )
                return
            
            # Step 4: Build nodes
            with st.spinner("Building node table..."):
                nodes = build_node_table(working, edges)
            
            # Success!
            st.success(f"✅ Built **{len(edges):,}** edges and **{len(nodes):,}** nodes")
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Edge List Preview")
                st.dataframe(edges.head(100), use_container_width=True)
                
                # Edge type breakdown
                st.write("**Edge Types:**")
                st.write(edges["edge_type"].value_counts().to_dict())
            
            with col2:
                st.subheader("Node Table Preview")
                st.dataframe(nodes.head(100), use_container_width=True)
                
                # Node type breakdown
                st.write("**Node Types:**")
                st.write(nodes["node_type"].value_counts().to_dict())
            
            # Download buttons
            st.subheader("📥 Download")
            
            col1, col2 = st.columns(2)
            
            with col1:
                edges_csv = edges.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download Edge List CSV",
                    data=edges_csv,
                    file_name="edges.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                nodes_csv = nodes.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download Node Table CSV",
                    data=nodes_csv,
                    file_name="nodes.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            # Optional: Combined data download
            with st.expander("Download combined raw data"):
                combined_csv = combined.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download Combined CSV",
                    data=combined_csv,
                    file_name="meltwater_combined.csv",
                    mime="text/csv"
                )
        
        except Exception as e:
            st.error(f"An error occurred: {e}")
            import traceback
            st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
