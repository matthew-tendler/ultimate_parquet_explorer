import streamlit as st
import pandas as pd
import pyarrow.parquet as pq
from io import BytesIO
import plotly.express as px
from st_aggrid import AgGrid, GridOptionsBuilder


st.set_page_config(
    page_title="Ultimate Parquet Data Explorer",
    layout="wide",
)

st.title("🧪 Ultimate Parquet Data Explorer")
st.write(
    "Upload a Parquet file, then explore, filter, visualize, and analyze it "
    "like a biostatistician or stat programmer."
)


# ---------------------------
# Sidebar - file upload and options
# ---------------------------

st.sidebar.header("Data Input")

uploaded_file = st.sidebar.file_uploader(
    "Upload a Parquet file",
    type=["parquet"],
)

sample_size = st.sidebar.number_input(
    "Sample size for exploration",
    min_value=100,
    max_value=100_000,
    value=5_000,
    step=500,
    help="The app will use up to this many rows for previews and visualizations.",
)

use_full_for_heavy_ops = st.sidebar.checkbox(
    "Use full dataset for heavy operations (summary stats, crosstabs)",
    value=False,
    help=(
        "If checked, some operations will run on the full dataset instead of the sample. "
        "This may be slower for large files."
    ),
)


# ---------------------------
# Helper functions
# ---------------------------

@st.cache_data(show_spinner=True)
def load_parquet_from_bytes(file_bytes: bytes) -> pd.DataFrame:
    import pyarrow.parquet as pq
    import pandas as pd
    import io

    # Read parquet with PyArrow
    table = pq.read_table(io.BytesIO(file_bytes))

    # Preserve true nulls and Arrow-native types
    df = table.to_pandas(types_mapper=pd.ArrowDtype)

    # Convert empty/whitespace-only strings to missing
    df = df.replace(r"^\s*$", pd.NA, regex=True)

    # Return cleaned dataframe
    return df


@st.cache_data(show_spinner=True)
def load_schema_from_bytes(file_bytes: bytes):
    """Return a PyArrow schema from Parquet bytes."""
    pf = pq.ParquetFile(BytesIO(file_bytes))
    return pf.schema_arrow


def get_sample(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Return a sampled DataFrame, respecting row count."""
    if len(df) <= n:
        return df
    return df.sample(n=n, random_state=0)


def profile_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Simple profiling - one row per column with basic info."""
    profiler_rows = []
    for col in df.columns:
        series = df[col]
        profiler_rows.append(
            {
                "column": col,
                "dtype": str(series.dtype),
                "non_null_count": int(series.count()),
                "missing_count": int(series.isna().sum()),
                "missing_percent": float(series.isna().mean() * 100.0),
                "unique_count": int(series.nunique(dropna=True)),
            }
        )
    return pd.DataFrame(profiler_rows)


def get_numeric_columns(df: pd.DataFrame):
    return list(df.select_dtypes(include="number").columns)


def get_categorical_columns(df: pd.DataFrame):
    # For now, treat object and category as categorical
    return list(df.select_dtypes(include=["object", "category"]).columns)


def extract_column_metadata(schema):
    """
    Extract SDTM/ADaM metadata (labels, formats) from PyArrow schema.
    Returns a dict mapping column names to metadata dicts.
    """
    metadata_dict = {}
    for field in schema:
        col_meta = {}
        field_meta = field.metadata or {}
        
        # Common metadata keys in SDTM/ADaM Parquet files
        # Try various common keys for labels
        label_keys = ["label", "Label", "LABEL", "description", "Description", "DESCRIPTION", "title", "Title"]
        format_keys = ["format", "Format", "FORMAT", "fmt", "Fmt"]
        unit_keys = ["unit", "Unit", "UNIT", "units", "Units"]
        
        for key in label_keys:
            if key in field_meta:
                # Metadata is stored as bytes, decode if needed
                label_val = field_meta[key]
                if isinstance(label_val, bytes):
                    col_meta["label"] = label_val.decode("utf-8", errors="ignore")
                else:
                    col_meta["label"] = str(label_val)
                break
        
        for key in format_keys:
            if key in field_meta:
                format_val = field_meta[key]
                if isinstance(format_val, bytes):
                    col_meta["format"] = format_val.decode("utf-8", errors="ignore")
                else:
                    col_meta["format"] = str(format_val)
                break
        
        for key in unit_keys:
            if key in field_meta:
                unit_val = field_meta[key]
                if isinstance(unit_val, bytes):
                    col_meta["unit"] = unit_val.decode("utf-8", errors="ignore")
                else:
                    col_meta["unit"] = str(unit_val)
                break
        
        # Also check if metadata is stored as JSON string
        if not col_meta.get("label") and "pandas" in field_meta:
            try:
                import json
                pandas_meta = json.loads(field_meta["pandas"].decode("utf-8") if isinstance(field_meta["pandas"], bytes) else field_meta["pandas"])
                if "column_name" in pandas_meta and "metadata" in pandas_meta:
                    meta = pandas_meta.get("metadata", {})
                    if "label" in meta:
                        col_meta["label"] = meta["label"]
                    if "format" in meta:
                        col_meta["format"] = meta["format"]
            except:
                pass
        
        metadata_dict[field.name] = col_meta
    
    return metadata_dict


def is_id_variable(col_name: str, metadata: dict = None) -> bool:
    """
    Identify ID variables that shouldn't be summarized.
    Checks column name patterns and metadata.
    """
    col_upper = col_name.upper()
    
    # Common ID variable patterns
    id_patterns = [
        col_upper.endswith("ID"),
        col_upper.endswith("IDN"),
        col_upper.endswith("_ID"),
        col_upper.endswith("_IDN"),
        col_upper in ["SITEID", "SITEGR1", "TRTPN", "USUBJID", "SUBJID"],
        "ID" in col_upper and any(x in col_upper for x in ["SITE", "SUBJ", "PAT", "TRT"]),
    ]
    
    # Check metadata for hints
    if metadata:
        label = metadata.get("label", "").upper()
        if "ID" in label and any(x in label for x in ["IDENTIFIER", "IDENTIFICATION", "ID NUMBER"]):
            return True
    
    return any(id_patterns)


def get_numeric_columns_excluding_ids(df: pd.DataFrame, column_metadata: dict = None):
    """
    Get numeric columns, excluding ID variables.
    """
    numeric_cols = get_numeric_columns(df)
    if column_metadata is None:
        column_metadata = {}
    
    # Filter out ID variables
    filtered_cols = [
        col for col in numeric_cols 
        if not is_id_variable(col, column_metadata.get(col, {}))
    ]
    return filtered_cols


# ---------------------------
# Main content
# ---------------------------

if uploaded_file is None:
    st.info("Upload a Parquet file in the sidebar to begin.")
else:
    # Load data and schema
    file_bytes = uploaded_file.read()
    full_df = load_parquet_from_bytes(file_bytes)
    schema = load_schema_from_bytes(file_bytes)
    column_metadata = extract_column_metadata(schema)
    sample_df = get_sample(full_df, sample_size)
    profile_df = profile_dataframe(sample_df)

    st.success(
        f"Loaded dataset with approximately {len(full_df):,} rows and {len(full_df.columns)} columns."
    )

    # Tabs
    tab_overview, tab_table, tab_filter, tab_viz, tab_stats, tab_crosstab, tab_missing = st.tabs(
        [
            "Schema & Overview",
            "Table Explorer",
            "Filters",
            "Visualizations",
            "Summary Stats",
            "Crosstabs",
            "Missing Data",
        ]
    )

    # ---------------------------
    # Tab 1 - Schema & Overview
    # ---------------------------
    with tab_overview:
        st.subheader("Schema with SDTM/ADaM Metadata")
        schema_rows = []
        for field in schema:
            meta = column_metadata.get(field.name, {})
            is_id = is_id_variable(field.name, meta)
            schema_rows.append(
                {
                    "column": field.name,
                    "label": meta.get("label", ""),
                    "format": meta.get("format", ""),
                    "unit": meta.get("unit", ""),
                    "arrow_type": str(field.type),
                    "nullable": field.nullable,
                    "is_id_variable": "Yes" if is_id else "No",
                }
            )
        schema_df = pd.DataFrame(schema_rows)

        # Display schema with metadata prominently
        st.dataframe(schema_df, use_container_width=True)
        
        # Show raw metadata if available
        has_metadata = any(column_metadata.values())
        if has_metadata:
            st.success(f"✓ Found metadata (labels/formats) for {sum(1 for m in column_metadata.values() if m)} columns")
        else:
            st.info("ℹ️ No variable labels or formats found in metadata. If your Parquet file contains SDTM/ADaM metadata, it should appear above.")

        st.subheader("Column profiling (sample-based)")
        # Enhance profile with labels
        profile_enhanced = profile_df.copy()
        profile_enhanced["label"] = profile_enhanced["column"].map(
            lambda x: column_metadata.get(x, {}).get("label", "")
        )
        profile_enhanced["is_id"] = profile_enhanced["column"].map(
            lambda x: is_id_variable(x, column_metadata.get(x, {}))
        )
        # Reorder columns to show label early
        cols_order = ["column", "label", "is_id", "dtype", "non_null_count", "missing_count", "missing_percent", "unique_count"]
        profile_enhanced = profile_enhanced[cols_order]
        st.dataframe(profile_enhanced, use_container_width=True)

    # ---------------------------
    # Tab 2 - Table Explorer (AG Grid)
    # ---------------------------
    with tab_table:
        st.subheader("Table Explorer (sample)")
        st.caption(
            "Showing a sampled subset for responsiveness. Adjust sample size in the sidebar. "
            "Select specific columns below to improve readability when viewing many columns."
        )

        # Column selection for better readability
        if "table_selected_columns" not in st.session_state:
            # Default to first 15 columns for readability
            st.session_state.table_selected_columns = list(sample_df.columns)[:min(15, len(sample_df.columns))]
        
        col_select_col1, col_select_col2, col_select_col3 = st.columns([4, 1, 1])
        with col_select_col1:
            selected_columns = st.multiselect(
                "Select columns to display (helps with readability when you have many columns)",
                options=list(sample_df.columns),
                default=st.session_state.table_selected_columns,
                help="Select which columns to display in the table. This helps improve readability when you have many columns.",
            )
        with col_select_col2:
            if st.button("All", use_container_width=True):
                st.session_state.table_selected_columns = list(sample_df.columns)
                st.rerun()
        with col_select_col3:
            if st.button("Reset", use_container_width=True):
                st.session_state.table_selected_columns = list(sample_df.columns)[:min(15, len(sample_df.columns))]
                st.rerun()
        
        # Update session state
        if selected_columns != st.session_state.table_selected_columns:
            st.session_state.table_selected_columns = selected_columns
        
        if not selected_columns:
            st.info("Select at least one column to display.")
        else:
            display_df = sample_df[selected_columns].copy()
            
            # Configure AG Grid with better column sizing
            gb = GridOptionsBuilder.from_dataframe(display_df)
            gb.configure_pagination(
                enabled=True,
                paginationAutoPageSize=False,
                paginationPageSize=50,
            )
            gb.configure_side_bar()
            gb.configure_default_column(
                resizable=True,
                sortable=True,
                filter=True,
                wrapText=True,  # Enable text wrapping
                autoHeight=True,  # Auto-adjust row height
                minWidth=100,  # Minimum column width
            )
            
            # Set better default widths for columns based on content
            for col in display_df.columns:
                col_type = str(display_df[col].dtype)
                # Calculate appropriate width based on content
                max_len = display_df[col].astype(str).str.len().max() if len(display_df) > 0 else 10
                col_name_len = len(col)
                
                # Set width based on type and content
                if 'object' in col_type or 'string' in col_type:
                    # For text columns, use max of content length or column name, capped
                    width = min(max(max_len * 8, col_name_len * 10, 120), 300)
                elif 'int' in col_type or 'float' in col_type:
                    width = max(col_name_len * 10, 100)
                elif 'date' in col_type.lower() or 'time' in col_type.lower():
                    width = 150
                else:
                    width = 120
                
                gb.configure_column(col, width=int(width))
            
            # Add column labels as tooltips
            for col in display_df.columns:
                meta = column_metadata.get(col, {})
                label = meta.get("label", "")
                if label:
                    gb.configure_column(col, headerTooltip=f"{col}: {label}")
            
            grid_options = gb.build()

            AgGrid(
                display_df,
                gridOptions=grid_options,
                height=600,
                theme="streamlit",
                allow_unsafe_jscode=True,
                update_mode="MODEL_CHANGED",
            )
            
            st.caption(f"Displaying {len(display_df):,} rows × {len(display_df.columns)} columns. Use column filters and sorting to explore the data.")

    # ---------------------------
    # Tab 3 - Filters (Expression-based)
    # ---------------------------
    with tab_filter:
        st.subheader("Expression Filters (pandas.query style)")
        st.write(
            "Enter a filter expression using pandas.query syntax. "
            "Examples:\n\n"
            "`AGE > 50`\n\n"
            "`SEX == 'F' and AGE >= 18`\n\n"
            "`ARMCD in ['A', 'B']`"
        )

        filter_expr = st.text_input("Filter expression", value="", placeholder="e.g. AGE > 50 and SEX == 'F'")

        target_df = full_df if use_full_for_heavy_ops else sample_df

        if filter_expr:
            try:
                filtered_df = target_df.query(filter_expr)
                st.success(f"Filter applied, {len(filtered_df):,} rows match.")
                st.dataframe(filtered_df.head(100), use_container_width=True)
                st.caption("Showing first 100 rows of filtered data.")
            except Exception as e:
                st.error(f"Could not apply filter: {e}")
        else:
            st.info("Enter a filter expression to see filtered results.")

    # ---------------------------
    # Tab 4 - Visualizations (Plotly)
    # ---------------------------
    with tab_viz:
        st.subheader("Visual Builder")

        viz_df = sample_df  # keep visuals responsive

        numeric_cols = get_numeric_columns(viz_df)
        cat_cols = get_categorical_columns(viz_df)

        if len(viz_df) == 0:
            st.warning("No data available for visualization.")
        else:
            # Create display names with labels for better UX
            def get_col_display_name(col):
                label = column_metadata.get(col, {}).get("label", "")
                if label:
                    return f"{col} ({label})"
                return col
            
            col_display_map = {get_col_display_name(col): col for col in viz_df.columns}
            col_display_options = list(col_display_map.keys())
            
            col1, col2, col3 = st.columns(3)

            with col1:
                x_display = st.selectbox("X axis", options=col_display_options, index=0)
                x_col = col_display_map[x_display]

            with col2:
                y_display = st.selectbox(
                    "Y axis (optional, often numeric)",
                    options=["(none)"] + col_display_options,
                    index=0,
                )
                y_col = col_display_map[y_display] if y_display != "(none)" else None

            with col3:
                color_display = st.selectbox(
                    "Color (optional, often categorical)",
                    options=["(none)"] + col_display_options,
                    index=0,
                )
                color_col = col_display_map[color_display] if color_display != "(none)" else None

            chart_type = st.selectbox(
                "Chart type",
                options=["Histogram", "Boxplot", "Scatter", "Bar"],
                index=0,
            )

            if st.button("Create chart"):
                plot_kwargs = {}
                if y_col:
                    plot_kwargs["y"] = y_col
                if color_col:
                    plot_kwargs["color"] = color_col

                try:
                    # Get labels for axis titles
                    x_label = column_metadata.get(x_col, {}).get("label", x_col)
                    labels_dict = {x_col: x_label}
                    if y_col:
                        y_label = column_metadata.get(y_col, {}).get("label", y_col)
                        labels_dict[y_col] = y_label
                    if color_col:
                        color_label = column_metadata.get(color_col, {}).get("label", color_col)
                        labels_dict[color_col] = color_label
                    
                    if chart_type == "Histogram":
                        fig = px.histogram(viz_df, x=x_col, **plot_kwargs, labels=labels_dict)
                    elif chart_type == "Boxplot":
                        fig = px.box(viz_df, x=x_col, **plot_kwargs, labels=labels_dict)
                    elif chart_type == "Scatter":
                        if not y_col:
                            st.error("Scatter plot requires a Y axis.")
                            fig = None
                        else:
                            fig = px.scatter(viz_df, x=x_col, **plot_kwargs, labels=labels_dict)
                    elif chart_type == "Bar":
                        if not y_col:
                            # Count by category
                            bar_labels = {x_col: x_label}
                            if color_col:
                                bar_labels[color_col] = column_metadata.get(color_col, {}).get("label", color_col)
                            fig = px.bar(
                                viz_df,
                                x=x_col,
                                color=color_col,
                                labels=bar_labels
                            )
                        else:
                            fig = px.bar(viz_df, x=x_col, **plot_kwargs, labels=labels_dict)
                    else:
                        fig = None

                    if fig is not None:
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Could not create chart: {e}")

    # ---------------------------
    # Tab 5 - Summary Stats
    # ---------------------------
    with tab_stats:
        st.subheader("Summary Statistics")

        stats_df_source = full_df if use_full_for_heavy_ops else sample_df
        all_numeric_cols = get_numeric_columns(stats_df_source)
        # Exclude ID variables by default
        numeric_cols = get_numeric_columns_excluding_ids(stats_df_source, column_metadata)
        id_numeric_cols = [col for col in all_numeric_cols if col not in numeric_cols]

        if not all_numeric_cols:
            st.warning("No numeric columns found for summary statistics.")
        else:
            if id_numeric_cols:
                st.info(
                    f"ℹ️ Excluded {len(id_numeric_cols)} ID variable(s) from default selection: "
                    f"{', '.join(id_numeric_cols[:5])}"
                    + (f" and {len(id_numeric_cols) - 5} more" if len(id_numeric_cols) > 5 else "")
                    + ". ID variables (like SITEID, TRTPN) are typically not meaningful to summarize."
                )
            
            # Create display names with labels
            def get_col_display_name(col):
                label = column_metadata.get(col, {}).get("label", "")
                if label:
                    return f"{col} ({label})"
                return col
            
            display_options = {get_col_display_name(col): col for col in all_numeric_cols}
            
            default_selected = numeric_cols[: min(5, len(numeric_cols))] if numeric_cols else []
            default_selected_display = [get_col_display_name(col) for col in default_selected]
            
            selected_display = st.multiselect(
                "Numeric columns to summarize",
                options=list(display_options.keys()),
                default=default_selected_display,
                help="ID variables are excluded by default. Select from all numeric columns if needed.",
            )
            
            selected_cols = [display_options[disp] for disp in selected_display]

            group_cols = st.multiselect(
                "Group by (categorical columns)",
                options=get_categorical_columns(stats_df_source),
                default=[],
            )

            if selected_cols:
                if group_cols:
                    grouped = stats_df_source.groupby(group_cols)[selected_cols]
                    stats_df = grouped.agg(
                        ["mean", "median", "std", "min", "max", "count"]
                    )
                else:
                    stats_df = stats_df_source[selected_cols].agg(
                        ["mean", "median", "std", "min", "max", "count"]
                    )
                st.dataframe(stats_df, use_container_width=True)
            else:
                st.info("Select at least one numeric column to summarize.")

    # ---------------------------
    # Tab 6 - Crosstabs
    # ---------------------------
    with tab_crosstab:
        st.subheader("Crosstab Builder")

        crosstab_df_source = full_df if use_full_for_heavy_ops else sample_df
        cat_cols = get_categorical_columns(crosstab_df_source)

        if len(cat_cols) < 2:
            st.warning("Need at least two categorical columns for crosstabs.")
        else:
            row_cat = st.selectbox("Row category", options=cat_cols)
            col_cat = st.selectbox("Column category", options=[c for c in cat_cols if c != row_cat])

            if row_cat and col_cat:
                ct = pd.crosstab(
                    crosstab_df_source[row_cat],
                    crosstab_df_source[col_cat],
                    dropna=False,
                    margins=True,
                    margins_name="Total",
                )
                st.dataframe(ct, use_container_width=True)

    # ---------------------------
    # Tab 7 - Missing Data
    # ---------------------------
    with tab_missing:
        st.subheader("Missing Data Overview")

        missing_df_source = full_df if use_full_for_heavy_ops else sample_df

        missing_counts = missing_df_source.isna().sum()
        missing_percents = missing_df_source.isna().mean() * 100.0
        missing_summary = pd.DataFrame(
            {
                "missing_count": missing_counts,
                "missing_percent": missing_percents,
                "dtype": missing_df_source.dtypes,
            }
        )
        missing_summary = missing_summary[missing_summary["missing_count"] > 0].sort_values(
            by="missing_percent", ascending=False
        )

        if missing_summary.empty:
            st.success("No missing data found in the dataset/sample.")
        else:
            st.dataframe(missing_summary, use_container_width=True)

            try:
                import seaborn as sns
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(
                    missing_df_source.isna(),
                    cbar=False,
                    yticklabels=False,
                    ax=ax,
                    cmap="viridis",
                )
                st.pyplot(fig)
            except ImportError:
                st.info("Install seaborn and matplotlib to see missing data heatmap visualization.")
