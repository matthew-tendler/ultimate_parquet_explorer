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
    """Load a Parquet file into a pandas DataFrame."""
    return pd.read_parquet(BytesIO(file_bytes))


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
        st.subheader("Schema")
        schema_rows = []
        for field in schema:
            schema_rows.append(
                {
                    "column": field.name,
                    "arrow_type": str(field.type),
                    "nullable": field.nullable,
                    "metadata": dict(field.metadata or {}),
                }
            )
        schema_df = pd.DataFrame(schema_rows)

        st.write("PyArrow schema:")
        st.dataframe(schema_df, use_container_width=True)

        st.subheader("Column profiling (sample-based)")
        st.dataframe(profile_df, use_container_width=True)

    # ---------------------------
    # Tab 2 - Table Explorer (AG Grid)
    # ---------------------------
    with tab_table:
        st.subheader("Table Explorer (sample)")
        st.caption(
            "Showing a sampled subset for responsiveness. Adjust sample size in the sidebar."
        )

        gb = GridOptionsBuilder.from_dataframe(sample_df)
        gb.configure_pagination(enabled=True)
        gb.configure_side_bar()
        gb.configure_default_column(
            resizable=True,
            sortable=True,
            filter=True,
        )
        grid_options = gb.build()

        AgGrid(
            sample_df,
            gridOptions=grid_options,
            height=500,
            theme="streamlit",
        )

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
            col1, col2, col3 = st.columns(3)

            with col1:
                x_col = st.selectbox("X axis", options=list(viz_df.columns), index=0)

            with col2:
                y_col = st.selectbox(
                    "Y axis (optional, often numeric)",
                    options=["(none)"] + list(viz_df.columns),
                    index=0,
                )

            with col3:
                color_col = st.selectbox(
                    "Color (optional, often categorical)",
                    options=["(none)"] + list(viz_df.columns),
                    index=0,
                )

            chart_type = st.selectbox(
                "Chart type",
                options=["Histogram", "Boxplot", "Scatter", "Bar"],
                index=0,
            )

            if st.button("Create chart"):
                plot_kwargs = {}
                if y_col != "(none)":
                    plot_kwargs["y"] = y_col
                if color_col != "(none)":
                    plot_kwargs["color"] = color_col

                try:
                    if chart_type == "Histogram":
                        fig = px.histogram(viz_df, x=x_col, **plot_kwargs)
                    elif chart_type == "Boxplot":
                        fig = px.box(viz_df, x=x_col, **plot_kwargs)
                    elif chart_type == "Scatter":
                        if y_col == "(none)":
                            st.error("Scatter plot requires a Y axis.")
                            fig = None
                        else:
                            fig = px.scatter(viz_df, x=x_col, **plot_kwargs)
                    elif chart_type == "Bar":
                        if y_col == "(none)":
                            # Count by category
                            fig = px.bar(
                                viz_df,
                                x=x_col,
                                color=color_col if color_col != "(none)" else None,
                            )
                        else:
                            fig = px.bar(viz_df, x=x_col, **plot_kwargs)
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
        numeric_cols = get_numeric_columns(stats_df_source)

        if not numeric_cols:
            st.warning("No numeric columns found for summary statistics.")
        else:
            selected_cols = st.multiselect(
                "Numeric columns to summarize",
                options=numeric_cols,
                default=numeric_cols[: min(5, len(numeric_cols))],
            )

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
