import streamlit as st
import pandas as pd
import pyarrow.parquet as pq
from io import BytesIO
import plotly.express as px
from st_aggrid import AgGrid, GridOptionsBuilder
import re


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

# Add sample data option
use_sample_data = st.sidebar.button(
    "📊 Load Sample Data",
    help="Load the sample ADaM dataset (adadas.parquet) to explore the app without uploading your own file.",
    use_container_width=True,
)

uploaded_file = st.sidebar.file_uploader(
    "Upload a Parquet file",
    type=["parquet"],
    help="Upload your own parquet file, or use the sample data button above to try the app.",
)

# Initialize session state for sample data
if "use_sample_data" not in st.session_state:
    st.session_state.use_sample_data = False

# Handle sample data button click
if use_sample_data:
    st.session_state.use_sample_data = True

# If user uploads a file, switch off sample data
if uploaded_file is not None:
    st.session_state.use_sample_data = False


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


def convert_sas_where_to_pandas(sas_expr: str) -> str:
    """
    Convert SAS WHERE syntax to pandas.query compatible syntax.
    
    Examples:
    - AGE > 50 AND SEX = 'F' -> AGE > 50 and SEX == 'F'
    - ARMCD IN ('A', 'B') -> ARMCD in ['A', 'B']
    """
    expr = sas_expr.strip()
    
    # Handle string comparisons: SEX = 'F' -> SEX == 'F'
    # Match: column = 'value' or column = "value"
    expr = re.sub(r'(\w+)\s*=\s*(["\'])([^"\']+)\2', r'\1 == \2\3\2', expr)
    
    # Handle numeric comparisons: AGE = 50 -> AGE == 50
    # Match: column = number (not already ==)
    expr = re.sub(r'(\w+)\s*=\s*(\d+(?:\.\d+)?)(?![=<>])', r'\1 == \2', expr)
    
    # Handle IN: ARMCD IN ('A', 'B') -> ARMCD in ['A', 'B']
    def replace_in(match):
        col = match.group(1)
        values = match.group(2)
        # Replace single quotes with double quotes and wrap in brackets
        values_list = values.replace("'", '"').replace('"', '"')
        return f"{col} in [{values_list}]"
    
    expr = re.sub(r'(\w+)\s+IN\s+\(([^)]+)\)', replace_in, expr, flags=re.IGNORECASE)
    
    # Handle AND/OR (ensure proper spacing and lowercase)
    expr = re.sub(r'\s+AND\s+', ' and ', expr, flags=re.IGNORECASE)
    expr = re.sub(r'\s+OR\s+', ' or ', expr, flags=re.IGNORECASE)
    expr = re.sub(r'\s+NOT\s+', ' not ', expr, flags=re.IGNORECASE)
    
    return expr


def convert_r_dplyr_to_pandas(r_expr: str) -> str:
    """
    Convert R dplyr filter syntax to pandas.query compatible syntax.
    
    Examples:
    - age > 50 & sex == 'F' -> age > 50 and sex == 'F'
    - armcd %in% c('A', 'B') -> armcd in ['A', 'B']
    """
    expr = r_expr.strip()
    
    # Replace R operators with Python equivalents
    expr = re.sub(r'\s+&\s+', ' and ', expr)
    expr = re.sub(r'\s+\|\s+', ' or ', expr)
    expr = re.sub(r'!\s*(\w+)', r'not \1', expr)
    
    # Handle %in%: armcd %in% c('A', 'B') -> armcd in ['A', 'B']
    def replace_in_r(match):
        col = match.group(1)
        values = match.group(2)
        # Replace single quotes with double quotes and wrap in brackets
        values_list = values.replace("'", '"').replace('"', '"')
        return f"{col} in [{values_list}]"
    
    expr = re.sub(r'(\w+)\s*%in%\s*c\(([^)]+)\)', replace_in_r, expr, flags=re.IGNORECASE)
    
    return expr


def build_pandas_query_from_conditions(conditions: list, df: pd.DataFrame) -> str:
    """
    Build a pandas query string from visual filter conditions.
    
    Args:
        conditions: List of dicts with keys: column, operator, value, logic
        df: DataFrame to get column types from
    
    Returns:
        pandas query string
    """
    if not conditions:
        return ""
    
    query_parts = []
    
    for i, cond in enumerate(conditions):
        col = cond.get('column', '')
        operator = cond.get('operator', '=')
        value = cond.get('value', '')
        logic = cond.get('logic', 'AND')
        
        if operator in ['IS NULL', 'IS NOT NULL']:
            # Handle NULL checks
            if operator == 'IS NULL':
                query_parts.append(f"{col}.isna()")
            elif operator == 'IS NOT NULL':
                query_parts.append(f"{col}.notna()")
        elif col:
            # Get column dtype to determine value handling
            col_dtype = str(df[col].dtype) if col in df.columns else 'object'
            is_numeric = 'int' in col_dtype or 'float' in col_dtype
            
            # Build condition based on operator
            if operator == '=':
                if is_numeric:
                    query_parts.append(f"{col} == {value}")
                else:
                    # String value needs quotes
                    value_clean = value.replace("'", "\\'").replace('"', '\\"')
                    query_parts.append(f"{col} == '{value_clean}'")
            elif operator == '!=':
                if is_numeric:
                    query_parts.append(f"{col} != {value}")
                else:
                    value_clean = value.replace("'", "\\'").replace('"', '\\"')
                    query_parts.append(f"{col} != '{value_clean}'")
            elif operator in ['>', '>=', '<', '<=']:
                query_parts.append(f"{col} {operator} {value}")
            elif operator == 'IN':
                # Parse comma-separated values
                values = [v.strip().strip("'\"") for v in value.split(',')]
                if is_numeric:
                    query_parts.append(f"{col} in [{', '.join(values)}]")
                else:
                    values_quoted = [f"'{v.replace(chr(39), chr(39)+chr(39))}'" for v in values]
                    query_parts.append(f"{col} in [{', '.join(values_quoted)}]")
            elif operator == 'CONTAINS':
                value_clean = value.replace("'", "\\'").replace('"', '\\"')
                query_parts.append(f"{col}.str.contains('{value_clean}', na=False)")
        
        # Add logic operator between conditions (except for last one)
        if i < len(conditions) - 1:
            query_parts.append(logic.lower())
    
    return ' '.join(query_parts)


# ---------------------------
# Main content
# ---------------------------

# Determine which file to use
file_to_use = None
data_source = None

if st.session_state.use_sample_data:
    # Load sample file from repository
    sample_file_path = "adadas.parquet"
    try:
        with open(sample_file_path, "rb") as f:
            file_to_use = BytesIO(f.read())
        data_source = "sample"
        st.sidebar.success("✓ Sample data loaded")
    except FileNotFoundError:
        st.sidebar.error(f"Sample file not found at {sample_file_path}")
        st.session_state.use_sample_data = False
elif uploaded_file is not None:
    file_to_use = uploaded_file
    data_source = "uploaded"

if file_to_use is None:
    st.info("👆 Upload a Parquet file in the sidebar, or click 'Load Sample Data' to try the app with the sample ADaM dataset.")
else:
    # Load data and schema
    file_bytes = file_to_use.read()
    if data_source == "sample":
        st.sidebar.info("ℹ️ Using sample data (adadas.parquet). You can still upload your own file above.")
    
    full_df = load_parquet_from_bytes(file_bytes)
    schema = load_schema_from_bytes(file_bytes)
    column_metadata = extract_column_metadata(schema)
    profile_df = profile_dataframe(full_df)

    st.success(
        f"Loaded dataset with approximately {len(full_df):,} rows and {len(full_df.columns)} columns."
        + (" (Sample data)" if data_source == "sample" else "")
    )

    # Tabs
    tab_overview, tab_table, tab_filter, tab_viz, tab_stats, tab_missing = st.tabs(
        [
            "Schema & Overview",
            "Table Explorer",
            "Filters",
            "Visualizations",
            "Summary Stats",
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

        st.subheader("Column profiling")
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
        st.subheader("Table Explorer")
        st.caption(
            "Select specific columns below to improve readability when viewing many columns."
        )

        # Column selection for better readability
        if "table_selected_columns" not in st.session_state:
            # Default to first 15 columns for readability
            st.session_state.table_selected_columns = list(full_df.columns)[:min(15, len(full_df.columns))]
        
        col_select_col1, col_select_col2, col_select_col3 = st.columns([4, 1, 1])
        with col_select_col1:
            selected_columns = st.multiselect(
                "Select columns to display (helps with readability when you have many columns)",
                options=list(full_df.columns),
                default=st.session_state.table_selected_columns,
                help="Select which columns to display in the table. This helps improve readability when you have many columns.",
            )
        with col_select_col2:
            if st.button("All", use_container_width=True):
                st.session_state.table_selected_columns = list(full_df.columns)
                st.rerun()
        with col_select_col3:
            if st.button("Reset", use_container_width=True):
                st.session_state.table_selected_columns = list(full_df.columns)[:min(15, len(full_df.columns))]
                st.rerun()
        
        # Update session state
        if selected_columns != st.session_state.table_selected_columns:
            st.session_state.table_selected_columns = selected_columns
        
        if not selected_columns:
            st.info("Select at least one column to display.")
        else:
            display_df = full_df[selected_columns].copy()
            
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
    # Tab 3 - Filters
    # ---------------------------
    with tab_filter:
        st.subheader("Data Filters")
        
        # Initialize filter conditions in session state
        if "filter_conditions" not in st.session_state:
            st.session_state.filter_conditions = []
        
        filter_mode = st.radio(
            "Filter mode",
            options=["Visual Builder", "Expression (SAS/R/Python)"],
            horizontal=True,
            help="Choose between a visual form-based builder or write expressions in your preferred syntax"
        )
        
        if filter_mode == "Visual Builder":
            st.write("**Build filters using the form below. Click 'Add Condition' to add more filters.**")
            
            # Display existing conditions
            if st.session_state.filter_conditions:
                st.write("**Current Filters:**")
                for i, condition in enumerate(st.session_state.filter_conditions):
                    with st.container():
                        col1, col2, col3, col4, col5 = st.columns([3, 2, 3, 1.5, 0.5])
                        
                        with col1:
                            # Get column display names with labels
                            def get_col_display_name(col):
                                label = column_metadata.get(col, {}).get("label", "")
                                if label:
                                    return f"{col} ({label})"
                                return col
                            
                            col_display_map = {get_col_display_name(col): col for col in full_df.columns}
                            col_display_options = list(col_display_map.keys())
                            
                            current_col_display = get_col_display_name(condition.get('column', full_df.columns[0]))
                            if current_col_display not in col_display_options:
                                current_col_display = col_display_options[0]
                            
                            selected_col_display = st.selectbox(
                                "Column",
                                options=col_display_options,
                                index=col_display_options.index(current_col_display) if current_col_display in col_display_options else 0,
                                key=f"filter_col_{i}",
                                label_visibility="collapsed"
                            )
                            condition['column'] = col_display_map[selected_col_display]
                        
                        with col2:
                            operator = st.selectbox(
                                "Operator",
                                options=["=", "!=", ">", ">=", "<", "<=", "IN", "CONTAINS", "IS NULL", "IS NOT NULL"],
                                index=["=", "!=", ">", ">=", "<", "<=", "IN", "CONTAINS", "IS NULL", "IS NOT NULL"].index(condition.get('operator', '=')),
                                key=f"filter_op_{i}",
                                label_visibility="collapsed"
                            )
                            condition['operator'] = operator
                        
                        with col3:
                            if operator not in ["IS NULL", "IS NOT NULL"]:
                                value = st.text_input(
                                    "Value",
                                    value=condition.get('value', ''),
                                    key=f"filter_val_{i}",
                                    placeholder="Enter value or comma-separated list for IN",
                                    label_visibility="collapsed"
                                )
                                condition['value'] = value
                            else:
                                st.write("")  # Empty space for alignment
                        
                        with col4:
                            if i < len(st.session_state.filter_conditions) - 1:
                                logic = st.selectbox(
                                    "Logic",
                                    options=["AND", "OR"],
                                    index=0 if condition.get('logic', 'AND') == 'AND' else 1,
                                    key=f"filter_logic_{i}",
                                    label_visibility="collapsed"
                                )
                                condition['logic'] = logic
                            else:
                                st.write("")  # Empty space for last condition
                        
                        with col5:
                            if st.button("🗑️", key=f"filter_remove_{i}", help="Remove this condition"):
                                st.session_state.filter_conditions.pop(i)
                                st.rerun()
            
            # Add new condition button
            col1, col2 = st.columns([4, 1])
            with col1:
                if st.button("➕ Add Condition", use_container_width=True):
                    st.session_state.filter_conditions.append({
                        'column': full_df.columns[0],
                        'operator': '=',
                        'value': '',
                        'logic': 'AND'
                    })
                    st.rerun()
            
            # Clear all and apply buttons
            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                if st.button("Clear All", use_container_width=True):
                    st.session_state.filter_conditions = []
                    st.rerun()
            
            with col2:
                apply_visual = st.button("Apply Filters", type="primary", use_container_width=True)
            
            # Apply visual filters
            if apply_visual and st.session_state.filter_conditions:
                try:
                    # Validate conditions have values (except for NULL checks)
                    valid_conditions = []
                    for cond in st.session_state.filter_conditions:
                        if cond['operator'] in ['IS NULL', 'IS NOT NULL']:
                            valid_conditions.append(cond)
                        elif cond.get('value', '').strip():
                            valid_conditions.append(cond)
                    
                    if valid_conditions:
                        pandas_query = build_pandas_query_from_conditions(valid_conditions, full_df)
                        filtered_df = full_df.query(pandas_query)
                        st.success(f"✅ Filter applied! {len(filtered_df):,} rows match (out of {len(full_df):,} total).")
                        st.code(f"Generated query: {pandas_query}", language="python")
                        st.dataframe(filtered_df.head(100), use_container_width=True)
                        st.caption(f"Showing first 100 rows of {len(filtered_df):,} filtered rows.")
                    else:
                        st.warning("⚠️ Please add at least one valid filter condition.")
                except Exception as e:
                    st.error(f"❌ Could not apply filter: {e}")
                    st.info("💡 Tip: Make sure values match the column data type (numbers for numeric columns, text for text columns).")
            elif apply_visual:
                st.info("ℹ️ Add at least one filter condition to apply filters.")
        
        else:  # Expression mode
            syntax_type = st.selectbox(
                "Syntax type",
                options=["SAS WHERE", "R dplyr", "Python pandas"],
                help="Choose the syntax you're most comfortable with"
            )
            
            if syntax_type == "SAS WHERE":
                st.write("""
                **SAS WHERE Syntax Examples:**
                - `AGE > 50 AND SEX = 'F'`
                - `ARMCD IN ('A', 'B')`
                - `AGE >= 18 AND (SEX = 'M' OR SEX = 'F')`
                - `AVAL IS NOT NULL`
                """)
                placeholder = "e.g. AGE > 50 AND SEX = 'F'"
            elif syntax_type == "R dplyr":
                st.write("""
                **R dplyr Syntax Examples:**
                - `age > 50 & sex == 'F'`
                - `armcd %in% c('A', 'B')`
                - `age >= 18 & (sex == 'M' | sex == 'F')`
                - `!is.na(aval)`
                """)
                placeholder = "e.g. age > 50 & sex == 'F'"
            else:  # Python pandas
                st.write("""
                **Python pandas.query Syntax Examples:**
                - `AGE > 50 and SEX == 'F'`
                - `ARMCD in ['A', 'B']`
                - `AGE >= 18 and (SEX == 'M' or SEX == 'F')`
                - `AVAL.notna()`
                """)
                placeholder = "e.g. AGE > 50 and SEX == 'F'"
            
            filter_expr = st.text_input(
                "Filter expression", 
                value="", 
                placeholder=placeholder,
                help="Enter your filter expression using the selected syntax"
            )
            
            if filter_expr:
                try:
                    # Convert to pandas query syntax if needed
                    if syntax_type == "SAS WHERE":
                        pandas_expr = convert_sas_where_to_pandas(filter_expr)
                        st.caption(f"Converted to pandas: `{pandas_expr}`")
                    elif syntax_type == "R dplyr":
                        pandas_expr = convert_r_dplyr_to_pandas(filter_expr)
                        st.caption(f"Converted to pandas: `{pandas_expr}`")
                    else:
                        pandas_expr = filter_expr
                    
                    filtered_df = full_df.query(pandas_expr)
                    st.success(f"✅ Filter applied! {len(filtered_df):,} rows match (out of {len(full_df):,} total).")
                    st.dataframe(filtered_df.head(100), use_container_width=True)
                    st.caption(f"Showing first 100 rows of {len(filtered_df):,} filtered rows.")
                except Exception as e:
                    st.error(f"❌ Could not apply filter: {e}")
                    st.info("💡 Tip: Check your syntax. Column names are case-sensitive. For SAS/R syntax, make sure operators and values are correctly formatted.")
            else:
                st.info("ℹ️ Enter a filter expression to see filtered results.")

    # ---------------------------
    # Tab 4 - Visualizations (Plotly)
    # ---------------------------
    with tab_viz:
        st.subheader("Visual Builder")

        viz_df = full_df

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

        all_numeric_cols = get_numeric_columns(full_df)
        # Exclude ID variables by default
        numeric_cols = get_numeric_columns_excluding_ids(full_df, column_metadata)
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
                options=get_categorical_columns(full_df),
                default=[],
            )

            if selected_cols:
                if group_cols:
                    grouped = full_df.groupby(group_cols)[selected_cols]
                    stats_df = grouped.agg(
                        ["mean", "median", "std", "min", "max", "count"]
                    )
                else:
                    stats_df = full_df[selected_cols].agg(
                        ["mean", "median", "std", "min", "max", "count"]
                    )
                st.dataframe(stats_df, use_container_width=True)
            else:
                st.info("Select at least one numeric column to summarize.")

    # ---------------------------
    # Tab 6 - Missing Data
    # ---------------------------
    with tab_missing:
        st.subheader("Missing Data Overview")

        missing_counts = full_df.isna().sum()
        missing_percents = full_df.isna().mean() * 100.0
        missing_summary = pd.DataFrame(
            {
                "missing_count": missing_counts,
                "missing_percent": missing_percents,
                "dtype": full_df.dtypes,
            }
        )
        missing_summary = missing_summary[missing_summary["missing_count"] > 0].sort_values(
            by="missing_percent", ascending=False
        )

        if missing_summary.empty:
            st.success("No missing data found in the dataset.")
        else:
            st.dataframe(missing_summary, use_container_width=True)

            # Create missing data heatmap using Plotly
            missing_matrix = full_df.isna().astype(int)
            
            # Sample the data if too large for visualization (limit to 1000 rows for performance)
            if len(missing_matrix) > 1000:
                missing_matrix_sample = missing_matrix.sample(n=1000, random_state=0)
                st.caption(f"Showing heatmap for 1,000 randomly sampled rows (out of {len(missing_matrix):,} total rows)")
            else:
                missing_matrix_sample = missing_matrix
            
            fig = px.imshow(
                missing_matrix_sample.T,
                labels=dict(x="Row", y="Column", color="Missing"),
                x=[f"Row {i+1}" for i in range(len(missing_matrix_sample))],
                y=missing_matrix_sample.columns.tolist(),
                color_continuous_scale="Viridis",
                aspect="auto",
                title="Missing Data Heatmap (Yellow = Missing, Purple = Present)"
            )
            fig.update_layout(height=max(400, len(missing_matrix_sample.columns) * 20))
            st.plotly_chart(fig, use_container_width=True)
