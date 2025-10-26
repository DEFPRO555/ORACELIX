# ORACELIX - Customer Segmentation Intelligence
# "Where Sacred Geometry Meets Artificial Intelligence"
# Precision. Purity. Perfection — The Geometry of Infinite Mind

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
import warnings
import os
import time
import base64
from pathlib import Path

warnings.filterwarnings('ignore')

# ============ ORACELIX CONFIGURATION ============
BRAND_NAME = "ORACELIX"
BRAND_TAGLINE = "Where Sacred Geometry Meets Artificial Intelligence"
BRAND_MOTTO = "Precision. Purity. Perfection — The Geometry of Infinite Mind"
PRIMARY_COLOR = "#1f77b4"
ACCENT_COLOR = "#ff7f0e"
HEADER_BANNER = "2.png"
FOOTER_BANNER = "4.png"
MIDDLE_BANNER = "3.png"

# ============ HELPER FUNCTIONS ============

def validate_environment():
    """Validate that required directories exist"""
    os.makedirs('reports', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('exports', exist_ok=True)

def load_image(image_path):
    """Load and encode image for display"""
    try:
        if os.path.exists(image_path):
            with open(image_path, "rb") as f:
                data = base64.b64encode(f.read()).decode()
                return f"data:image/png;base64,{data}"
    except:
        pass
    return None

def get_file_icon(file_type):
    """Get icon for file type"""
    icons = {
        'csv': '📊',
        'xlsx': '📗',
        'xls': '📗',
        'pdf': '📕'
    }
    return icons.get(file_type.lower(), '📄')

# ============ PAGE CONFIGURATION ============
st.set_page_config(
    page_title=f"{BRAND_NAME} - Customer Intelligence",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============ CUSTOM STYLING ============
st.markdown(f"""
<style>
    /* Main Theme */
    .stApp {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }}

    /* Header Styling */
    .oracelix-header {{
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }}

    .oracelix-title {{
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(45deg, #FFD700, #FFA500, #FFD700);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: 3px;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }}

    .oracelix-tagline {{
        color: #E0E0E0;
        font-size: 1.2rem;
        font-style: italic;
        margin-top: 0.5rem;
    }}

    .oracelix-motto {{
        color: #B8B8B8;
        font-size: 0.9rem;
        margin-top: 0.3rem;
        font-weight: 300;
    }}

    /* Card Styling */
    .metric-card {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        color: white;
        margin: 0.5rem 0;
    }}

    /* Section Headers */
    .section-header {{
        font-size: 2rem;
        color: #FFD700;
        margin-top: 2rem;
        margin-bottom: 1rem;
        font-weight: 600;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }}

    /* Sidebar Styling */
    .css-1d391kg {{
        background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%);
    }}

    /* Button Styling */
    .stButton>button {{
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s;
    }}

    .stButton>button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }}

    /* Progress Bar */
    .stProgress > div > div > div > div {{
        background: linear-gradient(45deg, #FFD700, #FFA500);
    }}

    /* File Uploader */
    .uploadedFile {{
        background: rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 1rem;
    }}
</style>
""", unsafe_allow_html=True)

# ============ DATA LOADING FUNCTIONS ============

@st.cache_data
def load_csv_data(file):
    """Load CSV file"""
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()
    return df

@st.cache_data
def load_excel_data(file):
    """Load Excel file (.xlsx, .xls)"""
    try:
        # Try reading all sheets
        excel_file = pd.ExcelFile(file)

        if len(excel_file.sheet_names) == 1:
            # Single sheet - load directly
            df = pd.read_excel(file, sheet_name=0)
        else:
            # Multiple sheets - let user choose
            st.sidebar.info(f"📗 Found {len(excel_file.sheet_names)} sheets")
            sheet_name = st.sidebar.selectbox(
                "Select Sheet:",
                excel_file.sheet_names
            )
            df = pd.read_excel(file, sheet_name=sheet_name)

        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        st.error(f"❌ Error loading Excel file: {str(e)}")
        return None

@st.cache_data
def load_pdf_data(file):
    """Load PDF file (extract tables)"""
    try:
        import tabula

        # Extract all tables from PDF
        tables = tabula.read_pdf(file, pages='all', multiple_tables=True)

        if len(tables) == 0:
            st.error("❌ No tables found in PDF")
            return None

        if len(tables) == 1:
            df = tables[0]
        else:
            st.sidebar.info(f"📕 Found {len(tables)} tables in PDF")
            table_idx = st.sidebar.selectbox(
                "Select Table:",
                range(len(tables)),
                format_func=lambda x: f"Table {x+1}"
            )
            df = tables[table_idx]

        df.columns = df.columns.str.strip()
        return df

    except ImportError:
        st.error("""
        ❌ **PDF Support Not Installed**

        To enable PDF upload, install:
        ```
        pip install tabula-py
        ```

        Note: Also requires Java Runtime Environment (JRE)
        """)
        return None
    except Exception as e:
        st.error(f"❌ Error loading PDF: {str(e)}")
        return None

@st.cache_data
def load_data(uploaded_file=None):
    """Universal data loader - supports CSV, Excel, PDF"""
    df = None
    file_type = None

    if uploaded_file is not None:
        # Get file extension
        file_ext = uploaded_file.name.split('.')[-1].lower()
        file_type = file_ext

        try:
            # Route to appropriate loader
            if file_ext == 'csv':
                df = load_csv_data(uploaded_file)
                st.success(f"📊 CSV loaded: {len(df):,} rows, {len(df.columns)} columns")

            elif file_ext in ['xlsx', 'xls']:
                df = load_excel_data(uploaded_file)
                if df is not None:
                    st.success(f"📗 Excel loaded: {len(df):,} rows, {len(df.columns)} columns")

            elif file_ext == 'pdf':
                df = load_pdf_data(uploaded_file)
                if df is not None:
                    st.success(f"📕 PDF loaded: {len(df):,} rows, {len(df.columns)} columns")
            else:
                st.error(f"❌ Unsupported file type: .{file_ext}")
                return None

        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            return None

    # Fallback: Show upload instructions (no default data files for cloud deployment)
    if df is None:
        st.warning("""
        📁 **Please Upload Your Data**

        Click the file uploader in the sidebar to get started!

        **Supported Formats:**
        - CSV (.csv)
        - Excel (.xlsx, .xls)
        - PDF (.pdf) - Tables only

        **Don't have data?** Download a sample CSV using the button in the sidebar.
        """)
        return None

    # Clean data
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
    for col in non_numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')

    return df

# ============ ML FUNCTIONS ============

@st.cache_data
def preprocess_data(df, selected_features):
    """Preprocess data: select features and scale"""
    X = df[selected_features].copy()
    X = X.fillna(X.median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X, X_scaled, scaler

@st.cache_data
def train_kmeans(X_scaled, n_clusters, random_state=42):
    """Train K-Means with automatic optimization"""
    # Use MiniBatch for large datasets
    if X_scaled.shape[0] > 5000:
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            batch_size=1000,
            max_iter=100,
            n_init=3
        )
    else:
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10
        )

    clusters = kmeans.fit_predict(X_scaled)
    return kmeans, clusters

# ============ MAIN APP ============

def main():
    validate_environment()

    # ============ HEADER BANNER ============
    header_banner = load_image(HEADER_BANNER)

    if header_banner:
        st.markdown(f"""
        <div style="width: 100%; margin-bottom: 2rem;">
            <img src="{header_banner}" style="width: 100%; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.3);">
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="oracelix-header">
            <div class="oracelix-title">🔮 {BRAND_NAME}</div>
            <div class="oracelix-tagline">{BRAND_TAGLINE}</div>
            <div class="oracelix-motto">{BRAND_MOTTO}</div>
        </div>
        """, unsafe_allow_html=True)

    # ============ SECURITY & PRIVACY NOTICE ============
    st.info("""
    🔒 **Your Data is 100% Safe & Private**
    - ✅ All processing happens locally on your computer
    - ✅ No data uploaded to cloud or external servers
    - ✅ Data deleted when you close your browser
    - ✅ No data stored or logged anywhere
    - ✅ Complete privacy guaranteed
    """)

    # ============ SIDEBAR ============
    st.sidebar.title("🔮 Navigation")

    sections = [
        "📊 Data Upload",
        "🔍 Data Explorer",
        "⚙️ Feature Engineering",
        "📈 Optimal Clusters",
        "🎯 Segmentation",
        "💼 Business Intelligence"
    ]

    selected_section = st.sidebar.radio("", sections)

    # ============ FILE UPLOAD ============
    st.sidebar.markdown("---")
    st.sidebar.subheader("📁 Upload Data")

    st.sidebar.info("""
    **Supported Formats:**
    - 📊 CSV (.csv)
    - 📗 Excel (.xlsx, .xls)
    - 📕 PDF (.pdf) - Tables only
    """)

    uploaded_file = st.sidebar.file_uploader(
        "Choose a file",
        type=['csv', 'xlsx', 'xls', 'pdf'],
        help="Upload customer data in CSV, Excel, or PDF format"
    )

    if uploaded_file:
        file_icon = get_file_icon(uploaded_file.name.split('.')[-1])
        st.sidebar.success(f"{file_icon} {uploaded_file.name}")
        st.sidebar.info(f"📏 {uploaded_file.size / 1024:.1f} KB")

    # Cache control
    st.sidebar.markdown("---")
    if st.sidebar.button("🗑️ Clear Cache"):
        st.cache_data.clear()
        st.sidebar.success("✅ Cache cleared!")
        st.rerun()

    # ============ LOAD DATA ============
    with st.spinner("🔮 Loading data with ORACELIX intelligence..."):
        df = load_data(uploaded_file)

    if df is None:
        st.stop()

    # Data info in sidebar
    st.sidebar.markdown("---")
    st.sidebar.metric("📊 Records", f"{len(df):,}")
    st.sidebar.metric("📋 Features", len(df.columns))
    st.sidebar.metric("💾 Memory", f"{df.memory_usage(deep=True).sum()/1024/1024:.2f} MB")

    # ============ SECTIONS ============

    if selected_section == "📊 Data Upload":
        st.markdown('<h2 class="section-header">📊 Data Upload & Overview</h2>', unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", f"{len(df):,}", delta="Active")
        with col2:
            st.metric("Features", len(df.columns))
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        with col4:
            st.metric("Data Quality", f"{(1 - df.isnull().sum().sum()/(len(df)*len(df.columns)))*100:.1f}%")

        st.subheader("📋 Data Preview")
        st.dataframe(df.head(20), use_container_width=True, height=400)

        st.subheader("📊 Data Statistics")
        st.dataframe(df.describe().T, use_container_width=True)

        # Download sample
        st.subheader("📥 Download Template")
        if st.button("Generate Sample CSV"):
            sample_data = {
                'CUST_ID': [f'CUST_{i:04d}' for i in range(1, 101)],
                'BALANCE': np.random.normal(1500, 500, 100),
                'PURCHASES': np.random.normal(2000, 800, 100),
                'CASH_ADVANCE': np.random.normal(500, 200, 100),
                'CREDIT_LIMIT': np.random.normal(5000, 1500, 100),
                'TENURE': np.random.randint(12, 60, 100)
            }
            sample_df = pd.DataFrame(sample_data)
            csv = sample_df.to_csv(index=False)

            st.download_button(
                label="📥 Download Sample CSV",
                data=csv,
                file_name="oracelix_sample_data.csv",
                mime="text/csv"
            )

    elif selected_section == "🔍 Data Explorer":
        st.markdown('<h2 class="section-header">🔍 Data Explorer</h2>', unsafe_allow_html=True)

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Remove ID columns
        exclude_cols = ['CUST_ID', 'SEGMENT_TAG']
        for col in exclude_cols:
            if col in numeric_cols:
                numeric_cols.remove(col)

        if len(numeric_cols) >= 2:
            st.subheader("📊 Correlation Matrix")
            corr_matrix = df[numeric_cols].corr()

            fig = px.imshow(
                corr_matrix,
                text_auto='.2f',
                aspect='auto',
                color_continuous_scale='RdBu_r',
                title='Feature Correlation Heatmap'
            )
            fig.update_layout(height=700)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("📈 Feature Distributions")
            col1, col2 = st.columns(2)

            with col1:
                feat1 = st.selectbox("Select Feature 1:", numeric_cols, index=0)
            with col2:
                feat2 = st.selectbox("Select Feature 2:", numeric_cols, index=min(1, len(numeric_cols)-1))

            col1, col2 = st.columns(2)

            with col1:
                fig1 = px.histogram(df, x=feat1, nbins=50, title=f'Distribution: {feat1}')
                fig1.add_vline(x=df[feat1].mean(), line_dash="dash", line_color="red")
                st.plotly_chart(fig1, use_container_width=True)

            with col2:
                fig2 = px.histogram(df, x=feat2, nbins=50, title=f'Distribution: {feat2}')
                fig2.add_vline(x=df[feat2].mean(), line_dash="dash", line_color="red")
                st.plotly_chart(fig2, use_container_width=True)

    elif selected_section == "⚙️ Feature Engineering":
        st.markdown('<h2 class="section-header">⚙️ Feature Engineering</h2>', unsafe_allow_html=True)

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['CUST_ID', 'SEGMENT_TAG']
        for col in exclude_cols:
            if col in numeric_cols:
                numeric_cols.remove(col)

        st.subheader("🎯 Select Features for Clustering")

        # Recommended features
        recommended = [
            'BALANCE', 'PURCHASES', 'CASH_ADVANCE',
            'CREDIT_LIMIT', 'PAYMENTS', 'TENURE'
        ]
        recommended = [f for f in recommended if f in numeric_cols]

        selected_features = st.multiselect(
            "Choose features:",
            numeric_cols,
            default=recommended if recommended else numeric_cols[:5]
        )

        if len(selected_features) < 2:
            st.warning("⚠️ Please select at least 2 features")
            st.stop()

        st.success(f"✅ Selected {len(selected_features)} features")

        # Preprocess
        X, X_scaled, scaler = preprocess_data(df, selected_features)

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Before Scaling:**")
            st.dataframe(X.describe().T, use_container_width=True)

        with col2:
            st.write("**After Scaling:**")
            X_scaled_df = pd.DataFrame(X_scaled, columns=selected_features)
            st.dataframe(X_scaled_df.describe().T, use_container_width=True)

        # PCA Visualization
        st.subheader("🔮 PCA Projection")
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        pca_df = pd.DataFrame({
            'PC1': X_pca[:, 0],
            'PC2': X_pca[:, 1]
        })

        fig = px.scatter(
            pca_df, x='PC1', y='PC2',
            title=f'PCA 2D Projection (Variance: {sum(pca.explained_variance_ratio_)*100:.1f}%)'
        )
        st.plotly_chart(fig, use_container_width=True)

        # Store in session state
        st.session_state['selected_features'] = selected_features
        st.session_state['X_scaled'] = X_scaled
        st.session_state['X'] = X
        st.session_state['scaler'] = scaler

    elif selected_section == "📈 Optimal Clusters":
        st.markdown('<h2 class="section-header">📈 Optimal Cluster Detection</h2>', unsafe_allow_html=True)

        if 'X_scaled' not in st.session_state:
            st.warning("⚠️ Please complete Feature Engineering first!")
            st.stop()

        X_scaled = st.session_state['X_scaled']

        # Show optimization info
        if X_scaled.shape[0] > 5000:
            st.info(f"""
            🔮 **ORACELIX Optimization Active**
            - Dataset: {X_scaled.shape[0]:,} records
            - Using MiniBatch K-Means for speed
            - Intelligent sampling for accuracy
            """)

        col1, col2 = st.columns(2)
        with col1:
            k_min = st.number_input("Min Clusters:", 2, 10, 2)
        with col2:
            k_max = st.number_input("Max Clusters:", 3, 15, 6)

        if k_min >= k_max:
            st.error("❌ Min must be less than Max")
            st.stop()

        if st.button("🔮 Detect Optimal Clusters", type="primary"):
            k_range = range(int(k_min), int(k_max) + 1)
            inertias = []
            silhouette_scores = []

            progress_bar = st.progress(0)
            status_text = st.empty()
            time_text = st.empty()

            start_time = time.time()

            for i, k in enumerate(k_range):
                iter_start = time.time()
                status_text.text(f"🔮 Computing K={k}... ({i+1}/{len(k_range)})")

                kmeans, clusters = train_kmeans(X_scaled, k)
                inertias.append(kmeans.inertia_)

                # Silhouette score with sampling
                if X_scaled.shape[0] > 5000:
                    sample_size = 5000
                    indices = np.random.choice(X_scaled.shape[0], sample_size, replace=False)
                    sil_score = silhouette_score(X_scaled[indices], clusters[indices])
                else:
                    sil_score = silhouette_score(X_scaled, clusters)

                silhouette_scores.append(sil_score)

                progress_bar.progress((i + 1) / len(k_range))

                iter_time = time.time() - iter_start
                remaining = (len(k_range) - i - 1) * iter_time
                time_text.text(f"⏱️ Time remaining: {remaining:.1f}s")

            total_time = time.time() - start_time
            progress_bar.empty()
            status_text.empty()
            time_text.empty()

            st.success(f"✅ Completed in {total_time:.1f}s!")

            # Plot results
            col1, col2 = st.columns(2)

            with col1:
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(
                    x=list(k_range), y=inertias,
                    mode='lines+markers',
                    marker=dict(size=10, color='#667eea'),
                    line=dict(width=3)
                ))
                fig1.update_layout(
                    title='Elbow Method: Inertia vs K',
                    xaxis_title='Clusters (K)',
                    yaxis_title='Inertia',
                    height=400
                )
                st.plotly_chart(fig1, use_container_width=True)

            with col2:
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=list(k_range), y=silhouette_scores,
                    mode='lines+markers',
                    marker=dict(size=10, color='#FFD700'),
                    line=dict(width=3)
                ))
                fig2.update_layout(
                    title='Silhouette Score vs K',
                    xaxis_title='Clusters (K)',
                    yaxis_title='Silhouette Score',
                    height=400
                )
                st.plotly_chart(fig2, use_container_width=True)

            # Recommendation
            optimal_k = list(k_range)[np.argmax(silhouette_scores)]

            st.subheader("🎯 Recommendation")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Optimal K (Silhouette)", optimal_k)
            with col2:
                st.metric("Best Silhouette Score", f"{max(silhouette_scores):.3f}")
            with col3:
                final_k = st.number_input("Select K:", int(k_min), int(k_max), int(optimal_k))

            st.session_state['optimal_k'] = final_k
            st.session_state['inertias'] = inertias
            st.session_state['silhouette_scores'] = silhouette_scores

        # Middle Banner after Optimal K
        middle_banner = load_image(MIDDLE_BANNER)
        if middle_banner:
            st.markdown(f"""
            <div style="width: 100%; margin: 2rem 0;">
                <img src="{middle_banner}" style="width: 100%; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.2);">
            </div>
            """, unsafe_allow_html=True)

    elif selected_section == "🎯 Segmentation":
        st.markdown('<h2 class="section-header">🎯 Customer Segmentation</h2>', unsafe_allow_html=True)

        if 'X_scaled' not in st.session_state or 'optimal_k' not in st.session_state:
            st.warning("⚠️ Please complete previous sections first!")
            st.stop()

        X_scaled = st.session_state['X_scaled']
        X = st.session_state['X']
        selected_features = st.session_state['selected_features']
        optimal_k = st.session_state['optimal_k']

        st.subheader(f"🔮 Training Model (K={optimal_k})")

        kmeans, clusters = train_kmeans(X_scaled, optimal_k)

        df_clustered = df.copy()
        df_clustered['Cluster'] = clusters

        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Clusters", optimal_k)
        with col2:
            sil_score = silhouette_score(X_scaled, clusters)
            st.metric("Silhouette Score", f"{sil_score:.3f}")
        with col3:
            dbi = davies_bouldin_score(X_scaled, clusters)
            st.metric("Davies-Bouldin", f"{dbi:.3f}")

        st.success("✅ Segmentation complete!")

        # Cluster distribution
        st.subheader("📊 Cluster Distribution")

        cluster_counts = df_clustered['Cluster'].value_counts().sort_index()

        fig = px.pie(
            values=cluster_counts.values,
            names=[f'Segment {i}' for i in cluster_counts.index],
            title='Customer Segments',
            hole=0.4
        )
        st.plotly_chart(fig, use_container_width=True)

        # PCA Visualization with clusters
        st.subheader("🔮 Segment Visualization")

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        pca_df = pd.DataFrame({
            'PC1': X_pca[:, 0],
            'PC2': X_pca[:, 1],
            'Segment': [f'Segment {c}' for c in clusters]
        })

        fig = px.scatter(
            pca_df, x='PC1', y='PC2', color='Segment',
            title=f'Customer Segments in PCA Space',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig, use_container_width=True)

        # Cluster profiles
        st.subheader("📋 Segment Profiles")
        cluster_profiles = df_clustered.groupby('Cluster')[selected_features].mean()
        cluster_profiles['Size'] = cluster_counts.values

        st.dataframe(cluster_profiles.round(2), use_container_width=True)

        # Store results
        st.session_state['df_clustered'] = df_clustered
        st.session_state['cluster_profiles'] = cluster_profiles
        st.session_state['kmeans'] = kmeans

    elif selected_section == "💼 Business Intelligence":
        st.markdown('<h2 class="section-header">💼 Business Intelligence</h2>', unsafe_allow_html=True)

        if 'df_clustered' not in st.session_state:
            st.warning("⚠️ Please complete Segmentation first!")
            st.stop()

        df_clustered = st.session_state['df_clustered']
        cluster_profiles = st.session_state['cluster_profiles']
        selected_features = st.session_state['selected_features']

        st.subheader("📊 Executive Dashboard")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Customers", f"{len(df_clustered):,}")
        with col2:
            st.metric("Segments", df_clustered['Cluster'].nunique())
        with col3:
            if 'BALANCE' in df_clustered.columns:
                avg_balance = df_clustered['BALANCE'].mean()
                st.metric("Avg Balance", f"${avg_balance:,.0f}")
        with col4:
            if 'PURCHASES' in df_clustered.columns:
                avg_purchases = df_clustered['PURCHASES'].mean()
                st.metric("Avg Purchases", f"${avg_purchases:,.0f}")

        st.subheader("🎯 Segment Insights")

        for i in range(df_clustered['Cluster'].nunique()):
            with st.expander(f"🔮 Segment {i} ({len(df_clustered[df_clustered['Cluster']==i]):,} customers)"):
                segment_data = df_clustered[df_clustered['Cluster'] == i]

                col1, col2 = st.columns([1, 2])

                with col1:
                    st.write("**Key Metrics:**")
                    for feat in selected_features[:5]:
                        if feat in segment_data.columns:
                            val = segment_data[feat].mean()
                            st.metric(feat, f"{val:.2f}")

                with col2:
                    st.write("**Sample Customers:**")
                    sample = segment_data[selected_features].head(5)
                    st.dataframe(sample, use_container_width=True)

        # Export
        st.subheader("📥 Export Results")

        col1, col2 = st.columns(2)

        with col1:
            csv = df_clustered.to_csv(index=False)
            st.download_button(
                label="📊 Download Segmented Data (CSV)",
                data=csv,
                file_name="oracelix_segments.csv",
                mime="text/csv"
            )

        with col2:
            profiles_csv = cluster_profiles.to_csv()
            st.download_button(
                label="📋 Download Profiles (CSV)",
                data=profiles_csv,
                file_name="oracelix_profiles.csv",
                mime="text/csv"
            )

    # ============ FOOTER BANNER ============
    st.markdown("---")

    footer_banner = load_image(FOOTER_BANNER)
    if footer_banner:
        st.markdown(f"""
        <div style="width: 100%; margin-top: 2rem;">
            <img src="{footer_banner}" style="width: 100%; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.3);">
        </div>
        """, unsafe_allow_html=True)

    # Additional footer text
    st.markdown(f"""
    <div style="text-align: center; color: #888; padding: 1rem;">
        <p style="font-size: 0.9rem; margin-top: 1rem;">
            🔒 Your data is 100% private and secure | Powered by Advanced ML | Optimized for Large Datasets
        </p>
        <p style="font-size: 0.8rem; color: #666;">
            © 2025 {BRAND_NAME} - All Rights Reserved
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
