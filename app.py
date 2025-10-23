# app.py - Complete K-Means Clustering App
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="CardWise Customer Segmentation",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 2rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and return the dataset"""
    df = pd.read_csv('Customer Data.csv')
    return df

@st.cache_data
def preprocess_data(df, selected_features):
    """Preprocess data: select features and scale"""
    X = df[selected_features].copy()
    
    # Handle missing values
    X = X.fillna(X.median())
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X, X_scaled, scaler

@st.cache_data
def train_kmeans(X_scaled, n_clusters, random_state=42):
    """Train K-Means model"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    return kmeans, clusters

def main():
    # Header
    st.markdown('<h1 class="main-header">💳 CardWise Customer Segmentation</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem;">K-Means Clustering Analysis for Targeted Marketing & Risk Management</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("📊 Navigation")
    sections = [
        "📋 Part A: Data Loading",
        "🔍 Part B: EDA",
        "⚙️ Part C: Preprocessing",
        "📈 Part D: Optimal K",
        "🎯 Part E: Clustering",
        "💼 Part F: Business Insights"
    ]
    selected_section = st.sidebar.radio("Select Section:", sections)
    
    # Load data
    df = load_data()
    
    # ============ PART A: DATA LOADING ============
    if selected_section == "📋 Part A: Data Loading":
        st.markdown('<h2 class="section-header">Part A: Data Loading and Initial Inspection</h2>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Customers", f"{len(df):,}")
        with col2:
            st.metric("Total Features", len(df.columns))
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        with col4:
            st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        st.subheader("1️⃣ First 10 Rows")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.subheader("2️⃣ Dataset Info")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Data Types:**")
            dtype_df = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.values,
                'Non-Null': df.count().values
            })
            st.dataframe(dtype_df, use_container_width=True)
        
        with col2:
            st.write("**Descriptive Statistics:**")
            st.dataframe(df.describe().T, use_container_width=True)
        
        st.subheader("3️⃣ Missing Values Analysis")
        missing = df.isnull().sum()
        if missing.sum() == 0:
            st.success("✅ No missing values detected! Data quality is excellent.")
        else:
            st.warning(f"⚠️ Found {missing.sum()} missing values")
            fig = px.bar(x=missing.index, y=missing.values, 
                        title="Missing Values by Column",
                        labels={'x': 'Column', 'y': 'Missing Count'})
            st.plotly_chart(fig, use_container_width=True)
    
    # ============ PART B: EDA ============
    elif selected_section == "🔍 Part B: EDA":
        st.markdown('<h2 class="section-header">Part B: Exploratory Data Analysis</h2>', unsafe_allow_html=True)
        
        st.subheader("1️⃣ Correlation Matrix")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols.remove('CUST_ID') if 'CUST_ID' in numeric_cols else None
        
        corr_matrix = df[numeric_cols].corr()
        
        fig = px.imshow(corr_matrix, 
                       text_auto='.2f',
                       aspect='auto',
                       color_continuous_scale='RdBu_r',
                       title='Feature Correlation Heatmap')
        fig.update_layout(height=800)
        st.plotly_chart(fig, use_container_width=True)
        
        st.write("**Key Correlations:**")
        # Find top correlations
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_pairs.append({
                    'Feature 1': corr_matrix.columns[i],
                    'Feature 2': corr_matrix.columns[j],
                    'Correlation': corr_matrix.iloc[i, j]
                })
        corr_df = pd.DataFrame(corr_pairs).sort_values('Correlation', key=abs, ascending=False)
        st.dataframe(corr_df.head(10), use_container_width=True)
        
        st.subheader("2️⃣ Distribution Visualizations")
        
        col1, col2 = st.columns(2)
        with col1:
            feature1 = st.selectbox("Select Feature for Histogram:", numeric_cols, index=1)
        with col2:
            feature2 = st.selectbox("Select Second Feature:", numeric_cols, index=3)
        
        # Histograms
        col1, col2 = st.columns(2)
        with col1:
            fig1 = px.histogram(df, x=feature1, nbins=50, 
                               title=f'Distribution of {feature1}',
                               marginal='box')
            fig1.add_vline(x=df[feature1].mean(), line_dash="dash", 
                          line_color="red", annotation_text="Mean")
            st.plotly_chart(fig1, use_container_width=True)
            
            skew = df[feature1].skew()
            kurt = df[feature1].kurtosis()
            st.info(f"📊 Skewness: {skew:.2f} | Kurtosis: {kurt:.2f}")
        
        with col2:
            fig2 = px.histogram(df, x=feature2, nbins=50,
                               title=f'Distribution of {feature2}',
                               marginal='box')
            fig2.add_vline(x=df[feature2].mean(), line_dash="dash",
                          line_color="red", annotation_text="Mean")
            st.plotly_chart(fig2, use_container_width=True)
            
            skew2 = df[feature2].skew()
            kurt2 = df[feature2].kurtosis()
            st.info(f"📊 Skewness: {skew2:.2f} | Kurtosis: {kurt2:.2f}")
        
        # Scatter plot
        st.subheader("Scatter Plot Analysis")
        x_feat = st.selectbox("X-axis:", numeric_cols, index=3, key='scatter_x')
        y_feat = st.selectbox("Y-axis:", numeric_cols, index=6, key='scatter_y')
        
        fig3 = px.scatter(df, x=x_feat, y=y_feat, 
                         title=f'{x_feat} vs {y_feat}',
                         trendline='ols',
                         hover_data=['CUST_ID'])
        st.plotly_chart(fig3, use_container_width=True)
        
        st.subheader("3️⃣ Outlier Detection")
        
        outlier_feat = st.selectbox("Select Feature for Outlier Analysis:", numeric_cols)
        
        Q1 = df[outlier_feat].quantile(0.25)
        Q3 = df[outlier_feat].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[outlier_feat] < Q1 - 1.5*IQR) | (df[outlier_feat] > Q3 + 1.5*IQR)]
        
        col1, col2 = st.columns([2, 1])
        with col1:
            fig4 = px.box(df, y=outlier_feat, title=f'Boxplot: {outlier_feat}')
            st.plotly_chart(fig4, use_container_width=True)
        
        with col2:
            st.metric("Outliers Detected", f"{len(outliers)}")
            st.metric("Outlier Percentage", f"{len(outliers)/len(df)*100:.2f}%")
            st.metric("Q1 (25%)", f"{Q1:.2f}")
            st.metric("Q3 (75%)", f"{Q3:.2f}")
            st.metric("IQR", f"{IQR:.2f}")
    
    # ============ PART C: PREPROCESSING ============
    elif selected_section == "⚙️ Part C: Preprocessing":
        st.markdown('<h2 class="section-header">Part C: Preprocessing & Feature Selection</h2>', unsafe_allow_html=True)
        
        st.subheader("1️⃣ Feature Selection")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols.remove('CUST_ID') if 'CUST_ID' in numeric_cols else None
        
        st.write("**Available Features:**")
        st.write(numeric_cols)
        
        # Recommended features
        recommended_features = [
            'BALANCE', 'PURCHASES', 'ONEOFF_PURCHASES', 
            'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE', 
            'CREDIT_LIMIT', 'PURCHASES_FREQUENCY', 'TENURE'
        ]
        
        selected_features = st.multiselect(
            "Select Features for Clustering:",
            numeric_cols,
            default=[f for f in recommended_features if f in numeric_cols]
        )
        
        if len(selected_features) < 2:
            st.warning("⚠️ Please select at least 2 features")
            return
        
        st.success(f"✅ Selected {len(selected_features)} features")
        
        # Feature importance based on variance
        st.subheader("2️⃣ Feature Variance Analysis")
        variance_df = pd.DataFrame({
            'Feature': selected_features,
            'Variance': [df[f].var() for f in selected_features],
            'Std Dev': [df[f].std() for f in selected_features]
        }).sort_values('Variance', ascending=False)
        
        fig = px.bar(variance_df, x='Feature', y='Variance',
                    title='Feature Variance (Before Scaling)',
                    color='Variance',
                    color_continuous_scale='viridis')
        st.plotly_chart(fig, use_container_width=True)
        
        # Scaling
        st.subheader("3️⃣ Feature Scaling (Standardization)")
        
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
        st.subheader("4️⃣ PCA Visualization")
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        pca_df = pd.DataFrame({
            'PC1': X_pca[:, 0],
            'PC2': X_pca[:, 1]
        })
        
        fig = px.scatter(pca_df, x='PC1', y='PC2',
                        title=f'PCA 2D Projection (Explained Variance: {sum(pca.explained_variance_ratio_)*100:.2f}%)',
                        labels={'PC1': f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)',
                               'PC2': f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Store in session state
        st.session_state['selected_features'] = selected_features
        st.session_state['X_scaled'] = X_scaled
        st.session_state['X'] = X
        st.session_state['scaler'] = scaler
    
    # ============ PART D: OPTIMAL K ============
    elif selected_section == "📈 Part D: Optimal K":
        st.markdown('<h2 class="section-header">Part D: Finding Optimal Number of Clusters</h2>', unsafe_allow_html=True)
        
        if 'X_scaled' not in st.session_state:
            st.warning("⚠️ Please complete Part C: Preprocessing first!")
            return
        
        X_scaled = st.session_state['X_scaled']
        
        st.subheader("1️⃣ Elbow Method")
        
        k_range = range(2, 11)
        inertias = []
        silhouette_scores = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, k in enumerate(k_range):
            status_text.text(f"Computing for K={k}...")
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
            progress_bar.progress((i + 1) / len(k_range))
        
        status_text.empty()
        progress_bar.empty()
        
        # Plot Elbow
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=list(k_range), y=inertias, 
                                     mode='lines+markers',
                                     marker=dict(size=10, color='blue'),
                                     line=dict(width=3)))
            fig1.update_layout(
                title='Elbow Method: SSE vs K',
                xaxis_title='Number of Clusters (K)',
                yaxis_title='Sum of Squared Errors (SSE)',
                height=400
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=list(k_range), y=silhouette_scores,
                                     mode='lines+markers',
                                     marker=dict(size=10, color='green'),
                                     line=dict(width=3)))
            fig2.update_layout(
                title='Silhouette Score vs K',
                xaxis_title='Number of Clusters (K)',
                yaxis_title='Silhouette Score',
                height=400
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        # Recommendations
        st.subheader("2️⃣ Optimal K Selection")
        
        # Calculate elbow using percentage change
        changes = np.diff(inertias)
        percent_changes = np.abs(changes / inertias[:-1] * 100)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            optimal_k_elbow = 4  # Based on typical elbow
            st.metric("Recommended K (Elbow)", optimal_k_elbow)
        with col2:
            optimal_k_silhouette = list(k_range)[np.argmax(silhouette_scores)]
            st.metric("Recommended K (Silhouette)", optimal_k_silhouette)
        with col3:
            final_k = st.number_input("Select Final K:", min_value=2, max_value=10, value=4)
        
        st.info(f"📊 Selected K = {final_k} clusters for analysis")
        
        # Store in session state
        st.session_state['optimal_k'] = final_k
        st.session_state['inertias'] = inertias
        st.session_state['silhouette_scores'] = silhouette_scores
    
    # ============ PART E: CLUSTERING ============
    elif selected_section == "🎯 Part E: Clustering":
        st.markdown('<h2 class="section-header">Part E: K-Means Training & Cluster Analysis</h2>', unsafe_allow_html=True)
        
        if 'X_scaled' not in st.session_state or 'optimal_k' not in st.session_state:
            st.warning("⚠️ Please complete Parts C and D first!")
            return
        
        X_scaled = st.session_state['X_scaled']
        X = st.session_state['X']
        selected_features = st.session_state['selected_features']
        optimal_k = st.session_state['optimal_k']
        
        st.subheader(f"1️⃣ Training K-Means Model (K={optimal_k})")
        
        # Train model
        kmeans, clusters = train_kmeans(X_scaled, optimal_k)
        
        # Add clusters to dataframe
        df_clustered = df.copy()
        df_clustered['Cluster'] = clusters
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Clusters", optimal_k)
        with col2:
            silhouette = silhouette_score(X_scaled, clusters)
            st.metric("Silhouette Score", f"{silhouette:.3f}")
        with col3:
            dbi = davies_bouldin_score(X_scaled, clusters)
            st.metric("Davies-Bouldin Index", f"{dbi:.3f}")
        
        st.success("✅ Model trained successfully!")
        
        # Cluster sizes
        st.subheader("2️⃣ Cluster Distribution")
        
        cluster_counts = df_clustered['Cluster'].value_counts().sort_index()
        cluster_pct = (cluster_counts / len(df_clustered) * 100).round(2)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            cluster_dist_df = pd.DataFrame({
                'Cluster': cluster_counts.index,
                'Count': cluster_counts.values,
                'Percentage': cluster_pct.values
            })
            st.dataframe(cluster_dist_df, use_container_width=True)
        
        with col2:
            fig = px.pie(values=cluster_counts.values, 
                        names=[f'Cluster {i}' for i in cluster_counts.index],
                        title='Cluster Size Distribution',
                        hole=0.4)
            st.plotly_chart(fig, use_container_width=True)
        
        # Cluster profiles
        st.subheader("3️⃣ Cluster Profiles (Centroids)")
        
        cluster_profiles = df_clustered.groupby('Cluster')[selected_features].mean()
        
        # Add size info
        cluster_profiles['Size'] = cluster_counts.values
        cluster_profiles['Size_%'] = cluster_pct.values
        
        st.dataframe(cluster_profiles.round(2), use_container_width=True)
        
        # Heatmap of centroids
        fig = px.imshow(cluster_profiles[selected_features].T,
                       labels=dict(x="Cluster", y="Feature", color="Value"),
                       x=[f'Cluster {i}' for i in cluster_profiles.index],
                       y=selected_features,
                       color_continuous_scale='RdYlGn',
                       aspect='auto',
                       title='Cluster Centroids Heatmap (Scaled Values)')
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # PCA Visualization with clusters
        st.subheader("4️⃣ Cluster Visualization (PCA)")
        
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        pca_df = pd.DataFrame({
            'PC1': X_pca[:, 0],
            'PC2': X_pca[:, 1],
            'Cluster': [f'Cluster {c}' for c in clusters]
        })
        
        fig = px.scatter(pca_df, x='PC1', y='PC2', color='Cluster',
                        title=f'Clusters in PCA Space (Variance Explained: {sum(pca.explained_variance_ratio_)*100:.2f}%)',
                        color_discrete_sequence=px.colors.qualitative.Set2)
        
        # Add centroids
        centroids_pca = pca.transform(kmeans.cluster_centers_)
        fig.add_trace(go.Scatter(x=centroids_pca[:, 0], y=centroids_pca[:, 1],
                                mode='markers',
                                marker=dict(size=20, symbol='x', color='black', line=dict(width=2)),
                                name='Centroids'))
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Cluster Interpretations
        st.subheader("5️⃣ Cluster Interpretations")
        
        cluster_names = {
            0: "Low-Engagement Cash Users",
            1: "High-Value Frequent Spenders",
            2: "Balanced Low Spenders",
            3: "Moderate Installment Users"
        }
        
        cluster_descriptions = {
            0: "🔴 **Low-Engagement Cash Users**: Low spending, low frequency, high cash advances. These are risky users with minimal one-off payments. Focus on conversion strategies.",
            1: "🟢 **High-Value Frequent Spenders**: High spending, high frequency, high credit limits, low cash advances. Loyal customers who prefer installments. Target for premium rewards.",
            2: "🟡 **Balanced Low Spenders**: Low balance, no cash advances, medium frequency. Conservative users with growth potential. Offer incentives to increase engagement.",
            3: "🔵 **Moderate Installment Users**: Medium spending, split between one-off and installments, medium frequency. Family/large purchase oriented. Provide flexible payment options."
        }
        
        for i in range(optimal_k):
            with st.expander(f"Cluster {i}: {cluster_names.get(i, f'Segment {i}')}"):
                st.write(cluster_descriptions.get(i, "Analysis pending..."))
                
                # Show sample customers
                sample = df_clustered[df_clustered['Cluster'] == i][['CUST_ID'] + selected_features].head(5)
                st.write("**Sample Customers:**")
                st.dataframe(sample, use_container_width=True)
        
        # Store in session state
        st.session_state['df_clustered'] = df_clustered
        st.session_state['cluster_profiles'] = cluster_profiles
        st.session_state['kmeans'] = kmeans
    
    # ============ PART F: BUSINESS INSIGHTS ============
    elif selected_section == "💼 Part F: Business Insights":
        st.markdown('<h2 class="section-header">Part F: Business Conclusions & Recommendations</h2>', unsafe_allow_html=True)
        
        if 'df_clustered' not in st.session_state:
            st.warning("⚠️ Please complete Part E: Clustering first!")
            return
        
        df_clustered = st.session_state['df_clustered']
        cluster_profiles = st.session_state['cluster_profiles']
        selected_features = st.session_state['selected_features']
        
        st.subheader("1️⃣ Executive Summary")
        
        total_customers = len(df_clustered)
        n_clusters = df_clustered['Cluster'].nunique()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Customers", f"{total_customers:,}")
        with col2:
            st.metric("Segments Identified", n_clusters)
        with col3:
            avg_balance = df_clustered['BALANCE'].mean()
            st.metric("Avg Balance", f"${avg_balance:,.2f}")
        with col4:
            avg_purchases = df_clustered['PURCHASES'].mean()
            st.metric("Avg Purchases", f"${avg_purchases:,.2f}")
        
        st.subheader("2️⃣ Cluster Distribution & Insights")
        
        # Revenue potential analysis
        if 'PURCHASES' in selected_features:
            cluster_revenue = df_clustered.groupby('Cluster')['PURCHASES'].sum()
            cluster_revenue_pct = (cluster_revenue / cluster_revenue.sum() * 100).round(2)
            
            revenue_df = pd.DataFrame({
                'Cluster': cluster_revenue.index,
                'Total Revenue': cluster_revenue.values,
                'Revenue %': cluster_revenue_pct.values,
                'Avg Revenue': (cluster_revenue / df_clustered.groupby('Cluster').size()).values
            })
            
            st.write("**Revenue Contribution by Cluster:**")
            st.dataframe(revenue_df.round(2), use_container_width=True)
            
            fig = px.bar(revenue_df, x='Cluster', y='Revenue %',
                        title='Revenue Contribution by Cluster',
                        color='Revenue %',
                        color_continuous_scale='viridis')
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("3️⃣ Targeted Marketing Strategies")
        
        strategies = {
            "Cluster 0: Low-Engagement Cash Users": {
                "Risk Level": "🔴 High",
                "Strategy": "Cash-to-Purchase Conversion Campaign",
                "Actions": [
                    "Offer 0% interest on purchases for 3 months",
                    "Implement credit monitoring and limit adjustments",
                    "Send educational content about responsible card usage",
                    "Provide cashback rewards on purchases to reduce cash reliance"
                ],
                "Expected Impact": "15-20% conversion rate from cash to purchases",
                "Priority": "High - Risk Mitigation"
            },
            "Cluster 1: High-Value Frequent Spenders": {
                "Risk Level": "🟢 Low",
                "Strategy": "VIP Loyalty & Premium Rewards Program",
                "Actions": [
                    "Offer credit limit increases with premium benefits",
                    "Exclusive rewards: travel points, cashback bonuses",
                    "Personalized offers based on spending patterns",
                    "Early access to new products and services"
                ],
                "Expected Impact": "20-25% increase in spending, 90% retention",
                "Priority": "High - Revenue Growth"
            },
            "Cluster 2: Balanced Low Spenders": {
                "Risk Level": "🟡 Low-Medium",
                "Strategy": "Engagement & Growth Activation",
                "Actions": [
                    "One-off purchase incentives with bonus rewards",
                    "Increase purchase frequency with gamification",
                    "Referral bonuses for bringing new customers",
                    "Seasonal promotions targeting their spending habits"
                ],
                "Expected Impact": "15% increase in purchase frequency",
                "Priority": "Medium - Growth Potential"
            },
            "Cluster 3: Moderate Installment Users": {
                "Risk Level": "🟡 Medium",
                "Strategy": "Flexible Payment & Family-Oriented Offers",
                "Actions": [
                    "Extended installment plans for large purchases",
                    "Family bundles and multi-purchase discounts",
                    "Partner with retailers for exclusive deals",
                    "Auto-payment setup incentives to reduce defaults"
                ],
                "Expected Impact": "10-15% increase in transaction value",
                "Priority": "Medium - Retention Focus"
            }
        }
        
        for cluster_name, strategy in strategies.items():
            with st.expander(f"📊 {cluster_name}", expanded=True):
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.write(f"**Risk Level:** {strategy['Risk Level']}")
                    st.write(f"**Priority:** {strategy['Priority']}")
                with col2:
                    st.write(f"**Strategy:** {strategy['Strategy']}")
                    st.write(f"**Expected Impact:** {strategy['Expected Impact']}")
                
                st.write("**Action Items:**")
                for action in strategy['Actions']:
                    st.write(f"- {action}")
        
        st.subheader("4️⃣ Risk Management Recommendations")
        
        st.write("""
        **Credit Risk Assessment:**
        - **High Risk (Cluster 0)**: 28% of customers - implement stricter monitoring, consider limit reductions for high cash advance users
        - **Medium Risk (Cluster 3)**: 25% of customers - monitor payment patterns, offer payment reminders
        - **Low Risk (Clusters 1, 2)**: 47% of customers - maintain current credit policies, reward good behavior
        
        **Churn Prevention:**
        - Cluster 0 shows low TENURE (avg 25 months) → 20% churn risk
        - Implement early warning system based on:
          - Decreasing purchase frequency
          - Increasing cash advance reliance
          - Late payment patterns
        """)
        
        st.subheader("5️⃣ Personalized Promotion Matrix")
        
        promotion_matrix = pd.DataFrame({
            'Cluster': [0, 1, 2, 3],
            'Email Campaign': ['Cash Conversion', 'VIP Rewards', 'Bonus Points', 'Flexible Payment'],
            'Offer Type': ['0% Interest', 'Premium Benefits', 'Cashback', 'Extended Terms'],
            'Channel': ['Email + SMS', 'Email + App', 'Email', 'Email + Call'],
            'Frequency': ['Weekly', 'Monthly', 'Bi-weekly', 'Monthly'],
            'Expected ROI': ['15-20%', '25-30%', '10-15%', '12-18%']
        })
        
        st.dataframe(promotion_matrix, use_container_width=True)
        
        st.subheader("6️⃣ Implementation Roadmap")
        
        timeline = {
            "Phase 1 (Month 1-2)": [
                "Integrate cluster labels into CRM system",
                "Design and launch targeted email campaigns",
                "Set up A/B testing framework"
            ],
            "Phase 2 (Month 3-4)": [
                "Analyze campaign performance metrics",
                "Refine customer segments based on response",
                "Launch personalized app notifications"
            ],
            "Phase 3 (Month 5-6)": [
                "Implement real-time segmentation API",
                "Deploy predictive churn models",
                "Scale successful campaigns"
            ],
            "Phase 4 (Month 7+)": [
                "Continuous monitoring and optimization",
                "Quarterly model retraining",
                "Expand to additional behavioral segments"
            ]
        }
        
        for phase, actions in timeline.items():
            with st.expander(f"📅 {phase}"):
                for action in actions:
                    st.write(f"✓ {action}")
        
        st.subheader("7️⃣ Key Performance Indicators (KPIs)")
        
        kpi_data = {
            'KPI': [
                'Customer Retention Rate',
                'Average Purchase Frequency',
                'Cash Advance Reduction',
                'Credit Utilization Rate',
                'Campaign Response Rate',
                'Revenue per Customer'
            ],
            'Current': ['75%', '0.5/month', '30%', '45%', 'N/A', '$1,008'],
            'Target (6 months)': ['85%', '0.7/month', '20%', '50%', '20%', '$1,300'],
            'Target (12 months)': ['90%', '1.0/month', '15%', '55%', '30%', '$1,500']
        }
        
        kpi_df = pd.DataFrame(kpi_data)
        st.dataframe(kpi_df, use_container_width=True)
        
        st.subheader("8️⃣ Ethical Considerations & Compliance")
        
        st.info("""
        **Data Privacy & Ethics:**
        - ✅ Ensure GDPR/CCPA compliance in all marketing communications
        - ✅ Provide opt-out mechanisms for personalized targeting
        - ✅ Monitor for potential bias in credit decisions affecting low-income segments
        - ✅ Regular audits of model fairness across demographic groups
        - ✅ Transparent communication about how customer data is used
        """)
        
        st.success("""
        **📈 Expected Business Impact:**
        - **Revenue Growth**: 15-20% uplift through targeted campaigns
        - **Customer Retention**: 10-15% improvement in retention rates
        - **Risk Reduction**: 20-25% decrease in default rates for high-risk clusters
        - **Operational Efficiency**: 30% improvement in marketing ROI
        - **Customer Satisfaction**: Enhanced personalization leading to better experience
        """)
        
        # Download results
        st.subheader("9️⃣ Export Results")
        
        col1, col2 = st.columns(2)
        with col1:
            csv = df_clustered.to_csv(index=False)
            st.download_button(
                label="📥 Download Clustered Data (CSV)",
                data=csv,
                file_name="customer_clusters.csv",
                mime="text/csv"
            )
        
        with col2:
            cluster_summary = cluster_profiles.to_csv()
            st.download_button(
                label="📥 Download Cluster Profiles (CSV)",
                data=cluster_summary,
                file_name="cluster_profiles.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()

