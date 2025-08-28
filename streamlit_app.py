import streamlit as st
import pandas as pd
import numpy as np
import asyncio
import os
import base64
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO, StringIO

# Import your existing EDA system
from app import MultiAgentEDASystem

# Page configuration
st.set_page_config(
    page_title="Multi-Agent EDA System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 1rem;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .upload-area {
        border: 2px dashed #3498db;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background-color: #f8f9fa;
        margin: 1rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'eda_results' not in st.session_state:
    st.session_state.eda_results = None
if 'current_data' not in st.session_state:
    st.session_state.current_data = None
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False

def main():
    # Main header
    st.markdown('<h1 class="main-header">🚀 Multi-Agent EDA System</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Powered by AutoGen AgentChat Framework</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown('<h3 class="sub-header">🔧 Configuration</h3>', unsafe_allow_html=True)
        
        # Model selection
        model_name = st.selectbox(
            "Select AI Model",
            ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
            index=0,
            help="Choose the OpenAI model for analysis"
        )
        
        # API Key input
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Enter your OpenAI API key"
        )
        
        # File upload section
        st.markdown('<h4 style="margin-top: 2rem;">📁 Data Upload</h4>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload your dataset for analysis"
        )
        
        if uploaded_file is not None:
            st.success(f"✅ File uploaded: {uploaded_file.name}")
            
            # Load and preview data
            try:
                data = pd.read_csv(uploaded_file)
                st.session_state.current_data = data
                
                st.markdown('<h5>📊 Data Preview</h5>', unsafe_allow_html=True)
                st.write(f"**Shape:** {data.shape[0]} rows × {data.shape[1]} columns")
                st.write(f"**Memory Usage:** {data.memory_usage(deep=True).sum() / 1024:.2f} KB")
                
                # Show first few rows
                with st.expander("View Data Sample"):
                    st.dataframe(data.head(10))
                
                # Show data info
                with st.expander("View Data Info"):
                    buffer = StringIO()
                    data.info(buf=buffer)
                    st.text(buffer.getvalue())
                
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
                return
        
        # Analysis button
        if st.session_state.current_data is not None and api_key:
            if st.button("🚀 Start EDA Analysis", type="primary", use_container_width=True):
                st.session_state.analysis_complete = False
                with st.spinner("🔄 Initializing Multi-Agent EDA System..."):
                    run_eda_analysis(api_key, model_name)
    
    # Main content area
    if st.session_state.current_data is not None:
        display_data_overview()
        
        if st.session_state.analysis_complete and st.session_state.eda_results:
            display_analysis_results()
    else:
        display_welcome_message()

def display_welcome_message():
    """Display welcome message and instructions."""
    st.markdown('<div class="upload-area">', unsafe_allow_html=True)
    st.markdown("""
    ## 📊 Welcome to Multi-Agent EDA System!
    
    This application uses advanced AI agents to perform comprehensive Exploratory Data Analysis.
    
    ### 🚀 How to get started:
    1. **Upload your CSV file** using the sidebar
    2. **Enter your OpenAI API key** in the sidebar
    3. **Click 'Start EDA Analysis'** to begin the multi-agent analysis
    4. **View results** including visualizations, insights, and reports
    
    ### 🎯 What you'll get:
    - **Data Quality Assessment** with missing value analysis
    - **Statistical Analysis** of all variables
    - **Professional Visualizations** for insights
    - **Comprehensive EDA Report** in Word format
    - **AI-Generated Insights** and recommendations
    
    ### 🔧 Supported Features:
    - CSV file upload and validation
    - Multi-agent collaborative analysis
    - Interactive visualizations
    - Professional report generation
    - Data quality assessment
    """)
    st.markdown('</div>', unsafe_allow_html=True)

def display_data_overview():
    """Display overview of the uploaded data."""
    data = st.session_state.current_data
    
    st.markdown('<h2 class="sub-header">📊 Dataset Overview</h2>', unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{data.shape[0]:,}</h3>
            <p>Total Rows</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{data.shape[1]}</h3>
            <p>Total Columns</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        missing_total = data.isnull().sum().sum()
        st.markdown(f"""
        <div class="metric-card">
            <h3>{missing_total:,}</h3>
            <p>Missing Values</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        memory_mb = data.memory_usage(deep=True).sum() / (1024 * 1024)
        st.markdown(f"""
        <div class="metric-card">
            <h3>{memory_mb:.2f}</h3>
            <p>Memory (MB)</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Data types and missing values
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<h4>📋 Data Types</h4>', unsafe_allow_html=True)
        dtype_df = pd.DataFrame({
            'Column': data.columns,
            'Data Type': data.dtypes.astype(str),
            'Non-Null Count': data.count()
        })
        st.dataframe(dtype_df, use_container_width=True)
    
    with col2:
        st.markdown('<h4>❓ Missing Values</h4>', unsafe_allow_html=True)
        missing_df = pd.DataFrame({
            'Column': data.columns,
            'Missing Count': data.isnull().sum(),
            'Missing %': (data.isnull().sum() / len(data) * 100).round(2)
        })
        missing_df = missing_df[missing_df['Missing Count'] > 0]
        if len(missing_df) > 0:
            st.dataframe(missing_df, use_container_width=True)
        else:
            st.success("✅ No missing values detected!")

def run_eda_analysis(api_key, model_name):
    """Run the EDA analysis using the multi-agent system."""
    try:
        # Set environment variable
        os.environ['OPENAI_API_KEY'] = api_key
        
        # Get the data directly from session state
        data = st.session_state.current_data
        
        # Debug info
        st.info(f"📊 Data shape: {data.shape}")
        st.info(f"📋 Columns: {', '.join(data.columns.tolist())}")
        
        # Initialize EDA system
        eda_system = MultiAgentEDASystem(model_name=model_name, api_key=api_key)
        
        # Run analysis directly with DataFrame
        with st.spinner("🔄 Running Multi-Agent EDA Analysis..."):
            # Create a new event loop for async operations
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                # Pass the DataFrame directly instead of file path
                results = loop.run_until_complete(
                    eda_system.run_eda_workflow_dataframe(data)
                )
                
                if results["status"] == "success":
                    st.session_state.eda_results = results
                    st.session_state.analysis_complete = True
                    st.success("🎉 EDA Analysis Completed Successfully!")
                    
                    # Cleanup
                    loop.run_until_complete(eda_system.cleanup())
                    
                else:
                    st.error(f"❌ Analysis failed: {results['message']}")
                    
            finally:
                loop.close()
                
    except Exception as e:
        st.error(f"❌ Error during analysis: {str(e)}")
        st.error(f"Error details: {type(e).__name__}: {str(e)}")

def display_analysis_results():
    """Display the EDA analysis results."""
    results = st.session_state.eda_results
    data = st.session_state.current_data
    
    st.markdown('<h2 class="sub-header">🎯 Analysis Results</h2>', unsafe_allow_html=True)
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Statistical Summary", 
        "📈 Visualizations", 
        "🔍 Key Insights", 
        "📝 Report Download",
        "📋 Raw Data"
    ])
    
    with tab1:
        display_statistical_summary(data, results)
    
    with tab2:
        display_visualizations(results)
    
    with tab3:
        display_key_insights(data, results)
    
    with tab4:
        display_report_download(results)
    
    with tab5:
        display_raw_data(data, results)

def display_statistical_summary(data, results):
    """Display statistical summary of the data."""
    st.markdown('<h3>📊 Statistical Summary</h3>', unsafe_allow_html=True)
    
    # Numerical variables summary
    numerical_cols = data.select_dtypes(include=[np.number]).columns
    if len(numerical_cols) > 0:
        st.markdown('<h4>🔢 Numerical Variables</h4>', unsafe_allow_html=True)
        
        # Descriptive statistics
        stats_df = data[numerical_cols].describe()
        st.dataframe(stats_df, use_container_width=True)
        
        # Correlation matrix
        if len(numerical_cols) > 1:
            st.markdown('<h4>🔗 Correlation Matrix</h4>', unsafe_allow_html=True)
            corr_matrix = data[numerical_cols].corr()
            
            # Create heatmap
            fig = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="RdBu",
                title="Correlation Matrix of Numerical Variables"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Categorical variables summary
    categorical_cols = data.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        st.markdown('<h4>📝 Categorical Variables</h4>', unsafe_allow_html=True)
        
        for col in categorical_cols:
            with st.expander(f"📊 {col} Analysis"):
                value_counts = data[col].value_counts()
                
                # Bar chart
                fig = px.bar(
                    x=value_counts.index,
                    y=value_counts.values,
                    title=f"{col} Distribution",
                    labels={'x': col, 'y': 'Count'}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Summary table
                summary_df = pd.DataFrame({
                    'Category': value_counts.index,
                    'Count': value_counts.values,
                    'Percentage': (value_counts.values / len(data) * 100).round(2)
                })
                st.dataframe(summary_df, use_container_width=True)

def display_visualizations(results):
    """Display the generated visualizations."""
    st.markdown('<h3>📈 Data Visualizations</h3>', unsafe_allow_html=True)
    
    if 'plots' in results and results['plots']:
        st.markdown(f"<div class='info-box'>Generated {len(results['plots'])} professional visualizations</div>", unsafe_allow_html=True)
        
        # Display each plot
        for i, plot_path in enumerate(results['plots']):
            if os.path.exists(plot_path):
                plot_name = os.path.basename(plot_path).replace('.png', '').replace('_', ' ').title()
                
                with st.expander(f"📊 {plot_name}"):
                    st.image(plot_path, use_column_width=True)
                    
                    # Download button for the plot
                    with open(plot_path, "rb") as file:
                        btn = st.download_button(
                            label=f"📥 Download {plot_name}",
                            data=file.read(),
                            file_name=os.path.basename(plot_path),
                            mime="image/png"
                        )
    else:
        st.warning("⚠️ No visualizations generated. Please check the analysis process.")

def display_key_insights(data, results):
    """Display key insights from the analysis."""
    st.markdown('<h3>🔍 Key Insights</h3>', unsafe_allow_html=True)
    
    # Basic insights
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<h4>📊 Dataset Overview</h4>', unsafe_allow_html=True)
        st.markdown(f"""
        - **Total Records**: {data.shape[0]:,}
        - **Variables**: {data.shape[1]}
        - **Data Types**: {len(data.select_dtypes(include=[np.number]).columns)} numerical, {len(data.select_dtypes(include=['object']).columns)} categorical
        """)
    
    with col2:
        st.markdown('<h4>❓ Data Quality</h4>', unsafe_allow_html=True)
        missing_data = data.isnull().sum()
        total_missing = missing_data.sum()
        
        if total_missing > 0:
            st.markdown(f"""
            - **Missing Values**: {total_missing:,}
            - **Variables with Missing Data**: {missing_data[missing_data > 0].count()}
            - **Data Completeness**: {((len(data) * len(data.columns) - total_missing) / (len(data) * len(data.columns)) * 100):.1f}%
            """)
        else:
            st.success("✅ Perfect data quality - no missing values!")
    
    # Numerical insights
    numerical_cols = data.select_dtypes(include=[np.number]).columns
    if len(numerical_cols) > 0:
        st.markdown('<h4>🔢 Numerical Variables Insights</h4>', unsafe_allow_html=True)
        
        for col in numerical_cols:
            with st.expander(f"📊 {col} Analysis"):
                col_data = data[col].dropna()
                
                # Distribution plot
                fig = px.histogram(
                    x=col_data,
                    title=f"{col} Distribution",
                    nbins=30,
                    marginal="box"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Statistics
                stats = {
                    'Mean': col_data.mean(),
                    'Median': col_data.median(),
                    'Std Dev': col_data.std(),
                    'Min': col_data.min(),
                    'Max': col_data.max(),
                    'Q1': col_data.quantile(0.25),
                    'Q3': col_data.quantile(0.75)
                }
                
                # Display stats in columns
                cols = st.columns(4)
                for i, (stat_name, stat_value) in enumerate(stats.items()):
                    with cols[i % 4]:
                        st.metric(stat_name, f"{stat_value:.2f}")

def display_report_download(results):
    """Display report download section."""
    st.markdown('<h3>📝 EDA Report</h3>', unsafe_allow_html=True)
    
    if 'report_path' in results and results['report_path']:
        report_path = results['report_path']
        
        if os.path.exists(report_path):
            st.markdown('<div class="success-box">', unsafe_allow_html=True)
            st.markdown(f"""
            ✅ **Report Generated Successfully!**
            
            **File**: {os.path.basename(report_path)}  
            **Size**: {os.path.getsize(report_path) / 1024:.1f} KB  
            **Generated**: {datetime.fromtimestamp(os.path.getmtime(report_path)).strftime('%Y-%m-%d %H:%M:%S')}
            """)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Download button
            with open(report_path, "rb") as file:
                btn = st.download_button(
                    label="📥 Download EDA Report (Word Document)",
                    data=file.read(),
                    file_name=os.path.basename(report_path),
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )
            
            st.markdown("""
            ### 📋 Report Contents:
            - Executive Summary
            - Data Overview and Quality Assessment
            - Statistical Analysis
            - Visual Analysis
            - Key Insights and Patterns
            - Conclusions and Recommendations
            - Technical Appendix
            """)
        else:
            st.error("❌ Report file not found. Please check the analysis process.")
    else:
        st.warning("⚠️ No report generated. Please check the analysis process.")

def display_raw_data(data, results):
    """Display raw data and analysis details."""
    st.markdown('<h3>📋 Raw Data & Analysis Details</h3>', unsafe_allow_html=True)
    
    # Data preview
    st.markdown('<h4>📊 Data Preview</h4>', unsafe_allow_html=True)
    
    # Show data with pagination
    page_size = st.selectbox("Rows per page", [10, 25, 50, 100])
    total_pages = len(data) // page_size + (1 if len(data) % page_size > 0 else 0)
    
    if total_pages > 1:
        page = st.selectbox("Page", range(1, total_pages + 1))
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, len(data))
        
        st.dataframe(data.iloc[start_idx:end_idx], use_container_width=True)
        st.markdown(f"Showing rows {start_idx + 1}-{end_idx} of {len(data)}")
    else:
        st.dataframe(data, use_container_width=True)
    
    # Analysis details
    if 'analysis' in results:
        st.markdown('<h4>🔍 Analysis Details</h4>', unsafe_allow_html=True)
        
        with st.expander("View Full Analysis Results"):
            st.json(results['analysis'])

if __name__ == "__main__":
    main()
