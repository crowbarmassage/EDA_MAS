#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal Streamlit app to test CSV upload functionality
"""
import streamlit as st
import pandas as pd

# Page config
st.set_page_config(
    page_title="CSV Upload Test",
    page_icon="📊",
    layout="wide"
)

st.title("CSV Upload Test")
st.write("This is a minimal test to verify CSV upload functionality.")

# Test file uploader
uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=['csv'],
    help="Upload your CSV file for testing"
)

if uploaded_file is not None:
    try:
        # Display file info
        st.success(f"File uploaded successfully: {uploaded_file.name}")
        st.write(f"File size: {uploaded_file.size} bytes")
        st.write(f"File type: {uploaded_file.type}")
        
        # Read the CSV
        data = pd.read_csv(uploaded_file)
        
        # Display basic info
        st.write(f"Data shape: {data.shape[0]} rows x {data.shape[1]} columns")
        
        # Show data preview
        st.subheader("Data Preview")
        st.dataframe(data.head())
        
        # Show column info
        st.subheader("Column Information")
        for col in data.columns:
            st.write(f"- {col}: {data[col].dtype}")
        
        # Show basic stats
        st.subheader("Basic Statistics")
        st.write(data.describe())
        
    except Exception as e:
        st.error(f"Error reading CSV file: {str(e)}")
else:
    st.info("Please upload a CSV file to test the functionality.")
    
    # Show instructions
    st.markdown("""
    ### Instructions:
    1. Click the "Browse files" button above
    2. Select a CSV file from your computer
    3. The file should upload and display automatically
    
    ### Expected behavior:
    - File upload widget should be visible
    - Click should open file browser
    - CSV files should be selectable
    - File should upload and process immediately
    """)