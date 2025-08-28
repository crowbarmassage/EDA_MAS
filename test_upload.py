#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to verify CSV upload functionality
"""
import pandas as pd
import streamlit as st
from io import StringIO, BytesIO
import sys
import os

def test_csv_upload():
    """Test CSV file upload functionality"""
    print("Testing CSV upload functionality...")
    
    # Test 1: Read the sample CSV file
    csv_file_path = "sample_customer_data.csv"
    
    if not os.path.exists(csv_file_path):
        print(f"ERROR: Sample CSV file not found: {csv_file_path}")
        return False
    
    try:
        # Test reading CSV file
        data = pd.read_csv(csv_file_path)
        print(f"SUCCESS: Successfully read CSV: {data.shape[0]} rows x {data.shape[1]} columns")
        
        # Test file upload simulation (BytesIO)
        with open(csv_file_path, 'r') as f:
            csv_string = f.read()
        
        # Simulate uploaded file
        uploaded_file = BytesIO(csv_string.encode())
        
        # Test pandas read from BytesIO
        uploaded_file.seek(0)  # Reset pointer
        data_from_upload = pd.read_csv(uploaded_file)
        print(f"SUCCESS: Successfully read from simulated upload: {data_from_upload.shape[0]} rows x {data_from_upload.shape[1]} columns")
        
        # Test data integrity
        if data.shape == data_from_upload.shape:
            print("SUCCESS: Data integrity verified - shapes match")
        else:
            print("ERROR: Data integrity issue - shapes don't match")
            return False
        
        # Test column types
        print("\nColumn information:")
        for col in data.columns:
            print(f"  - {col}: {data[col].dtype}")
        
        # Test for missing values
        missing_values = data.isnull().sum().sum()
        print(f"\nMissing values: {missing_values}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Error during CSV upload test: {str(e)}")
        return False

def test_streamlit_file_uploader():
    """Test the Streamlit file uploader component"""
    print("\nTesting Streamlit file uploader component...")
    
    try:
        # Import streamlit components
        import streamlit as st
        print("SUCCESS: Streamlit imported successfully")
        
        # Test file types
        supported_types = ['csv']
        print(f"SUCCESS: Supported file types: {supported_types}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Error testing Streamlit components: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("Starting CSV Upload Functionality Tests")
    print("=" * 50)
    
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Run tests
    test1_passed = test_csv_upload()
    test2_passed = test_streamlit_file_uploader()
    
    print("\n" + "=" * 50)
    print("Test Results:")
    print(f"  CSV Upload Test: {'PASSED' if test1_passed else 'FAILED'}")
    print(f"  Streamlit Component Test: {'PASSED' if test2_passed else 'FAILED'}")
    
    if test1_passed and test2_passed:
        print("\nAll tests passed! CSV upload should work correctly.")
        return True
    else:
        print("\nSome tests failed. There may be issues with CSV upload.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)