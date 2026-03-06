import streamlit as st
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run dashboard
from pages.dashboard import main

if __name__ == "__main__":
    main()
