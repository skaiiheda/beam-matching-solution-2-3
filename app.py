import os
import sys

import streamlit as st

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run dashboard
from pages.dashboard import main

if __name__ == "__main__":
    pages = [
        st.Page("pages/dashboard.py", title="Dashboard"),  # Ваша страница
    ]

    # position="hidden" убирает меню из сайдбара
    pg = st.navigation(pages, position="hidden")
    main()
