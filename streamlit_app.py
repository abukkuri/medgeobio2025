import streamlit as st

st.set_page_config(page_title="Double Bind Evolution Models", layout="centered")

st.title("Double Bind Cancer Therapy Models")
st.write("""
Welcome!  
Use the sidebar to choose between:

1. **rCost Model** — carrying capacity fixed, resistance reduces growth rate.
2. **KCost Model** — growth rate fixed, resistance reduces carrying capacity.

These simulations show how resistance evolves under double bind therapy.
""")

st.info("Choose a model from the sidebar under **Pages**.")