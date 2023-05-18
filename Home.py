import streamlit as st

# Add a logo image at the top
logo_image = "logo.jpeg"
tambil_image = 'tambil.png'



st.set_page_config(
    page_title="AyuAIra",
    page_icon="👨‍⚕️",
    
)
st.write("# Welcome to AyuAIra! 👨‍⚕️")
st.markdown(
    """
    Your Intelligent Companion for Holistic Arthritis Management
   
"""
)
st.image(logo_image, width=300)



st.sidebar.success("Select Option.")

st.markdown(
    """
    The proposed system integrates advanced technology with traditional knowledge of Ayurvedic medicine to provide a comprehensive solution for the early detection and effective management of arthritis.
    ### App Features 👈
    
    - X-Ray Arthritis Detector
    - Arthritis Blood Report Analyzer
    - Arthritis Treatment Recommendation
    - Continuous Monitoring And Feedback
   
"""
)