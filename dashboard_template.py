#######################
# Import libraries
import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA

#######################
# Page configuration
st.set_page_config(
    page_title="Japan Birth Rate", # Replace this with your Project's Title
    page_icon="assets/JapanIcon.png", # You may replace this with a custom icon or emoji related to your project
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

#######################

# Initialize page_selection in session state if not already set
if 'page_selection' not in st.session_state:
    st.session_state.page_selection = 'about'  # Default page

# Function to update page_selection
def set_page_selection(page):
    st.session_state.page_selection = page

# Sidebar
with st.sidebar:

    # Sidebar Title (Change this with your project's title)
    st.title('Japan Birth Rate')

    # Page Button Navigation
    st.subheader("Pages")

    if st.button("About", use_container_width=True, on_click=set_page_selection, args=('about',)):
        st.session_state.page_selection = 'about'
    
    if st.button("Dataset", use_container_width=True, on_click=set_page_selection, args=('dataset',)):
        st.session_state.page_selection = 'dataset'

    if st.button("EDA", use_container_width=True, on_click=set_page_selection, args=('eda',)):
        st.session_state.page_selection = "eda"

    if st.button("Data Cleaning / Pre-processing", use_container_width=True, on_click=set_page_selection, args=('data_cleaning',)):
        st.session_state.page_selection = "data_cleaning"

    if st.button("Machine Learning", use_container_width=True, on_click=set_page_selection, args=('machine_learning',)): 
        st.session_state.page_selection = "machine_learning"

    if st.button("Prediction", use_container_width=True, on_click=set_page_selection, args=('prediction',)): 
        st.session_state.page_selection = "prediction"

    if st.button("Conclusion", use_container_width=True, on_click=set_page_selection, args=('conclusion',)):
        st.session_state.page_selection = "conclusion"

    # Project Members
    st.subheader("Members")
    st.markdown("1. Dominic Ryan C. Gonzales\n2. Jorge Christian B. Manalang\n3. Kirk Adrian E. Sunico\n4. Charles Matthew L. Dela Cruz\n5. Liandro E. Refulle")

#######################
# Data

# Load data
dataset = pd.read_csv("data/japan_birth.csv")

#######################

# Pages

# About Page
if st.session_state.page_selection == "about":
    st.header("ℹ️ About")

    # Your content for the ABOUT page goes here

# Dataset Page
elif st.session_state.page_selection == "dataset":
    st.header("📊 Dataset")

    # st.write("Japan Birth Rate Dataset")
    # st.write("")

    st.markdown("""
    ## 📊 Japan Birth Statistics Dataset Overview

    - **Link to dataset**: [Japan Birth Statistics on Kaggle](https://www.kaggle.com/datasets/webdevbadger/japan-birth-statistics)

    This dataset covers various statistics related to births in Japan, such as birth rate, gender ratio, population, death rate, parents' ages, and the number of children per family. The data spans from **1899 to 2022**, although some years are missing due to record losses during World War II. This extensive dataset offers a historical perspective of more than **120 years**.

    - **Content**  
      The dataset contains approximately **50 columns**, though we will focus on a selected subset to facilitate our project goals of analyzing and predicting Japan's birth rate trends.

    ### Project Goals
    Our project aims to predict Japan’s birth rate for the next five years using historical data from this dataset. Despite data collection stopping in 2022, the historical coverage provides a solid foundation for forecasting trends. We’ll explore correlations between birth rate and factors like parents' ages and external events, such as economic shifts, pandemics, and natural disasters, to see how these influences might affect birth rates.

    ### Proposed Models
    Given the time-based nature of our predictions, we will utilize time series models, particularly the **ARIMA model**, to forecast future trends. Additionally, we’ll experiment with regression models to identify factors that significantly impact the birth rate.

""")


    # Your content for your DATASET page goes here

# EDA Page
elif st.session_state.page_selection == "eda":
    st.header("📈 Exploratory Data Analysis (EDA)")


    col = st.columns((1.5, 4.5, 2), gap='medium')

    # Your content for the EDA page goes here

    with col[0]:
        st.markdown('#### Graphs Column 1')


    with col[1]:
        st.markdown('#### Graphs Column 2')
        
    with col[2]:
        st.markdown('#### Graphs Column 3')

# Data Cleaning Page
elif st.session_state.page_selection == "data_cleaning":
    st.header("🧼 Data Cleaning and Data Pre-processing")

    # Your content for the DATA CLEANING / PREPROCESSING page goes here

# Machine Learning Page
elif st.session_state.page_selection == "machine_learning":
    st.header("🤖 Machine Learning")

    # Your content for the MACHINE LEARNING page goes here

# Prediction Page
elif st.session_state.page_selection == "prediction":
    st.header("👀 Prediction")

    # Your content for the PREDICTION page goes here

# Conclusions Page
elif st.session_state.page_selection == "conclusion":
    st.header("📝 Conclusion")

    # Your content for the CONCLUSION page goes here