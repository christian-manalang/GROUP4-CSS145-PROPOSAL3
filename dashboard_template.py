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

    # Your content for your DATASET page goes here

    st.markdown("""
    ### Japan Birth Statistics Dataset Overview

    **Link to dataset**: [Japan Birth Statistics on Kaggle](https://www.kaggle.com/datasets/webdevbadger/japan-birth-statistics)

    This dataset covers various statistics related to births in Japan, such as birth rate, gender ratio, population, death rate, parents' ages, and the number of children per family. The data spans from **1899 to 2022**, though some years are missing due to record losses during World War II, providing over **120 years** of historical perspective.

    **Content**  
    The dataset contains around **50 columns**; however, only a selected subset will be used for our analysis. The main focus will be on predicting Japan's birth rate trends over the next few years.

    ### Project Goals
    Our project aims to use this dataset to forecast the birth rate in Japan for the next five years. Although data collection stops at 2022, the extensive historical records provide a solid foundation for predicting trends. We will also explore correlations between birth rate and factors like parents' ages and external events, such as economic downturns, pandemics, and natural disasters, to understand how these factors may influence birth rates.

    ### Proposed Models
    Given the time-series nature of this prediction, we will employ time series models, particularly **ARIMA**. Additionally, regression models will be tested to determine which factors are most influential in affecting Japan's birth rate.

    ### Dataset Preview
    """)
                
    st.dataframe(dataset, use_container_width=True, hide_index=True)

    st.markdown("""
    ### Descriptive Statistics
    """)

    st.dataframe(dataset.describe(), use_container_width=True)
                
    st.markdown("""
    ### Summary Statistics
    Below is an overview of some key statistics related to the birth rate and influencing factors.

    - **Mean Birth Rate**: The dataset provides an average annual birth rate over the years, showing trends across decades.
    - **Parent's Average Age**: Insights into the age at which parents tend to have children, highlighting shifts in demographic behavior.
    - **Death Rate Influence**: The correlation between birth rate and mortality rates across different periods, particularly in times of historical crisis.

    [Insert summary statistics table here with columns like Mean Birth Rate, Mean Parents' Age, and Death Rate]

    ### Analysis Insights
    - **Fluctuations in Birth Rate**: Historical data reveals birth rate changes due to various external factors, such as economic and social conditions.
    - **Effect of Parents' Ages**: Trends in the age at which individuals start families can have significant implications on birth rates, especially as socio-economic factors evolve.
    - **Historical Events and Birth Rate**: The dataset allows us to observe how major events, such as natural disasters, economic downturns, and public health crises, impacted birth rates.

    With these insights and models, we aim to provide a predictive analysis of Japan's future birth rate, contributing to demographic research and planning.
    """)


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