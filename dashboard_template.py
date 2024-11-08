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

# List of columns to keep
columns_to_keep = [
    'year', 'birth_total', 'birth_male', 'birth_female', 'birth_rate',
    'birth_gender_ratio', 'population_total',
    'population_male', 'population_female'
]
# Drop irrelevant columns
dfnew = dataset[columns_to_keep]

# Forward-fill missing values directly with .ffill()
dfnew.loc[:, ['birth_total', 'birth_male', 'birth_female', 'birth_rate', 'birth_gender_ratio']] = dfnew[['birth_total', 'birth_male', 'birth_female', 'birth_rate', 'birth_gender_ratio']].ffill()



#######################

# Pages

# About Page
if st.session_state.page_selection == "about":
    st.header("ℹ️ About")

    st.markdown(""" 

    A data-driven web application to analyze and predict Japan's birth rate trends using historical data from 1899 to 2022. This project applies Exploratory Data Analysis (EDA), Time Series Forecasting, and Regression Modeling to identify key factors influencing birth rates and project future rates over the next five years.

    #### Pages

    1.'Dataset' - Overview of Japan's birth statistics, including birth rates, parent ages, gender ratios, and factors like economic and calamity impacts on birth trends.
    2.'EDA' - Exploratory analysis on birth rate fluctuations and demographic patterns over the years, with visualizations highlighting correlations between key factors.
    3.'Data' Preprocessing - Cleaning and transforming historical data for accurate modeling, handling missing values, and selecting relevant columns.
    4.'Modeling' - Using ARIMA for time-series forecasting and Regression Analysis to assess the impact of factors like parent ages and economic changes on birth rates.
    5.'Prediction' - Interactive prediction feature to estimate Japan's future birth rate, providing insights based on historical patterns and influential factors.
    6.'Conclusion' - Summarized insights on Japan's birth trends, key factors, and model performance in predicting future rates.
                """)



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
    Given the time-series nature of this prediction, we will employ a linear regression model which would be used to test which features would impact birth rate the most in our machine learning predictions.

    ### Dataset Preview
    """)
                
    st.dataframe(dfnew, use_container_width=True, hide_index=True)

    st.markdown("""
    ### Descriptive Statistics
    """)

    st.dataframe(dfnew.describe(), use_container_width=True)
                
    st.markdown("""
    The results from `dfnew.describe` show to us the different statistics that can be found within this new dataset that we have cleaned and filled with appropriate data that is missing. Here specifically we see how in each year there is an average of 1,641,856 births every year with a standard deviation of around 439,835 per year which is quite significant but understandable. It is also seen that ever since 1899, the lowest birth total recorded for the dataset was 770,759.
    Moving on, we can see how when it comes to birth rate, which is the main point of this project. The minimum value found was 6.3 while 25% was 9.97, 50% at 18.70, 75% at 32.32 and finally the maximum birth rate recorded is 36.20 which is also the latest data from 2022.
    We can see initially that the birth rate of Japan from the 25th, 50th, and 75th percentiles were gradually increasing, but now we hear from the news how their birth rate is sharply decreasing as each year passes by, based on this we can probably conclude that the minimum value is actually Japan's current birth rate as based on graphs later, we would see how things took a turn for the worse it Japan's birth rate.
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