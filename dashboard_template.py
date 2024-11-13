#######################
# Import libraries
import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import io

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

    1. `Dataset` - Overview of Japan's birth statistics, including birth rates, parent ages, gender ratios, and factors like economic and calamity impacts on birth trends.
    2. `EDA` - Exploratory analysis on birth rate fluctuations and demographic patterns over the years, with visualizations highlighting correlations between key factors.
    3. `Data Cleaning / Pre-processing` - Cleaning and transforming historical data for accurate modeling, handling missing values, and selecting relevant columns.
    4. `Machine Learning` - Using ARIMA for time-series forecasting and Regression Analysis to assess the impact of factors like parent ages and economic changes on birth rates.
    5. `Prediction` - Interactive prediction feature to estimate Japan's future birth rate, providing insights based on historical patterns and influential factors.
    6. `Conclusion` - Summarized insights on Japan's birth trends, key factors, and model performance in predicting future rates.
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

    buffer = io.StringIO()
    dfnew.info(buf=buffer)
    s = buffer.getvalue()

    st.code(s, language="python")

    st.markdown("""
    The `dfnew.info()` shows to us the several different columns that will be mainly used for this project. 
    The float columns suggest that it contains decimals while the integer columns are mainly whole numbers.
    """)

    st.write(dfnew.describe())

    st.markdown("""
    The results from `dfnew.describe` show to us the different statistics that can be found within this new dataset that we have cleaned and filled with appropriate data that is missing. Here specifically we see how in each year there is an average of 1,641,856 births every year with a standard deviation of around 439,835 per year which is quite significant but understandable. It is also seen that ever since 1899, the lowest birth total recorded for the dataset was 770,759.

    Moving on, we can see how when it comes to birth rate, which is the main point of this project. The minimum value found was 6.3 while 25% was 9.97, 50% at 18.70, 75% at 32.32 and finally the maximum birth rate recorded is 36.20 which is also the latest data from 2022.

    We can see initially that the birth rate of Japan from the 25th, 50th, and 75th percentiles were gradually increasing, but now we hear from the news how their birth rate is sharply decreasing as each year passes by, based on this we can probably conclude that the minimum value is actually Japan's current birth rate as based on graphs later, we would see how things took a turn for the worse it Japan's birth rate.
    """)

    st.write("Unique Birth Rates:", dfnew['birth_rate'].unique())
    
    st.markdown("""
    As we have said from earlier, the birth rate of Japan is actually declining from the years. With that said we can 
    safely assume that once again it would continue towards a downwards trend unless an external factor is brought 
    upon their country which would help their birth rate immensely.

    It would also be appropriate to mention now how there are large gaps of differences in certain places of birth rate values, 
    this is because in Japan's history there has been many occurances which made their birth rate lower.

    To mention a few, obviously the World Wars played a part in it, lowering overall birth rate, however, it is also 
    essential to take note that in the 1960s Japan birth rate plumetted all because of their superstition that females 
    born during 1966 would have a bad personality and often kill their husbands. This is called the Hinoe-Uma (Fire horse) 
    if you want to search about it more.
    """)
    
    st.header("💡 Insights")
    st.markdown('#### Birth Rate in Japan (1899 - 2022)')

    plt.figure(figsize=(12, 6))
    plt.plot(dfnew['year'], dfnew['birth_rate'], marker='o', linestyle='-', color='b')
    plt.xlabel('Year')
    plt.ylabel('Birth Rate')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)
    
    st.markdown("""
    In this line graph, we can visually see how the birth rate of Japan at one point was still high, consistently 
    dipping and rising as wars, illnesses, and other external factor affect the country. However at a certain point, 
    the birth rate suddenly dipped once and there it continously dropped lower and lower to what it is now in present time.
    """)
    
    st.markdown('#### Total Births in Japan (1899 - 2022)')

    plt.figure(figsize=(12, 6))
    plt.plot(dfnew['year'], dfnew['birth_total'], marker='o', linestyle='-', color='r')
    plt.xlabel('Year')
    plt.ylabel('Total Births')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)
    
    st.markdown("""
    In this line graph, we can visually see the number of total births in Japan accounting for both male and female. 
    As we can see it has been declining fast from 1980 and onwards, this is probably due to the ever changing culture 
    of Japan and how they treat relationships and having children in general.
    """)
    
    st.markdown('#### Total Male Births in Japan (1899 - 2022)')

    plt.figure(figsize=(12, 6))
    plt.plot(dfnew['year'], dfnew['birth_male'], color='blue', marker='o', linestyle='-')
    plt.xlabel('Year')
    plt.ylabel('Total Male Births')
    plt.grid(True)
    st.pyplot(plt)

    st.markdown("""
    This line graph shows the total birth of males separated from the total number of births.
    """)
    
    st.markdown('#### Total Female Births in Japan (1899 - 2022)')

    plt.figure(figsize=(12, 6))
    plt.plot(dfnew['year'], dfnew['birth_female'], color='red', marker='o', linestyle='-')
    plt.xlabel('Year')
    plt.ylabel('Total Female Births')
    plt.grid(True)
    st.pyplot(plt)
    
    st.markdown("""
    Likewise, this line graph then shows the total number of births of females separated from the total number of births.
    """)
    
    st.markdown('#### Total Births of Males and Females in Japan (1899 - 2022)')

    plt.figure(figsize=(12, 6))
    plt.plot(dfnew['year'], dfnew['birth_male'], label='Male Births', color='blue', marker='o', linestyle='-')
    plt.plot(dfnew['year'], dfnew['birth_female'], label='Female Births', color='red', marker='o', linestyle='-')
    plt.xlabel('Year')
    plt.ylabel('Total Births')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)
    
    st.markdown("""
    In this line graph we combined both the births of females and males to see how much of a difference they have when 
    it comes to the deviation between the two genders. This can also be a way to see how accurate the males per females 
    born is. Generally we can see how males always dominated over the females born which is quite normal in many other countries.
    """)

# Data Cleaning Page
elif st.session_state.page_selection == "data_cleaning":
    st.header("🧼 Data Cleaning and Data Pre-processing")

    st.dataframe(dataset.head(), use_container_width=True, hide_index=True)

    st.markdown("As of now this is our current dataset, it is full of unecessary columns that we won't be using for this project. Currently we plan to deal with this by separating this dataset and choosing the columns we need then inserting that into a separate dataset so that we only show the data and columns that we need.")

    st.dataframe(dfnew.head(), use_container_width=True, hide_index=True)

    st.markdown("Because of this, we chose to specifically make use of certain columns by separating them from the actual dataset, and as mentioned earlier, we placed it into a separate one, this is so that we can be more organized and have a clearer vision of what columns and data we will use")

    st.code("dfnew.loc[:, ['birth_total', 'birth_male', 'birth_female', 'birth_rate', 'birth_gender_ratio']] = dfnew[['birth_total', 'birth_male', 'birth_female', 'birth_rate', 'birth_gender_ratio']].ffill()")
    
    st.markdown("For this new dataset, it still contained missing data due to the World War, a lot of data actually. So we decided to make use of forward fill to make up for this gap in the dataset, just so that we can have a better source of data than having none at all.")

    st.code("features = dfnew[['year', 'population_total', 'birth_total']] target = dfnew['birth_rate']")

    st.markdown("These are the specific features that we would be using for our machine learning later, here the data is being prepared and setup for machine learning use later on")

    st.code("X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)")

    st.markdown("As you can see these is where we have split our data for training and testing. You can see that for our features we have set year, population total, and birth total. While for our target it would be birth rate since this is what we want to predict using our model.")

    st.markdown("After this we would be then be able to proceed with using the data for machine learning purposes in the project.")


# Machine Learning Page
elif st.session_state.page_selection == "machine_learning":
    st.header("🤖 Machine Learning")
    
    st.subheader("Linear Regression")
    st.code("""

   features = ['year', 'birth_total', 'birth_male', 'birth_female', 'birth_rate','birth_gender_ratio', 'population_total','population_male', 'population_female']
dfnew['birth_rate']
     
    """)
    st.markdown("""
    Linear regression is a type of model which makes use of different variables to predict a specific variable. This variable that we choose to predict must be dependent on these independent variables in order to see how these independent variables ultimately impact the dependent variable.

In our case, we chose year, population_total, and birth_total as our independent variables, while birth_rate is our dependent variable, being the variable that we want to predict for this project.

By using linear regression model, we want to see how these three variables play a part in affecting a country's birth rate, in our case it is Japan.")
    """)
    

# Prediction Page
elif st.session_state.page_selection == "prediction":
    st.header("👀 Prediction")

    st.markdown('#### Actual vs Predicted Birth Rate')

    features = dfnew[['year', 'population_total', 'birth_total']]  # Add or modify as needed
    target = dfnew['birth_rate']  # Target variable for prediction

# Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Initialize the model
    model = LinearRegression()

# Training the model
    model.fit(X_train, y_train)

# Make predictions
    y_pred = model.predict(X_test)

# Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

# Visualizing the results with a line graph
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.reset_index(drop=True), label='Actual Birth Rate', color='blue', marker='o')
    plt.plot(y_pred, label='Predicted Birth Rate', color='orange', marker='x')
    plt.xlabel('Sample Index')
    plt.ylabel('Birth Rate')
    plt.legend()
    plt.grid()
    st.pyplot(plt)

    st.markdown("""
    So this line graph is the trained model wherein as we can see here it is quite accurate in the sense that it follows through with both the actual birth rate and the model's prediction.

    With that in mind it seems that the model itself works fine and it is the right model to use for this kind of dataset. Though we might still try to use ARIMA but it still remains confusing for most us to use.      
    """)

    st.markdown('#### Predicted Birth Rate on Unseen Data')

    # Example unseen data, made randomly in decrements of 10000-50000 starting from year 2022 in the japan_birth.csv
    df_unseen = pd.DataFrame({
        'year': [2023, 2024, 2025, 2026, 2027],
        'birth_total': [762011, 751055, 726175, 699909, 678123],
        'population_total': [122003123, 121981454, 121956174, 121929406, 121782339 ]
    })

    #assign the unseen_features to the df_unseen to use for later.
    unseen_features = df_unseen[['year', 'population_total', 'birth_total']]

    # Make predictions on the unseen data.
    unseen_predictions = model.predict(unseen_features)

    # Display predictions.
    print("Predicted Birth Rates for Unseen Data:")
    print(unseen_predictions)

    #visualization through line graph.
    plt.figure(figsize=(10, 6))
    plt.plot(unseen_predictions, label='Predicted Birth Rate', color='green', marker='x')
    plt.xlabel('Sample Index')
    plt.ylabel('Predicted Birth Rate')
    plt.legend()
    plt.grid()
    st.pyplot(plt) 

    st.markdown("""
   So here is a line graph depicting the model's performance when it comes to unseen data, in this case we gathered these different variables from random decrements from the latest data in the dataset which is from 2022, so from there we randomly decremented it each year by around 10000-30000.

    Based on our opinions regarding this model is that it is quite accurate, as you can see it is clearly depicting Japan's birth rate, however it is shocking that the predicted birth rate is higher than the actual dataset values.

    We think that this is quite probably because of the fact that there are many other factors to birth rate and that we probably need to add more features but we can still also experiment and see what features may impact the birth rate other than the ones that were mentioned in the main dataset which were birth total and population total. 
        """)

# Conclusions Page
elif st.session_state.page_selection == "conclusion":
    st.header("📝 Conclusion")

    # Your content for the CONCLUSION page goes here