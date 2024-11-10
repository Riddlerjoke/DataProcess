from data.fonctions import *

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Title of the web app
st.title("CSV File Uploader")

# Upload CSV file
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# If a file is uploaded
if uploaded_file is not None:

    # Read the file as a Pandas DataFrame
    df = read_users_csv(uploaded_file)

    # Display the shape of the dataset
    display_shape(df)

    # Display 5 random rows from the dataset
    display_random_rows(df)

    # Display the columns names, their data types and percentage of missing values
    display_dtypes_and_na(df)

    # bar plot of missing values by column
    barplot_missing_values(df)

    # Create a heatmap of missing values by column
    heatmap_missing_values(df)

    # Display the 5 rows with the most missing values
    emptiest_rows(df)

    # Display the descriptive statistics of the dataset
    display_df_describe(df)

    # Display the correlation heatmap
    correlation_heatmap(df)

    # Display the strongest correlations if any with "call to action"
    # too many possible errors, so we use a try/except block
    try:
        check_if_correlations(df)
    except:
        st.write("An error occurred while checking for correlations.")





















