import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


############# mini_eda.py #############

def read_users_csv(uploaded_file) -> pd.DataFrame:
    """
    CSV file uploaded by a user is returned as a Pandas DataFrame
    """
    df = pd.read_csv(uploaded_file)
    # remove the unnamed column
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    return df


def display_shape(df: pd.DataFrame) -> None:
    """
    Display the shape of the dataset
    """
    st.subheader("Shape of the Dataset")
    st.write("There are", df.shape[0], "rows and", df.shape[1], "columns in the dataset.")


def display_random_rows(df: pd.DataFrame) -> None:
    """
    Display 5 random rows from the dataset
    """
    st.subheader("Sample of 5 Random Rows from the Dataset")
    st.dataframe(df.sample(5), height=240, width=800)


def display_dtypes_and_na(df: pd.DataFrame) -> None:
    """
    create a new dataframe with column names, data types, and percentage of missing values
    """
    # create a list of column names
    column_names = list(df.columns)
    # create a list of data types of each column
    data_types = list(df.dtypes)
    # create a list of percentage of missing values in each column
    missing_values = list(df.isnull().sum() / df.shape[0] * 100)

    # create a new dataframe
    df_df = pd.DataFrame({"Column Name": column_names, "Data Type": data_types, "Missing Values (%)": missing_values})
    # display the new dataframe
    st.subheader("Data Types and Missing Values by Column")
    st.dataframe(df_df, height=240, width=800)


def heatmap_missing_values(df: pd.DataFrame) -> None:
    """
    Create a heatmap of missing values by column
    """
    st.subheader("Heatmap of Missing Values by Column")
    fig, ax = plt.subplots(figsize=(8, 6))
    # set the background color of the plot to a light yellow
    ax.set_facecolor("#f7e8a1")
    # set the background color of the figure to a light orange
    fig.set_facecolor("#eb804c")
    sns.heatmap(df.isnull(), cmap=["#f7e8a1", "#7a88cc"], cbar=False, ax=ax)
    plt.xticks(rotation=75)
    ax.set_title("Missing Values by Column")
    ax.set_xlabel("Features")
    ax.set_ylabel("Rows")
    st.pyplot(fig)


def barplot_missing_values(df: pd.DataFrame) -> None:
    """
    bar plot of missing values by column
    """
    st.subheader("Bar Plot of Missing Values by Column")
    fig, ax = plt.subplots(figsize=(8, 6))
    # set the background color of the plot to a light yellow
    ax.set_facecolor("#f7e8a1")
    # set the background color of the figure to a light orange
    fig.set_facecolor("#eb804c")
    sns.barplot(x=df.columns, y=df.isnull().sum(), ax=ax)
    # tilt the x-axis labels
    plt.xticks(rotation=75)
    plt.title("Missing Values by Column")
    plt.xlabel("Features")
    plt.ylabel("Sum of Missing Values")
    st.pyplot(fig)


def emptiest_rows(df: pd.DataFrame) -> None:
    """
    display the top 5 rows with the most missing values
    """
    st.subheader("Rows with the Most Missing Values")
    missing_values = df.isna().sum(axis=1)
    # Sort the dataframe by the number of missing values in each row
    df_sorted = df.loc[missing_values.sort_values(ascending=False).index]
    # Select the top 5 rows with the most missing values
    df_top_5 = df_sorted.head(5)
    # Display the top 5 rows
    st.dataframe(df_top_5, height=240, width=800)


def display_df_describe(df: pd.DataFrame) -> None:
    """
    Display the summary statistics of the dataset
    """
    st.subheader("df.describe(); Summary Statistics of the Dataset")
    st.dataframe(df.describe(), height=240, width=800)


def correlation_heatmap(df: pd.DataFrame) -> None:
    """
    Display the correlation heatmap
    """
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(8, 6))
    # set the background color of the plot to a light yellow
    ax.set_facecolor("#f7e8a1")
    # set the background color of the figure to a light orange
    fig.set_facecolor("#eb804c")

    # Get the numerical columns
    num_cols = df.select_dtypes(include=["int", "float"]).columns.tolist()
    # Calculate the correlations
    corr = df[num_cols].corr()
    # Display the heatmap
    sns.heatmap(corr, annot=True, cmap="coolwarm")

    plt.xticks(rotation=75)
    ax.set_title("Correlation Heatmap")
    ax.set_xlabel("Features")
    ax.set_ylabel("Rows")
    st.pyplot(fig)


def check_if_correlations(df: pd.DataFrame) -> None:
    """
    Check if there are any correlations above 0.7 or below -0.7 in the input DataFrame.
    If there are, display the highest correlation and a prediction message for the two columns with the highest correlation.
    If there are more than one correlation above 0.7, also display the second highest correlation and prediction message.
    If there are no correlations above 0.7, find correlations between 0.51 and 0.7 in absolute value and display them with a message.
    If there are no correlations above 0.51, display a message saying no correlations were found.
    """
    # Get the numerical columns
    num_cols = df.select_dtypes(include=["int", "float"]).columns.tolist()

    # Check if there are at least two numerical columns
    if len(num_cols) < 2:
        st.warning("At least two numerical columns are required.")
        return

    # Calculate the correlations
    corr = df[num_cols].corr()

    # Find the highest absolute correlation
    max_corr = corr.abs().unstack().sort_values(ascending=False).drop_duplicates()
    max_corr = max_corr[max_corr != 1]
    max_corr = max_corr[max_corr.abs() >= 0.7]

    # Display the highest correlation and prediction message
    if not max_corr.empty:
        col1, col2 = max_corr.index[0]
        if corr.loc[col1, col2] > 0:
            prediction = f"I might be able to predict future {col1} using {col2}"
        else:
            prediction = f"I might be able to predict future {col2} using {col1}"
        st.write(f"Highest correlation: {corr.loc[col1, col2]:.2f}")
        st.write(prediction)

        # Find the second highest absolute correlation
        if len(max_corr) > 1:
            col3, col4 = max_corr.index[1]
            if corr.loc[col3, col4] > 0:
                prediction = f"I might be able to predict future {col3} using {col4}"
            else:
                prediction = f"I might be able to predict future {col4} using {col3}"
            st.write(f"Second highest correlation: {corr.loc[col3, col4]:.2f}")
            st.write(prediction)

    else:
        st.write("No correlations found.")
        # Find correlations between 0.51 and 0.7 in absolute value
        mid_corr = corr.abs().unstack().sort_values(ascending=False).drop_duplicates()
        mid_corr = mid_corr[mid_corr != 1]
        mid_corr = mid_corr[(mid_corr.abs() >= 0.51) & (mid_corr.abs() < 0.7)]

        # Display the mid correlations and message
        if not mid_corr.empty:
            for col5, col6 in mid_corr.index:
                st.write(f"Correlation between {col5} and {col6}: {corr.loc[col5, col6]:.2f}")
                st.write(
                    f"There is a correlation between {col5} and {col6}, but it's not strong enough to be sure if I can use one to predict the other.")

############# ask_csv #############