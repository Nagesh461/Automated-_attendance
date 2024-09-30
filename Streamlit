import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

st.set_page_config(layout="wide")

# Title
st.title('Automated Data Analysis')
st.text('Student attendance analysis')

# Function for EDA
def home(uploaded_file):
    if uploaded_file:
        st.header('Begin exploring the data using the menu on the left')
    else:
        st.header('To begin please upload a file')

def data_summary(df):
    st.header('Statistics of Dataframe')
    st.write(df.describe(include='all'))

def data_shape(df):
    st.header('Show Shape')
    st.write(df.shape)

def data_head(df):
    st.header('Show Head')
    st.write(df.head())

def data_tail(df):
    st.header('Show Tail')
    st.write(df.tail())

# Function for plotting histogram
def plot_histogram(df, column):
    plt.figure(figsize=(10, 5))
    plt.hist(df[column].dropna(), bins=30, edgecolor='k')
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    st.pyplot(plt)

# Function for plotting a bar chart based on student attendance
def plot_student_attendance(df, subject_column, student_column):
    plt.figure(figsize=(15, 6))  # Increase the figure size for horizontal scroll
    plt.bar(df[student_column], df[subject_column], color='skyblue')
    plt.title(f'Student Attendance in {subject_column}')
    plt.xlabel(student_column)
    plt.ylabel('Attendance')
    plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
    plt.tight_layout()
    
    # Set limits to allow for horizontal scrolling
    st.pyplot(plt)
    st.write(f"<style>div.stPlotlyChart {{overflow-x: auto;}}</style>", unsafe_allow_html=True)

# Function for plotting attendance of a specific student across different subjects
def plot_individual_student_attendance(df, student_name, subject_columns, student_column):
    student_data = df[df[student_column] == student_name][subject_columns]
    if not student_data.empty:
        student_data.T.plot(kind='line', marker='o', figsize=(10, 6), legend=False, color='skyblue')
        plt.title(f'Attendance for {student_name} Across Subjects')
        plt.xlabel('Subjects')
        plt.ylabel('Attendance')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(plt)
    else:
        st.error("No attendance data found for the selected student.")

# Function for plotting attendance as an area chart
def plot_student_attendance_area(df, subject_column, student_column):
    plt.figure(figsize=(15, 6))  # Increase the figure size for horizontal scroll
    plt.fill_between(df[student_column], df[subject_column], color='skyblue', alpha=0.5)
    plt.title(f'Student Attendance Area Chart for {subject_column}')
    plt.xlabel(student_column)
    plt.ylabel('Attendance')
    plt.xticks(rotation=90)
    plt.tight_layout()
    
    # Set limits to allow for horizontal scrolling
    st.pyplot(plt)
    st.write(f"<style>div.stPlotlyChart {{overflow-x: auto;}}</style>", unsafe_allow_html=True)

# Function for plotting a pie chart
def plot_pie_chart(df):
    if df.shape[0] >= 1:  # Check if there's at least one row
        pie_data = df.iloc[0].dropna()  # Get the first row and drop NaN values
        pie_data = pie_data[pie_data.apply(np.isreal)]  # Keep only numerical data

        plt.figure(figsize=(10, 6))
        plt.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=140)
        plt.title('Pie Chart of First Row Data')
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        st.pyplot(plt)
    else:
        st.warning("Not enough data to generate a pie chart. Please ensure there is at least one row of data.")

# Function to add a total row to the DataFrame
def add_total_row(df):
    df_sum = df.select_dtypes(include=[np.number]).fillna(0).sum()
    sum_df = pd.DataFrame(df_sum).T
    sum_df.index = ['Total']
    df_with_total = pd.concat([df, sum_df])
    return df_with_total

# Function to add a total column to the DataFrame
def add_total_column(df):
    df['Row_Total'] = df.select_dtypes(include=[np.number]).fillna(0).sum(axis=1)
    return df

# Function to calculate percentage based on the first row's values (keeping the first row as total)
def add_column_percentage_based_on_first_row(df):
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    first_row = df.iloc[0]  # Assuming first row contains the total values

    # Calculate percentage for each column based on the first row
    df_percentage = df[numerical_cols[:]].apply(lambda x: (x / first_row[numerical_cols[0:]]) * 100, axis=1)
    df_percentage = df_percentage.fillna(0).round(2)
    df_percentage.columns = [f"{col}_Col%" for col in df_percentage.columns]  # Update column names
    return df_percentage

# Function to generate a combined DataFrame with totals and percentages
def generate_combined_df(df):
    df_with_total = add_total_row(df)
    df_with_total_column = add_total_column(df_with_total)
    df_with_col_percentage = add_column_percentage_based_on_first_row(df_with_total_column)

    # Positioning percentage columns right next to the original columns
    combined_df = df_with_total_column.copy()
    for col in df_with_col_percentage.columns:
        combined_df.insert(combined_df.columns.get_loc(col.split('_Col%')[0]) + 1, col, df_with_col_percentage[col])
    
    return combined_df

# Sidebar
st.sidebar.title('Sidebar')

# File uploader with support for CSV, Excel, TXT files
upload_file = st.sidebar.file_uploader('Upload a file (CSV, Excel, TXT)', type=['csv', 'xlsx', 'xls', 'txt'])

# Reading the uploaded file
if upload_file is not None:
    file_type = upload_file.name.split('.')[-1]
    
    if file_type == 'csv':
        df = pd.read_csv(upload_file)
    elif file_type in ['xlsx', 'xls']:
        df = pd.read_excel(upload_file)
    elif file_type == 'txt':
        df = pd.read_csv(upload_file, delimiter='\t')

    # Dynamically use the first column for student names
    student_column = df.columns[0]  # Get the first column as student names

    # Dynamically identify subject columns (assuming they're numerical and not the student column)
    subject_columns = df.columns[1:][df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').notnull().all()].tolist()

    df_cleaned = df.copy()
    df_cleaned[subject_columns] = df_cleaned[subject_columns].apply(pd.to_numeric, errors='coerce')

    # Sidebar navigation
    st.sidebar.title('EDA Options')
    show_home = st.sidebar.checkbox('Home')
    show_summary = st.sidebar.checkbox('Data Summary')
    show_shape = st.sidebar.checkbox('Data Shape')
    show_head = st.sidebar.checkbox('Data Head')
    show_tail = st.sidebar.checkbox('Data Tail')

    # Display options based on checkboxes
    if show_home:
        home(upload_file)
    if show_summary:
        data_summary(df_cleaned)
    if show_shape:
        data_shape(df_cleaned)
    if show_head:
        data_head(df_cleaned)
    if show_tail:
        data_tail(df_cleaned)

    # Sidebar plotting options
    st.sidebar.title('Plot Options')
    plot_bar_chart_option = st.sidebar.checkbox('Plot Bar Chart')
    plot_individual_attendance_option = st.sidebar.checkbox('Plot Individual Student Attendance')
    plot_area_chart_option = st.sidebar.checkbox('Plot Area Chart')  # Checkbox for Area Chart
    plot_pie_chart_option = st.sidebar.checkbox('Plot Pie Chart')  # Checkbox for Pie Chart

    # Add a checkbox to show DataFrame with all totals and column percentages
    show_all_totals_and_percentages = st.sidebar.checkbox('Show All Totals and Percentages')

    # Plotting based on checkboxes
    numerical_cols = df_cleaned.select_dtypes(include=[np.number]).columns

    if show_all_totals_and_percentages:
        combined_df = generate_combined_df(df_cleaned)
        st.header('Combined DataFrame with Totals and Percentages')
        st.write(combined_df)

    plot_histogram_option = st.sidebar.checkbox('Plot Histogram')

    if plot_histogram_option and len(numerical_cols) > 0:
        selected_col = st.sidebar.selectbox('Select a column for histogram', numerical_cols)
        plot_histogram(df_cleaned, selected_col)

    if plot_bar_chart_option and subject_columns:
        selected_subject = st.sidebar.selectbox('Select a subject for attendance bar chart', subject_columns)
        plot_student_attendance(df_cleaned, selected_subject, student_column)

    if plot_individual_attendance_option:
        selected_student = st.sidebar.selectbox(f'Select a student ({student_column}) for attendance across subjects', df_cleaned[student_column].unique())
        plot_individual_student_attendance(df_cleaned, selected_student, subject_columns, student_column)

    if plot_area_chart_option and subject_columns:
        selected_subject_area = st.sidebar.selectbox('Select a subject for attendance area chart', subject_columns)
        plot_student_attendance_area(df_cleaned, selected_subject_area, student_column)

    if plot_pie_chart_option:
        plot_pie_chart(df_cleaned)
