import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

# Load the Excel data
try:
    df = pd.read_excel('./gymData.xlsx', sheet_name='Sheet1')
except FileNotFoundError:
    st.error("Excel file not found. Please check the file path.")
    st.stop()

# Clean the data - Rename columns if needed
df.columns = ['Index', 'Time', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
df.drop(columns=['Index'], inplace=True)

# Ensure that all columns for days are numeric
day_columns = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
df[day_columns] = df[day_columns].apply(pd.to_numeric, errors='coerce')


# Get crowd data for manually selected time
def get_manual_crowd(df, selected_time, selected_day):
    crowd_data = df[df['Time'] == selected_time]

    if not crowd_data.empty:
        return crowd_data[selected_day].values[0]  # Return the crowd data for the selected day
    else:
        return None

# Normalize the time column
def normalize_time_format(time_str):
    time_str = time_str.strip()  # Remove any leading or trailing whitespace
    if 'a.m.' in time_str:
        return time_str.replace('a.m.', 'AM')
    if 'p.m.' in time_str:
        return time_str.replace('p.m.', 'PM')
    return time_str

# Normalize the time column in the DataFrame
df['Time'] = df['Time'].apply(normalize_time_format)

# Remove any rows where 'Time' is not in the correct format
df = df[df['Time'].str.match(r'^\d{1,2}:\d{2} (AM|PM)$', na=False)]

# Function to parse time safely
def parse_time(time_str):
    try:
        return datetime.datetime.strptime(time_str, '%I:%M %p')
    except ValueError:
        return None  # Return None if parsing fails

# Get the earliest time from the DataFrame that is within operational hours
def get_earliest_time():
    # Define operational hours
    operational_times = {
        'Monday': ((6, 0), (22, 0)),   # 6 AM - 10 PM
        'Tuesday': ((6, 0), (22, 0)),  # 6 AM - 10 PM
        'Wednesday': ((6, 0), (22, 0)),# 6 AM - 10 PM
        'Thursday': ((6, 0), (22, 0)), # 6 AM - 10 PM
        'Friday': ((6, 0), (20, 0)),    # 6 AM - 8 PM
        'Saturday': ((9, 0), (20, 0)),  # 9 AM - 8 PM
        'Sunday': ((11, 0), (22, 0)),   # 11 AM - 10 PM
    }

    earliest_time = {}
    for day in operational_times.keys():
        open_hour, close_hour = operational_times[day]
        times_for_day = df['Time'].unique()
        
        filtered_times = []
        for time in times_for_day:
            parsed_time = parse_time(time)
            if parsed_time:
                if (open_hour[0] < parsed_time.hour < close_hour[0]) or \
                   (open_hour[0] == parsed_time.hour and open_hour[1] <= parsed_time.minute) or \
                   (close_hour[0] == parsed_time.hour and parsed_time.minute == 0):
                    filtered_times.append(time)

        if filtered_times:
            earliest_time[day] = min(filtered_times, key=lambda t: parse_time(t))

    return earliest_time

# Check if the gym is closed based on the day and time
def is_gym_closed(current_day, current_hour):
    current_time = parse_time(current_hour)

    # Define operational hours for each day
    operational_hours = {
        'Monday': ((6, 0), (22, 0)),   # 6 AM - 10 PM
        'Tuesday': ((6, 0), (22, 0)),  # 6 AM - 10 PM
        'Wednesday': ((6, 0), (22, 0)),# 6 AM - 10 PM
        'Thursday': ((6, 0), (22, 0)), # 6 AM - 10 PM
        'Friday': ((6, 0), (20, 0)),    # 6 AM - 8 PM
        'Saturday': ((9, 0), (20, 0)),  # 9 AM - 8 PM
        'Sunday': ((11, 0), (22, 0)),   # 11 AM - 10 PM
    }

    # Check if current time is outside the operational hours
    open_time, close_time = operational_hours[current_day]
    if (current_time is None or 
        (current_time.hour < open_time[0]) or 
        (current_time.hour > close_time[0]) or 
        (current_time.hour == close_time[0] and current_time.minute > 0)):
        return True
    return False

# Get the current crowd data
def get_current_crowd(df):
    current_time = datetime.datetime.now()
    current_day = current_time.strftime('%A')  # Get the current day (e.g., Monday, Tuesday)
    current_hour = current_time.strftime('%I:%M %p')  # Get the current time in AM/PM format (e.g., 02:00 PM)

    # Filter data for the current time
    crowd_data = df[df['Time'] == current_hour]

    if not crowd_data.empty:
        return crowd_data[current_day].values[0], current_day, current_hour  # Return crowd data, day, and time
    else:
        return None, current_day, current_hour

# Create a scatter plot for crowd visualization
def create_scatter_plot(crowd_count):
    fig, ax = plt.subplots(figsize=(11, 7))

    # Set background color to white
    ax.set_facecolor('white')

    # Generate random positions for the dots
    if crowd_count > 0:
        x_positions = np.random.uniform(0, 10, crowd_count)
        y_positions = np.random.uniform(0, 6, crowd_count)

        # Plot the dots
        ax.scatter(x_positions, y_positions, color='blue', s=100)

    plt.title("Current Gym Crowd Visualization")
    plt.xlim(0, 10)
    plt.ylim(0, 6)
    plt.axis("off")
    return fig

# Streamlit App Interface
st.title("Gym Crowd Visualizer")

# User input: Select mode (Automatic or Manual)
mode = st.selectbox("Select Mode:", ["Automatic", "Manual Time Selection"])

# Get the earliest gym time data within operational hours
earliest_times = get_earliest_time()

if mode == "Automatic":
    # Display current crowd based on server time
    st.header("Current Gym Crowd Levels")
    current_crowd, current_day, current_hour = get_current_crowd(df)

    # Check if the current time is outside operational hours
    if is_gym_closed(current_day, current_hour):
        st.write("The gym is closed! Here are the operating hours:")
        st.write("""
        **Operating Hours:**
        - **Monday - Thursday:** 6 AM - 10 PM
        - **Friday:** 6 AM - 8 PM
        - **Saturday:** 9 AM - 8 PM
        - **Sunday:** 11 AM - 10 PM
        """)
    else:
        if current_crowd is not None:
            # Check if the crowd data indicates closure
            if current_crowd == 'X':
                st.write("The gym is closed!")
            else:
                st.write(f"The current gym crowd is roughly {int(current_crowd)} people out of a maximum capacity of 60.")
                fig = create_scatter_plot(int(current_crowd))
                st.pyplot(fig)
        else:
            st.write("No data available for this time.")

elif mode == "Manual Time Selection":
    # User input: Select day and time
    selected_day = st.selectbox("Select Day:", day_columns)
    
    # Set the default selected time to the earliest time for the selected day
    default_time = earliest_times.get(selected_day, df['Time'].min())
    selected_time = st.selectbox("Select Time:", df['Time'].unique(), index=list(df['Time'].unique()).index(default_time))

    st.header("Gym Crowd Levels")

    # Get the manually selected crowd data
    manual_crowd = df.loc[df['Time'] == selected_time, selected_day].values

    if manual_crowd.size > 0:
        manual_crowd_value = manual_crowd[0]  # Get the first value
        # Check if the gym is closed
        if manual_crowd_value == 'X' or is_gym_closed(selected_day, selected_time):
            st.write("The gym is closed!")
        else:
            st.write(f"The gym crowd at {selected_time} on {selected_day} is {int(manual_crowd_value)} people out of a maximum capacity of 60.")
            fig = create_scatter_plot(int(manual_crowd_value))
            st.pyplot(fig)
    else:
        st.write("No crowd data available for the selected time.")


# Footer credit
st.markdown("Created by BUS Marketing Team W&L")
