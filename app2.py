'''
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import datetime

# Load the gym floor layout image
try:
    gym_layout = Image.open("./section_I.png")  # Update the path if needed
except FileNotFoundError:
    st.error("Gym layout image not found. Please check the file path.")
    st.stop()

# Load the Excel data
try:
    df = pd.read_excel('./BUS 211 Data Tanner Hurless 9.9.24.xlsx', sheet_name='Sheet1')
except FileNotFoundError:
    st.error("Excel file not found. Please check the file path.")
    st.stop()

# Clean the data - Rename columns if needed
df.columns = ['Index', 'Time', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
df.drop(columns=['Index'], inplace=True)

# Ensure that all columns for days are numeric
day_columns = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
df[day_columns] = df[day_columns].apply(pd.to_numeric, errors='coerce')

# Get the current day and time
def get_current_crowd(df):
    current_time = datetime.datetime.now()
    current_day = current_time.strftime('%A')  # Get the current day (e.g., Monday, Tuesday)
    current_hour = current_time.strftime('%I:%M %p')  # Get the current time in AM/PM format (e.g., 02:00 PM)

    # Filter data for the current time
    crowd_data = df[df['Time'] == current_hour]

    if not crowd_data.empty:
        return crowd_data[current_day].values[0]  # Return the crowd data for the current day
    else:
        return None

# Get crowd data for manually selected time
def get_manual_crowd(df, selected_time, selected_day):
    crowd_data = df[df['Time'] == selected_time]

    if not crowd_data.empty:
        return crowd_data[selected_day].values[0]  # Return the crowd data for the selected day
    else:
        return None

# Create gym layout with crowd visualization
def create_gym_visualization(crowd_count):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Display gym layout as background
    ax.imshow(gym_layout, extent=[0, 10, 0, 6])
    
    # Define section positions and total sections in the gym
    sections = {
        "Chest Press": (3, 4),
        "Dumbbells": (8, 5),
        "Leg Press": (7, 2),
        "Seated Row": (2, 3),
        "Quad Extension": (6, 4),
    }

    # Plot for each section based on crowd count
    for section, (x, y) in sections.items():
        section_crowd = min(crowd_count, 60)  # Ensure no section has more than max capacity
        for i in range(section_crowd):
            # Scatter dots for crowd representation
            ax.scatter(x + np.random.uniform(-0.1, 0.1), y + np.random.uniform(-0.1, 0.1), color='blue', s=50)

        # Color the rectangle based on crowd levels
        if section_crowd < 15:
            color = "green"
        elif 15 <= section_crowd < 30:
            color = "yellow"
        else:
            color = "red"

        ax.add_patch(plt.Rectangle((x-0.5, y-0.5), 1, 1, color=color, alpha=0.3))

    plt.title("Gym Layout with Crowd Visualization")
    plt.xlim(0, 10)
    plt.ylim(0, 6)
    plt.axis("off")
    return fig

# Streamlit App Interface
st.title("Gym Crowd Visualizer")

# User input: Select mode (Automatic or Manual)
mode = st.selectbox("Select Mode:", ["Automatic (Server Time)", "Manual Time Selection"])

if mode == "Automatic (Server Time)":
    # Display current crowd based on server time
    st.header("Current Gym Crowd Levels (Automatic)")
    current_crowd = get_current_crowd(df)

    if current_crowd is not None:
        st.write(f"The current gym crowd is {current_crowd} people out of a maximum capacity of 60.")
        fig = create_gym_visualization(int(current_crowd))
        st.pyplot(fig)
    else:
        st.write("No data available for this time.")

elif mode == "Manual Time Selection":
    # User input: Select day and time
    selected_day = st.selectbox("Select Day:", day_columns)
    selected_time = st.selectbox("Select Time:", df['Time'].unique())

    st.header("Gym Crowd Levels (Manual)")

    # Get the manually selected crowd data
    manual_crowd = get_manual_crowd(df, selected_time, selected_day)

    if manual_crowd is not None:
        st.write(f"The gym crowd at {selected_time} on {selected_day} is {manual_crowd} people out of a maximum capacity of 60.")
        fig = create_gym_visualization(int(manual_crowd))
        st.pyplot(fig)
    else:
        st.write("No data available for the selected time and day.")

# Footer credit
st.markdown("Created by BUS Marketing Team W&L")
'''
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

# Load the Excel data
try:
    df = pd.read_excel('./BUS 211 Data Tanner Hurless 9.9.24.xlsx', sheet_name='Sheet1')
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


# Function to normalize time formats
def normalize_time_format(time_str):
    time_str = time_str.strip()  # Remove any leading or trailing whitespace
    if 'a.m.' in time_str:
        return time_str.replace('a.m.', 'AM')
    if 'p.m.' in time_str:
        return time_str.replace('p.m.', 'PM')
    return time_str

# Normalize the time column in the DataFrame
df['Time'] = df['Time'].apply(normalize_time_format)

# Check if the gym is closed based on the day and time
def is_gym_closed(current_day, current_hour):
    current_time = datetime.datetime.strptime(current_hour, '%I:%M %p')

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
    if (current_time.hour < open_time[0]) or (current_time.hour > close_time[0]) or \
       (current_time.hour == close_time[0] and current_time.minute > 0):
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
# Define the area boundaries (adjust these values to fit your layout)
    x_min, x_max = 0, 10  # Horizontal boundaries
    y_min, y_max = 0, 6    # Vertical boundaries

    # Generate random positions for the dots, constrained within the defined boundaries
    if crowd_count > 0:
        x_positions = np.random.uniform(x_min, x_max, crowd_count)
        y_positions = np.random.uniform(y_min, y_max, crowd_count)

        # Plot the dots
        ax.scatter(x_positions, y_positions, color='blue', s=100)

    plt.title("Current Gym Crowd Visualization")
    
    # Set the limits of the plot
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    plt.axis("off")
    return fig

# Streamlit App Interface
st.title("Gym Crowd Visualizer")

# User input: Select mode (Automatic or Manual)
mode = st.selectbox("Select Mode:", ["Automatic Update", "Manual Time Selection"])

if mode == "Automatic Update":
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
                st.write(f"The current gym crowd is {current_crowd} people out of a maximum capacity of 60.")
                fig = create_scatter_plot(int(current_crowd))
                st.pyplot(fig)
        else:
            st.write("No data available for this time.")

elif mode == "Manual Time Selection":
    # User input: Select day and time
    selected_day = st.selectbox("Select Day:", day_columns)
    selected_time = st.selectbox("Select Time:", df['Time'].unique())

    st.header("Gym Crowd Levels")

    # Get the manually selected crowd data
    manual_crowd = df.loc[df['Time'] == selected_time, selected_day].values

    if manual_crowd.size > 0:
        manual_crowd_value = manual_crowd[0]  # Get the first value
        # Check if the gym is closed
        if manual_crowd_value == 'X' or is_gym_closed(selected_day, selected_time):
            st.write("The gym is closed!")
        else:
            st.write(f"The gym crowd at {selected_time} on {selected_day} is {manual_crowd_value} people out of a maximum capacity of 60.")
            fig = create_scatter_plot(int(manual_crowd_value))
            st.pyplot(fig)
    else:
        st.write("No data available for the selected time and day.")

# Footer credit
st.markdown("Created by BUS Marketing Team W&L")
