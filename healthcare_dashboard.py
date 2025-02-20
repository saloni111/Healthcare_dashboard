# import streamlit as st
# import pandas as pd
# import numpy as np
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.schema import HumanMessage
# import folium
# from folium.plugins import HeatMap, MarkerCluster
# from streamlit_folium import folium_static
# import plotly.express as px

# # Initialize Google Gemini 2.0 API
# api_key = 'AIzaSyA6AgtkkCpe8TIyS4mimA7YJ8e-jF6Jclo'
# llm = ChatGoogleGenerativeAI(api_key=api_key, model="gemini-pro")

# # Load healthcare data
# data_path = '/Users/saloni/Downloads/PublicHealthMonitoringThroughRetinalImaging.csv'
# try:
#     healthcare_data = pd.read_csv(data_path)
# except FileNotFoundError:
#     st.error("CSV file not found! Please check the file path.")
#     st.stop()

# # Check for required columns
# required_columns = ['Region', 'Referral_Urgency', 'Avg_Transfer_Time_Minutes', 'Treatment_Success_Rate', 'Center_Operational_Capacity']
# missing_columns = [col for col in required_columns if col not in healthcare_data.columns]
# if missing_columns:
#     st.error(f"Missing columns in the dataset: {', '.join(missing_columns)}")
#     st.stop()

# # Handle missing values in numeric columns
# numeric_columns = ['Referral_Urgency', 'Avg_Transfer_Time_Minutes', 'Treatment_Success_Rate', 'Center_Operational_Capacity']
# for col in numeric_columns:
#     healthcare_data[col] = pd.to_numeric(healthcare_data[col], errors='coerce')
# healthcare_data.fillna(0, inplace=True)

# # Approximate coordinates for regions (replace with actual latitude/longitude if available)
# region_coordinates = {
#     "North": [41.8781, -87.6298],  # Example: Chicago, IL
#     "South": [29.7604, -95.3698],  # Example: Houston, TX
#     "East": [40.7128, -74.0060],   # Example: New York City, NY
#     "West": [34.0522, -118.2437],  # Example: Los Angeles, CA
#     "Central": [39.0997, -94.5786] # Example: Kansas City, MO
# }

# def create_map(data):
#     m = folium.Map(location=[39.8283, -98.5795], zoom_start=4)
    
#     # Prepare heatmap data
#     heat_data = []
#     for region, coords in region_coordinates.items():
#         filtered_data = data[data['Region'] == region]
#         avg_capacity = filtered_data['Center_Operational_Capacity'].mean()
#         avg_waiting_time = filtered_data['Avg_Transfer_Time_Minutes'].mean()
#         success_rate = filtered_data['Treatment_Success_Rate'].mean()
        
#         popup_text = f"""
#         <b>Region:</b> {region}<br>
#         <b>Avg Operational Capacity:</b> {avg_capacity:.2f}<br>
#         <b>Avg Waiting Time:</b> {avg_waiting_time:.2f} mins<br>
#         <b>Treatment Success Rate:</b> {success_rate:.2f}%
#         """
        
#         folium.Marker(
#             location=coords,
#             popup=popup_text,
#             tooltip=f"{region} Region"
#         ).add_to(m)
        
#         # Add to heatmap data (latitude, longitude, weight)
#         heat_data.append([coords[0], coords[1], avg_capacity])
    
#     # Add heatmap layer
#     HeatMap(heat_data).add_to(m)
    
#     return m

# def create_time_series(data, x_col, y_col, title):
#     grouped_data = data.groupby(x_col)[y_col].mean().reset_index()
#     fig = px.bar(grouped_data, x=x_col, y=y_col, title=title,
#                  labels={x_col: "Region", y_col: y_col.replace("_", " ")})
#     fig.update_layout(autosize=True)
#     return fig

# # Streamlit page configuration
# st.set_page_config(page_title="Public Healthcare Dashboard", layout="wide")
# st.title("Public Healthcare Monitoring Dashboard")

# # Display dataset in Streamlit
# st.subheader("Healthcare Referral Data")
# st.dataframe(healthcare_data)

# # Choose an analysis type
# analysis_type = st.selectbox("Choose an Analysis Type:", ["Trend Analysis", "Heat Map", "Service Gap Identification"])

# if analysis_type == "Heat Map":
#     st.subheader("Healthcare Centers Heat Map")
#     map_description = """
#     The heat map below shows the density of healthcare centers based on their operational capacity.
    
#     **Legend**:
#       - Bright Red: High operational capacity.
#       - Dimmer Colors: Lower operational capacity.
      
#     This visualization helps identify regions with more resources and those that may need additional support.
#     """
#     st.markdown(map_description)
    
#     map = create_map(healthcare_data)
#     folium_static(map)

# elif analysis_type == "Trend Analysis":
#     st.subheader("Trends in Healthcare Data")
    
#     time_series_option = st.selectbox(
#         "Choose a metric to visualize:",
#         [ "Avg Transfer Time (Minutes)", "Treatment Success Rate"]
#     )

        
#     if time_series_option == "Avg Transfer Time (Minutes)":
#         fig = create_time_series(healthcare_data, x_col="Region", y_col="Avg_Transfer_Time_Minutes", title="Average Transfer Time by Region")
#         description = """
#         The bar chart below shows the average transfer time across different regions.
        
#         **Insight**:
#           - Longer transfer times may indicate logistical challenges or resource shortages.
#           - Use this chart to optimize transfer operations.
#         """
#         st.markdown(description)
        
#     else:
#         fig = create_time_series(healthcare_data, x_col="Region", y_col="Treatment_Success_Rate", title="Treatment Success Rate by Region")
#         description = """
#         The bar chart below shows the average treatment success rate across different regions.
        
#         **Insight**:
#           - Higher success rates indicate better healthcare outcomes in those regions.
#           - Regions with lower success rates may require targeted interventions or specialized care.
#         """
#         st.markdown(description)
    
#     st.plotly_chart(fig)

# elif analysis_type == "Service Gap Identification":
#     st.subheader("Service Gap Identification")
#     gap_description = """
#     This section identifies potential service gaps based on operational capacity and waiting times.
    
#     Use this analysis to find regions that need additional resources or improvements in efficiency.
    
#     **Example Insights**:
#       - Regions with high waiting times but low operational capacity may need more resources.
#       - Regions with low success rates may require specialized care or equipment upgrades.
      
#       This analysis helps prioritize resource allocation and improve healthcare outcomes.
#     """
#     st.markdown(gap_description)


import streamlit as st
import pandas as pd
import numpy as np
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import folium_static
import plotly.express as px

# Initialize Google Gemini 2.0 API
api_key = 'AIzaSyA6AgtkkCpe8TIyS4mimA7YJ8e-jF6Jclo'
llm = ChatGoogleGenerativeAI(api_key=api_key, model="gemini-pro")

# Load healthcare data
data_path = '/Users/saloni/Downloads/PublicHealthMonitoringThroughRetinalImaging.csv'
try:
    healthcare_data = pd.read_csv(data_path)
except FileNotFoundError:
    st.error("CSV file not found! Please check the file path.")
    st.stop()

# Check for required columns
required_columns = ['Region', 'Referral_Urgency', 'Avg_Transfer_Time_Minutes', 'Treatment_Success_Rate', 'Center_Operational_Capacity']
missing_columns = [col for col in required_columns if col not in healthcare_data.columns]
if missing_columns:
    st.error(f"Missing columns in the dataset: {', '.join(missing_columns)}")
    st.stop()

# Handle missing values in numeric columns
numeric_columns = ['Referral_Urgency', 'Avg_Transfer_Time_Minutes', 'Treatment_Success_Rate', 'Center_Operational_Capacity']
for col in numeric_columns:
    healthcare_data[col] = pd.to_numeric(healthcare_data[col], errors='coerce')
healthcare_data.fillna(0, inplace=True)

# Approximate coordinates for regions (replace with actual latitude/longitude if available)
region_coordinates = {
    "North": [41.8781, -87.6298],  # Example: Chicago, IL
    "South": [29.7604, -95.3698],  # Example: Houston, TX
    "East": [40.7128, -74.0060],   # Example: New York City, NY
    "West": [34.0522, -118.2437],  # Example: Los Angeles, CA
    "Central": [39.0997, -94.5786] # Example: Kansas City, MO
}

def create_map(data):
    m = folium.Map(location=[39.8283, -98.5795], zoom_start=4)
    
    # Prepare heatmap data
    heat_data = []
    for region, coords in region_coordinates.items():
        filtered_data = data[data['Region'] == region]
        avg_capacity = filtered_data['Center_Operational_Capacity'].mean()
        avg_waiting_time = filtered_data['Avg_Transfer_Time_Minutes'].mean()
        success_rate = filtered_data['Treatment_Success_Rate'].mean()
        
        popup_text = f"""
        <b>Region:</b> {region}<br>
        <b>Avg Operational Capacity:</b> {avg_capacity:.2f}<br>
        <b>Avg Waiting Time:</b> {avg_waiting_time:.2f} mins<br>
        <b>Treatment Success Rate:</b> {success_rate:.2f}%
        """
        
        folium.Marker(
            location=coords,
            popup=popup_text,
            tooltip=f"{region} Region"
        ).add_to(m)
        
        # Add to heatmap data (latitude, longitude, weight)
        heat_data.append([coords[0], coords[1], avg_capacity])
    
    # Add heatmap layer
    HeatMap(heat_data).add_to(m)
    
    return m

def create_time_series(data, x_col, y_col, title):
    grouped_data = data.groupby(x_col)[y_col].mean().reset_index()
    fig = px.bar(grouped_data, x=x_col, y=y_col, title=title,
                 labels={x_col: "Region", y_col: y_col.replace("_", " ")})
    fig.update_layout(autosize=True)
    return fig

# Streamlit page configuration
st.set_page_config(page_title="Public Healthcare Dashboard", layout="wide")
st.title("Public Healthcare Monitoring Dashboard")

# Display dataset in Streamlit
st.subheader("Healthcare Referral Data")
st.dataframe(healthcare_data)

# Choose an analysis type
analysis_type = st.selectbox("Choose an Analysis Type:", ["Trend Analysis", "Heat Map", "Service Gap Identification"])

if analysis_type == "Heat Map":
    st.subheader("Healthcare Centers Heat Map")
    
    map_description = """
    The heat map below shows the density of healthcare centers based on their operational capacity.
    
    **Legend**:
      - Bright Red: High operational capacity.
      - Dimmer Colors: Lower operational capacity.
      
    This visualization helps identify regions with more resources and those that may need additional support.
    """
    st.markdown(map_description)
    
    map = create_map(healthcare_data)
    folium_static(map)

elif analysis_type == "Trend Analysis":
    st.subheader("Trends in Healthcare Data")
    
    time_series_option = st.selectbox(
        "Choose a metric to visualize:",
        ["Avg Transfer Time (Minutes)", "Treatment Success Rate"]
    )
    
    if time_series_option == "Avg Transfer Time (Minutes)":
        fig = create_time_series(healthcare_data, x_col="Region", y_col="Avg_Transfer_Time_Minutes", title="Average Transfer Time by Region")
        
        description = """
        The bar chart below shows the average transfer time across different regions.
        
        **Insight**:
          - Longer transfer times may indicate logistical challenges or resource shortages.
          - Use this chart to optimize transfer operations.
        """
        st.markdown(description)
        
    else:
        fig = create_time_series(healthcare_data, x_col="Region", y_col="Treatment_Success_Rate", title="Treatment Success Rate by Region")
        
        description = """
        The bar chart below shows the average treatment success rate across different regions.
        
        **Insight**:
          - Higher success rates indicate better healthcare outcomes in those regions.
          - Regions with lower success rates may require targeted interventions or specialized care.
        """
        st.markdown(description)
    
    st.plotly_chart(fig)

elif analysis_type == "Service Gap Identification":
    # AI insights generation using LangChain and Google Gemini API
    prompt = """
    Given the following dataset containing public healthcare referral data, 
    perform the following tasks:
    1. Identify potential service gaps in the healthcare centers based on operational capacity, waiting time, and specialist availability.
    Dataset:
    {data}   
    """
    prompt_format = prompt.format(data=healthcare_data.to_json(orient="split"))
    messages = [HumanMessage(content=prompt_format)]
    response = llm.invoke(messages)
    st.subheader("Service Gap Identification")
    st.write(response.content)
    




  


