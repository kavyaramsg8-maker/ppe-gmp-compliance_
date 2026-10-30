import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import os

# --- CONFIGURATION ---
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "cloud_data", "ppe_cloud.db")

st.set_page_config(page_title="GMP PPE Dashboard", layout="wide")
st.title("🏭 GMP PPE Monitoring Dashboard")
st.markdown("Live cloud monitoring of PPE compliance.")

def load_data():
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    
    try:
        conn = sqlite3.connect(DB_PATH)
        # Fetching the last 100 events
        query = "SELECT * FROM events ORDER BY event_id DESC LIMIT 100"
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error reading database: {e}")
        return pd.DataFrame()

# Fetch data
df = load_data()

if df.empty:
    st.warning("No events recorded yet. Start the camera and trigger a violation!")
else:
    # Create a combined column for better charts (e.g., "Missing glove")
    if 'type' in df.columns and 'item' in df.columns:
        df['Violation Type'] = df.apply(lambda x: f"{x['type']} - {x['item']}" if pd.notnull(x['item']) else x['type'], axis=1)
    else:
        df['Violation Type'] = df['type']

    # Top Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Events", len(df))
    
    # Most common violation
    if not df['Violation Type'].mode().empty:
        col2.metric("Most Common", df['Violation Type'].mode()[0])
        
    if 'camera_id' in df.columns:
        col3.metric("Cameras Active", df['camera_id'].nunique())

    st.markdown("---")

    # Charts
    st.subheader("📊 Event Statistics")
    chart_col1, chart_col2 = st.columns(2)
    
    # Chart 1: Specific Violations
    fig1 = px.bar(df['Violation Type'].value_counts().reset_index(), 
                  x='Violation Type', y='count', 
                  title="Specific Violations Detected", color='Violation Type')
    chart_col1.plotly_chart(fig1, use_container_width=True)

    # Chart 2: Events over time (using recorded_at)
    time_col = 'recorded_at' if 'recorded_at' in df.columns else 'captured_at'
    if time_col in df.columns:
        try:
            df[time_col] = pd.to_datetime(df[time_col])
            # Round to the nearest minute for grouping
            df_time = df.groupby([df[time_col].dt.strftime('%Y-%m-%d %H:%M')]).size().reset_index(name='count')
            fig2 = px.line(df_time, x=time_col, y='count', title="Events Over Time")
            chart_col2.plotly_chart(fig2, use_container_width=True)
        except Exception as e:
            chart_col2.info(f"Could not plot time chart: {e}")

    st.markdown("---")

    # Data Table with Snapshots
    st.subheader("🚨 Recent Events (with Snapshots)")
    
    # Try to show images if the snapshot column contains file paths
    if 'snapshot' in df.columns:
        for index, row in df.head(5).iterrows(): # Show top 5 for performance
            with st.expander(f"{row['Violation Type']} on {row['camera_id']} at {row[time_col]}"):
                st.write(f"**Track ID:** {row.get('track_id', 'N/A')}")
                
                # If snapshot is a file path, try to display it
                snapshot_path = row['snapshot']
                if isinstance(snapshot_path, str) and os.path.exists(snapshot_path):
                    st.image(snapshot_path, caption="Violation Snapshot")
                elif isinstance(snapshot_path, str) and os.path.exists(os.path.join(os.path.dirname(__file__), "..", snapshot_path)):
                    st.image(os.path.join(os.path.dirname(__file__), "..", snapshot_path), caption="Violation Snapshot")
                else:
                    st.info("Snapshot path is not directly accessible from this folder.")
    else:
        st.dataframe(df, use_container_width=True)

# Add a refresh button
if st.button("🔄 Refresh Data"):
    st.rerun()