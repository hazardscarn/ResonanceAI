import streamlit as st
import pandas as pd
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import numpy as np


def create_heatmap(df, value_col):
    center_lat, center_lon = df['latitude'].mean(), df['longitude'].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
    heat_data = [[row['latitude'], row['longitude'], row[value_col]] for _, row in df.iterrows()]
    HeatMap(heat_data, radius=25, blur=15).add_to(m)
    return m

def create_segment_map(df):
    center_lat, center_lon = df['latitude'].mean(), df['longitude'].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
    
    colors = {'HA-HP': '#00FF00', 'HA-LP': '#0000FF', 'LA-HP': '#FFFF00', 'LA-LP': '#FF0000'}
    
    for _, row in df.iterrows():
        popup_text = f"""
        <b>{row['strategy']}</b><br>
        Affinity: {row['affinity']:.3f}<br>
        Popularity: {row['popularity']:.3f}
        """
        
        folium.CircleMarker(
            [row['latitude'], row['longitude']],
            radius=8,
            color=colors[row['segment']],
            fillColor=colors[row['segment']],
            fillOpacity=0.8,
            popup=popup_text
        ).add_to(m)
    return m


def process_data(conservative_df,progressive_df):

    conservative_df1=conservative_df.copy()
    progressive_df1=progressive_df.copy()

    conservative_df1.rename({'affinity':'conservative_affinity','popularity':'conservative_popularity'},axis=1,inplace=True)
    progressive_df1.rename({'affinity':'progressive_affinity','popularity':'progressive_popularity'},axis=1,inplace=True)

    df2=pd.merge(conservative_df1,progressive_df1[['latitude', 'longitude', 'progressive_affinity', 'progressive_popularity']],
                              on=['latitude', 'longitude'], how='inner')
    df2['conservative_net_popularity']=(df2['conservative_popularity']-df2['progressive_popularity'])*100
    df2['progressive_net_popularity']=(df2['progressive_popularity']-df2['conservative_popularity'])*100
    

    return df2
