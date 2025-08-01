import requests
import json
from typing import Dict, List, Optional, Any,Union
import time
from dataclasses import dataclass
from pprint import pprint
import logging
import pandas as pd
import numpy as np

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.qloo import QlooAPIClient, QlooSignals, QlooAudience
from src.secret_manager import SecretManager,SecretConfig
from src.heatmap import get_complete_heatmap_analysis


project_id = SecretConfig.get_google_cloud_project()
location = SecretConfig.get_google_cloud_location()
qloo_api_key = SecretConfig.get_qloo_api_key()
client = QlooAPIClient(api_key=qloo_api_key)
logger = logging.getLogger(__name__)


# location_query: str,
#     entity_names: Optional[Union[str, List[str]]] = None,
#     tag_names: Optional[Union[str, List[str]]] = None,
#     audience_ids: Optional[List[str]] = None,  # NEW: Add audience_ids parameter
#     audience_weight: Optional[float] = None,   # NEW: Add audience_weight parameter
#     age: Optional[str] = None,
#     gender: Optional[str] = None,
#     boundary: Optional[str] = None,
#     bias_trends: Optional[str] = None,
#     limit: int = 50

def get_candidate_base(candidate_name: str,
                       opponent_name: str,
                       location_query: str,
                       age: Optional[str] = None,
                       gender: Optional[str] = None,
                       limit: int = 50):
    
    """
    Analyzes a candidate's popularity against an opponent in a specific location
    and provides a simple competitive segmentation.

    Args:
        candidate_name: The name of the primary candidate.
        opponent_name: The name of the opponent.
        location_query: The geographical area to analyze.
        age: Optional age filter.
        gender: Optional gender filter.
        limit: The number of data points to retrieve.

    Returns:
        A pandas DataFrame with a comparative popularity analysis of candidate to an opponent
        Exception if the analysis fails.
    """
    for attempt in range(2): # Retry once
        try:
            # Get data for the primary candidate
            candidate_df = get_complete_heatmap_analysis(
                location_query=location_query,
                entity_names=[candidate_name],
                age=age,
                gender=gender,
                limit=limit
            )
            
            # Check if DataFrame is empty or missing required columns
            if candidate_df.empty or 'latitude' not in candidate_df.columns:
                logger.warning(f"No valid data found for candidate: {candidate_name}")
                return pd.DataFrame()
            
            # Correctly drop rows where key metrics are missing
            candidate_df = candidate_df.dropna(subset=['popularity', 'latitude', 'longitude'])
            if candidate_df.empty:
                logger.warning(f"No data with valid coordinates found for candidate: {candidate_name}")
                return pd.DataFrame()
            
            # Get data for the opponent
            opponent_df = get_complete_heatmap_analysis(
                location_query=location_query,
                entity_names=[opponent_name],
                age=age,
                gender=gender,
                limit=limit
            )
            
            # Check if DataFrame is empty or missing required columns
            if opponent_df.empty or 'latitude' not in opponent_df.columns:
                logger.warning(f"No valid data found for opponent: {opponent_name}")
                return pd.DataFrame()
            
            opponent_df = opponent_df.rename(columns={'popularity': 'opponent_popularity'})
            opponent_df = opponent_df[['latitude', 'longitude', 'opponent_popularity']].dropna()
            if opponent_df.empty:
                logger.warning(f"No data with valid coordinates found for opponent: {opponent_name}")
                return pd.DataFrame()

            # Use an inner merge to ensure we only compare locations where both have data
            merged_df = pd.merge(candidate_df, opponent_df, on=['latitude', 'longitude'], how='inner')

            if merged_df.empty:
                logger.warning("No overlapping locations found between candidate and opponent.")
                return pd.DataFrame()

            # Calculate net popularity score (candidate vs opponent)
            merged_df['net_popularity'] = (merged_df['popularity'] - merged_df['opponent_popularity']) * 100

            # Define thresholds for simple popularity comparison
            WIN_THRESHOLD = 2.0
            LOSE_THRESHOLD = -2.0

            # Create simple segments based on net popularity
            conditions = [
                merged_df['net_popularity'] > WIN_THRESHOLD,   # Candidate is clearly leading
                merged_df['net_popularity'] < LOSE_THRESHOLD   # Candidate is clearly trailing
            ]
            choices = [
                'Leading Opponent',
                'Trailing Opponent'
            ]
            # Anything between the thresholds is 'Competitive'
            merged_df['popularity_status'] = np.select(conditions, choices, default='Similar to Opponent')
            
            return merged_df

        except Exception as e:
            if attempt == 0:
                time.sleep(2) # Wait for 2 seconds before retrying
            else:
                logger.error(f"Failed to get candidate base analysis after retries for '{candidate_name}': {e}")
                # Return empty DataFrame instead of raising exception
                return pd.DataFrame()
            

def get_political_base(candidate_base: str,
                       location_query: str,
                       age: Optional[str] = None,
                       gender: Optional[str] = None,
                       limit: int = 50):
    
    """
    Collects the popularity of candidates base at a given location and demographics 
    Args:
        candidate_base: The base Candidate Belongs to (progressive,conservative,center)
        location_query: The geographical area to analyze.
        age: Optional age filter.
        gender: Optional gender filter.

    Returns:
        A pandas DataFrame with the popularity of the political base of the candidate
        Exception if the analysis fails.
    """

    for attempt in range(2): # Retry once
        try:
            if candidate_base.lower() == 'progressive':
                base_audience = ['urn:audience:political_preferences:politically_progressive']
            elif candidate_base.lower() == 'conservative':
                base_audience = ['urn:audience:political_preferences:politically_conservative']
            else: # Default to center if not progressive or conservative
                base_audience = ['urn:audience:political_preferences:center']

            df1 = get_complete_heatmap_analysis(
                        location_query=location_query,
                        audience_ids=base_audience,
                        age=age,
                        gender=gender,
                        limit=limit # Pass limit to the heatmap analysis
                    )
            
            # Check if DataFrame is empty or missing required columns
            if df1.empty or 'latitude' not in df1.columns:
                logger.warning(f"No valid data found for political base: {candidate_base}")
                return pd.DataFrame()
            
            df1 = df1.dropna(subset=['popularity', 'latitude', 'longitude'])
            if df1.empty:
                logger.warning(f"No data with valid coordinates found for political base: {candidate_base}")
                return pd.DataFrame()
                
            df1.rename(columns={'popularity': 'base_popularity'}, inplace=True)
            df1 = df1[['latitude', 'longitude', 'base_popularity']]

            return df1
        except Exception as e:
            if attempt == 0:
                time.sleep(2) # Wait for 2 seconds before retrying
            else:
                logger.error(f"Failed to get political base analysis after retries for '{candidate_base}': {e}")
                # Return empty DataFrame instead of raising exception
                return pd.DataFrame()
            

def get_targeted_base(candidate_name: str,
                    tag_name: str,
                       location_query: str,
                       age: Optional[str] = None,
                       gender: Optional[str] = None,
                       limit: int = 50):
    
    """
    Collects the popularity of the candidate among a given tag group
    and for a given demographic.
    Args:
        candidate_name: The name of the primary candidate
        tag_name: The tag group that we want to target (e.g., 'sports', 'outdoors').
        location_query: The geographical area to analyze.
        age: Optional age filter.
        gender: Optional gender filter.
        limit: The number of data points to retrieve.

    Returns:
        A pandas DataFrame with popularity data for the candidate in the tag group, or an
        Exception if the analysis fails.
    """
    # Ensure tag_name is always a list for get_complete_heatmap_analysis
    tag_names_list = [tag_name] if isinstance(tag_name, str) else tag_name

    for attempt in range(2): # Retry once
        try:
            df1 = get_complete_heatmap_analysis(
                        location_query=location_query,
                        tag_names=tag_names_list,
                        age=age,
                        gender=gender,
                        limit=limit
                    )
            
            # Check if DataFrame is empty or missing required columns
            if df1.empty or 'latitude' not in df1.columns:
                logger.warning(f"No valid data found for tag: {tag_name}")
                return pd.DataFrame()
            
            if df1.empty:
                logger.warning(f"No data returned from heatmap analysis for tags: {tag_name}")
                return pd.DataFrame() # Return empty DataFrame on no data

            df1 = df1.dropna(subset=['popularity', 'latitude', 'longitude'])
            if df1.empty:
                logger.warning(f"No data with valid coordinates found for tag: {tag_name}")
                return pd.DataFrame()
            
            # Create a descriptive and valid column name from the list of tags
            df1.rename(columns={'popularity': 'tag_popularity'}, inplace=True)
            df1['tag_name'] = "tag_" + tag_name.replace(' ', '_')
            df1 = df1[['latitude', 'longitude', 'tag_name','tag_popularity']]

            return df1
        except Exception as e:
            if attempt == 0:
                time.sleep(2) # Wait for 2 seconds before retrying
            else:
                logger.error(f"Failed to get targeted base analysis after retries for tags '{tag_name}': {e}")
                # Return empty DataFrame instead of raising exception
                return pd.DataFrame()