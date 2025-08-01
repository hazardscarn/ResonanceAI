import requests
import json
from typing import Dict, List, Optional, Any,Union
from dataclasses import dataclass
from pprint import pprint
import logging
import pandas as pd
import numpy as np
from google.adk.tools import ToolContext
from google.genai import types
import io

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.qloo import QlooAPIClient, QlooSignals, QlooAudience
from src.secret_manager import SecretManager,SecretConfig
from .subtools import get_candidate_base,get_political_base,get_targeted_base


project_id = SecretConfig.get_google_cloud_project()
location = SecretConfig.get_google_cloud_location()
qloo_api_key = SecretConfig.get_qloo_api_key()
client = QlooAPIClient(api_key=qloo_api_key)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)


# ================================
# TOOL 1: CREATE CANDIDATE ANALYSIS
# ================================

async def create_candidate_analysis(
    tool_context: ToolContext,
    candidate_name: str,
    opponent_name: str,
    candidate_base: str,
    location: str,
    age: Optional[str] = None,
    gender: Optional[str] = None,
    tag_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Performs comprehensive political candidate analysis comparing popularity across demographics and issues.
    
    [Previous docstring content remains the same...]
    """
    
    # Validate required parameters
    if not all([candidate_name, opponent_name, candidate_base, location]):
        raise ValueError("candidate_name, opponent_name, candidate_base, and location are required")
    
    # Validate age parameter if provided
    valid_ages = [
        "24_and_younger", "25_to_29", "30_to_34", "35_and_younger", 
        "36_to_55", "35_to_44", "45_to_54", "55_and_older"
    ]
    if age and age not in valid_ages:
        raise ValueError(f"age must be one of: {', '.join(valid_ages)}")
    
    # Validate gender parameter if provided
    if gender and gender not in ["male", "female"]:
        raise ValueError("gender must be 'male', 'female', or None")
    
    # Set default empty list for tag_names if not provided
    if tag_names is None:
        tag_names = []
    
    logger.info(f"Starting candidate analysis for {candidate_name} vs {opponent_name} in {location}")
    
    try:
        # Call the existing analysis functions
        opponent_analysis = get_candidate_base(
            candidate_name=candidate_name,
            opponent_name=opponent_name,
            location_query=location,
            age=age,
            gender=gender
        )
        logger.info(f"Opponent Analysis completed: {opponent_analysis.shape[0]} segments")
        
        # Check if opponent analysis is empty
        if opponent_analysis.empty:
            return {
                "status": "error",
                "message": f"No overlapping data found between {candidate_name} and {opponent_name} in {location}. This could be due to location not being recognized or candidates not having enough data in this area.",
                "suggestions": [
                    "Try a different location (e.g., 'New York' instead of 'NYC')",
                    "Check candidate name spelling",
                    "Try a broader geographic area",
                    "Verify the location exists in the database"
                ]
            }
        
        base_analysis = get_political_base(
            candidate_base=candidate_base,
            location_query=location,
            age=age,
            gender=gender
        )
        logger.info(f"Base Analysis completed: {base_analysis.shape[0]} segments")
        
        # Check if base analysis is empty
        if base_analysis.empty:
            return {
                "status": "error",
                "message": f"No political base data found for {candidate_base} in {location}.",
                "suggestions": [
                    "Try a different location",
                    "Check if the political base is valid (progressive, conservative, center)",
                    "Try a broader geographic area"
                ]
            }

        # Process tag analysis if tags provided
        tag_analysis_df = pd.DataFrame()
        if tag_names:
            for tag_name in tag_names:
                tag_analysis = get_targeted_base(
                    candidate_name=candidate_name,
                    tag_name=tag_name,
                    location_query=location,
                    age=age,
                    gender=gender
                )
                
                logger.info(f"Tag Analysis for {tag_name} completed: {tag_analysis.shape[0]} segments")

                if not tag_analysis.empty:
                    tag_analysis_df = pd.concat([tag_analysis_df, tag_analysis], ignore_index=True)

            if not tag_analysis_df.empty:
                tag_analysis_df = tag_analysis_df.pivot_table(
                    index=['latitude', 'longitude'], 
                    columns='tag_name', 
                    values='tag_popularity', 
                    aggfunc='first'
                ).reset_index()

        # Merge all analysis dataframes
        df1 = pd.merge(opponent_analysis, base_analysis, on=['latitude', 'longitude'], how='left')
        df4 = pd.merge(df1, tag_analysis_df, on=['latitude', 'longitude'], how='left')

        # Check if final merged dataframe is empty
        if df4.empty:
            return {
                "status": "error",
                "message": f"No merged data available for analysis in {location}. The datasets may not have overlapping geographic coverage.",
                "suggestions": [
                    "Try a different location with more data coverage",
                    "Use broader demographic filters",
                    "Try different candidates that may have more data"
                ]
            }

        # Apply analysis logic
        WIN_THRESHOLD = 2.0
        LOSE_THRESHOLD = -2.0

        # Calculate base popularity comparison
        df4['net_base_popularity'] = (df4['popularity'] - df4['base_popularity']) * 100
        
        conditions = [
            df4['net_base_popularity'] > WIN_THRESHOLD,
            df4['net_base_popularity'] < LOSE_THRESHOLD
        ]
        choices = [
            'More Popular than Party',
            'Less Popular than Party'
        ]
        df4['base_popularity_status'] = np.select(conditions, choices, default='Similar Popularity to Party')

        # Process tag columns for issue-based analysis
        tag_cols = [col for col in df4.columns if col.startswith('tag_')]
        
        for tag_col in tag_cols:
            tag_name = tag_col.replace('tag_', '')
            net_col = f'net_{tag_name}_popularity'
            base_col = f'base_{tag_name}_popularity'
            
            if base_col in df4.columns:
                df4[net_col] = (df4[tag_col] - df4[base_col]) * 100
            else:
                df4[net_col] = (df4[tag_col] - df4[tag_col].mean()) * 100
            
            conditions = [
                df4[net_col].isna(),
                df4[net_col] > WIN_THRESHOLD,
                df4[net_col] < LOSE_THRESHOLD
            ]

            choices = [
                'Unknown',
                'Candidate More Popular with this Issue',
                'Candidate Less Popular with this Issue'
            ]
            
            status_col = f'issue_{tag_name}_popularity_status'
            df4[status_col] = np.select(conditions, choices, default='Similar Popularity')

        logger.info(f"Analysis completed with {df4.shape[0]} segments and {df4.shape[1]} columns")

        # Save DataFrame as artifact
        df_buffer = io.BytesIO()
        df4.to_pickle(df_buffer)
        df_bytes = df_buffer.getvalue()
        
        artifact_part = types.Part(
            inline_data=types.Blob(
                mime_type="application/octet-stream",
                data=df_bytes
            )
        )
        
        # Create descriptive filename
        safe_candidate = candidate_name.lower().replace(' ', '_').replace('.', '')
        safe_location = location.lower().replace(' ', '_').replace('.', '')
        filename = f"candidate_analysis_{safe_candidate}_{safe_location}.pkl"
        
        version = await tool_context.save_artifact(filename=filename, artifact=artifact_part)
        logger.info(f"Successfully saved analysis as artifact '{filename}' version {version}")
        
        # Store references in state
        tool_context.state["temp:candidate_analysis_artifact"] = filename
        tool_context.state["temp:analysis_metadata"] = {
            "candidate_name": candidate_name,
            "opponent_name": opponent_name,
            "candidate_base": candidate_base,
            "location": location,
            "age": age,
            "gender": gender,
            "tag_names": tag_names,
            "rows": df4.shape[0],
            "columns": df4.shape[1],
            "artifact_version": version
        }
        
        # Calculate segment counts safely
        rally_base_df = df4[df4['strategy']=='Rally the Base'].shape[0] if 'strategy' in df4.columns else 0
        hidden_goldmine_df = df4[df4['strategy']=='Hidden Goldmine'].shape[0] if 'strategy' in df4.columns else 0
        bring_them_over_df = df4[df4['strategy']=='Bring Them Over'].shape[0] if 'strategy' in df4.columns else 0
        deep_conversion_df = df4[df4['strategy']=='Deep Conversion'].shape[0] if 'strategy' in df4.columns else 0

        return {
            "status": "success",
            "message": f"Candidate analysis completed for {candidate_name} vs {opponent_name} in {location}",
            "artifact_filename": filename,
            "artifact_version": version,
            "analysis_summary": {
                "total_locations": df4.shape[0],
                "data_columns": df4.shape[1],
                "candidate": candidate_name,
                "opponent": opponent_name,
                "candidate_base": candidate_base,
                "location": location,
                "demographics": {
                    "age": age or "all_ages",
                    "gender": gender or "all_genders"
                },
                "tags_analyzed": tag_names,
                "rally_base": f"""Rally Base: Those location segment where there is high affinity and high popularity for Candidate.
                                    This is your main vote base. There are {rally_base_df} rally base locations ({rally_base_df/df4.shape[0]*100:.1f}% of total locations)""",
                "bring_them_over":f"""Bring Them Over: Those location segment where there is low affinity and high popularity for Candidate.
                                    This could be opponent's playground but the candidate is really popular here. You can campaign on the candidates appeal here more.
                                    There are {bring_them_over_df} Bring Them Over locations ({bring_them_over_df/df4.shape[0]*100:.1f}% of total locations)""",
                "hidden_goldmine":f"""Hidden Gold: Those location segment where there is high affinity but low popularity for Candidate.
                                    This is the segment that needs the Candidate introduced more. They need to know him better. There are {hidden_goldmine_df} rally base locations ({hidden_goldmine_df/df4.shape[0]*100:.1f}% of total locations)""",
                "deep_conversion":f"""Deep Conversion: Those location segment where there is low affinity and low popularity for Candidate.
                                    This is probably opponent base. There are {deep_conversion_df} rally base locations ({deep_conversion_df/df4.shape[0]*100:.1f}% of total locations)""",
  
            },
            "next_steps": [
                f"Further targeted steps can be done to identify whom to target for what for these 2 segments only: Rally Base, Hidden Goldmine"
            ]
        }
        
    except Exception as e:
        logger.error(f"Error in candidate analysis: {e}")
        return {
            "status": "error",
            "message": f"Analysis failed: {str(e)}",
            "error_details": str(e),
            "suggestions": [
                "Check that the location name is valid and recognizable (e.g., 'Phoenix, Arizona' instead of 'Philadephia')",
                "Verify candidate names are spelled correctly",
                "Try using a broader geographic area",
                "Check the logs for more specific error details"
            ]
        }

#######################################
#TOOL 3 - FIND THE SEGMENTS TO TARGET
#########################################

async def identify_rally_segments(
    tool_context: ToolContext,
    artifact_filename: Optional[str] = None,
    affinity_threshold: float = 0.6,
    popularity_threshold: float = 0.5
) -> Dict[str, Any]:
    """
    This tool loads the detailed analysis artifact and identifies different subsets/groups to target on Rally the base segement  
    
    Args:
        artifact_filename (str, optional): Specific artifact file to analyze.
                                         If not provided, uses the most recent analysis from state.
        affinity_threshold (float): Minimum affinity score for Rally the Base (default: 0.6)
        popularity_threshold (float): Minimum popularity score for Rally the Base (default: 0.5)
    
    Returns:
        Dict[str, Any]: Comprehensive targeting analysis with actionable segments
    """
    
    try:
        # Get artifact filename from parameter or state
        if artifact_filename is None:
            artifact_filename = tool_context.state.get("temp:candidate_analysis_artifact")
            if not artifact_filename:
                return {
                    "status": "error",
                    "message": "No candidate analysis found. Please run create_candidate_analysis first."
                }
        
        # Load analysis metadata from state
        metadata = tool_context.state.get("temp:analysis_metadata", {})
        
        # Load the analysis DataFrame
        artifact = await tool_context.load_artifact(filename=artifact_filename)
        
        if not artifact or not artifact.inline_data:
            return {
                "status": "error", 
                "message": f"Could not load analysis data from '{artifact_filename}'"
            }
        
        import io
        df = pd.read_pickle(io.BytesIO(artifact.inline_data.data))
        logger.info(f"Loaded analysis data: {df.shape[0]} segments, {df.shape[1]} columns")
        
        candidate_name = metadata.get('candidate_name', 'Candidate')
        opponent_name = metadata.get('opponent_name', 'Opponent')
        location = metadata.get('location', 'Location')
        
        # ================================
        # 1. DEFINE RALLY THE BASE SEGMENT
        # ================================
        
        # Rally the Base: High Affinity + High Popularity locations
        
        
        rally_base_df = df[df['strategy']=='Rally the Base'].copy()
        total_rally_base = len(rally_base_df)
        total_segments = len(df)
        
        if total_rally_base == 0:
            return {
                "status": "warning",
                "message": f"No Rally the Base segments found with affinity >= {affinity_threshold} and popularity >= {popularity_threshold}",
                "recommendation": "Lower thresholds or check data quality"
            }
        
        logger.info(f"Rally the Base segment: {total_rally_base} locations ({round((total_rally_base/total_segments)*100, 1)}%)")
        
        # ================================
        # 2. TARGET TYPE 1: SWING VOTERS
        # (Rally Base locations where trailing opponent)
        # ================================
        
        # Identify trailing locations within Rally the Base
        
        swing_voters_df = rally_base_df[rally_base_df['popularity_status']=='Trailing Opponent'].copy()
        num_swing_locations = len(swing_voters_df)
        swing_percentage = round((num_swing_locations / total_rally_base) * 100, 1) if total_rally_base > 0 else 0
        
        # Get detailed stats for swing voters
        swing_stats = {}
        if num_swing_locations > 0:
            avg_net_popularity = swing_voters_df.get('net_popularity', swing_voters_df['popularity'] - swing_voters_df.get('opponent_popularity', 0)).mean()
            avg_affinity = swing_voters_df['affinity'].mean()
            avg_popularity = swing_voters_df['popularity'].mean()
            
            
            swing_stats = {
                "total_locations": num_swing_locations,
                "percentage_of_rally_base": swing_percentage,
                "average_net_popularity": round(avg_net_popularity, 2),
                "average_affinity": round(avg_affinity, 2),
                "average_popularity": round(avg_popularity, 2),
                "campaign_strategy": f"Target high-affinity supporters who are currently have {opponent_name} more popular than {candidate_name}",
                "message_focus": "Reinforce commitment to core values while addressing opponent's advantages",
                "urgency": "HIGH" if swing_percentage > 15 else "MEDIUM"
            }
        
        # ================================
        # 3. TARGET TYPE 2: BASE INTRODUCTION
        # (Rally Base locations less popular than party)
        # ================================
        
        base_intro_mask = rally_base_df['base_popularity_status'] == 'Less Popular than Party'
        base_intro_df = rally_base_df[base_intro_mask].copy()
        num_base_intro_locations = len(base_intro_df)
        base_intro_percentage = round((num_base_intro_locations / total_rally_base) * 100, 1) if total_rally_base > 0 else 0
        
        # Get detailed stats for base introduction
        base_intro_stats = {}
        if num_base_intro_locations > 0:
            avg_net_base_popularity = base_intro_df['net_base_popularity'].mean() if 'net_base_popularity' in base_intro_df.columns else 0
            avg_affinity = base_intro_df['affinity'].mean()
            avg_base_popularity = base_intro_df['base_popularity'].mean() if 'base_popularity' in base_intro_df.columns else 0
            
            # Top priority base introduction locations
            top_base_targets = base_intro_df.nlargest(5, 'affinity')[
                ['latitude', 'longitude', 'affinity', 'base_popularity', 'net_base_popularity', 'geohash']
            ].to_dict('records')
            
            base_intro_stats = {
                "total_locations": num_base_intro_locations,
                "percentage_of_rally_base": base_intro_percentage,
                "average_net_base_performance": round(avg_net_base_popularity, 2),
                "average_affinity": round(avg_affinity, 2),
                "average_base_popularity": round(avg_base_popularity, 2),
                "top_priority_targets": top_base_targets,
                "campaign_strategy": "Introduce candidate to party base using party ideology and shared values. This locations have more party supporters who are yet to have the same affection to candidate",
                "message_focus": "Emphasize party alignment, shared goals, and candidate's commitment to base priorities",
                "urgency": "HIGH" if base_intro_percentage > 25 else "MEDIUM"
            }
        
        # ================================
        # 4. TARGET TYPE 3: ISSUE-BASED TARGETING
        # (Rally Base locations weak on specific issues)
        # ================================
        
        issue_targeting = {}
        issue_cols = [col for col in rally_base_df.columns if col.startswith('issue_') and col.endswith('_popularity_status')]
        
        for issue_col in issue_cols:
            issue_name = issue_col.replace('issue_', '').replace('_popularity_status', '')
            
            # Find Rally Base locations where candidate is weak on this issue
            issue_weak_mask = rally_base_df[issue_col] == 'Candidate Less Popular with this Issue'
            issue_weak_df = rally_base_df[issue_weak_mask].copy()
            num_issue_locations = len(issue_weak_df)
            issue_percentage = round((num_issue_locations / total_rally_base) * 100, 1) if total_rally_base > 0 else 0
            
            if num_issue_locations > 0:
                # Get corresponding net popularity column for this issue
                net_issue_col = f'net_{issue_name}_popularity'
                tag_col = f'tag_{issue_name}'
                
                avg_issue_performance = issue_weak_df[net_issue_col].mean() if net_issue_col in issue_weak_df.columns else 0
                avg_affinity = issue_weak_df['affinity'].mean()
                
                # Top priority locations for this issue
                sort_col = 'affinity'  # Default to affinity
                if net_issue_col in issue_weak_df.columns:
                    # Sort by worst performance first (most negative), then by highest affinity
                    issue_weak_df_sorted = issue_weak_df.sort_values([net_issue_col, 'affinity'], ascending=[True, False])
                else:
                    issue_weak_df_sorted = issue_weak_df.sort_values('affinity', ascending=False)
                
                top_issue_targets = issue_weak_df_sorted.head(5)[
                    ['latitude', 'longitude', 'affinity'] + 
                    ([net_issue_col] if net_issue_col in issue_weak_df.columns else []) +
                    ['geohash']
                ].to_dict('records')
                
                # Determine urgency based on issue impact
                urgency = "HIGH" if issue_percentage > 20 else "MEDIUM" if issue_percentage > 10 else "LOW"
                
                issue_targeting[issue_name] = {
                    "total_locations": num_issue_locations,
                    "percentage_of_rally_base": issue_percentage,
                    "average_issue_performance": round(avg_issue_performance, 2),
                    "average_affinity": round(avg_affinity, 2),
                    "top_priority_targets": top_issue_targets,
                    "campaign_strategy": f"Address {issue_name.replace('_', ' ')} concerns with detailed policy positions. These locations have candidate less popular on those flollow this issue",
                    "message_focus": f"Showcase candidate's expertise and commitment on {issue_name.replace('_', ' ')} issues",
                    "urgency": urgency,
                    "recommended_actions": [
                        f"Develop detailed {issue_name.replace('_', ' ')} policy briefs",
                        f"Schedule {issue_name.replace('_', ' ')}-focused events in target areas",
                        f"Target these locations for {issue_name.replace('_', ' ')} related ads"
                    ]
                }
        
        # ================================
        # 5. OVERALL RALLY THE BASE SUMMARY
        # ================================
        
        rally_base_summary = {
            "total_rally_base_locations": total_rally_base,
            "percentage_of_total_segments": round((total_rally_base / total_segments) * 100, 1),
            "affinity_threshold_used": affinity_threshold,
            "popularity_threshold_used": popularity_threshold,
            "average_affinity": round(rally_base_df['affinity'].mean(), 2),
            "average_popularity": round(rally_base_df['popularity'].mean(), 2),
            "geographic_distribution": {
                "strongest_rally_areas": rally_base_df.nlargest(5, 'affinity')[
                    ['latitude', 'longitude', 'affinity', 'popularity', 'geohash']
                ].to_dict('records'),
                "center_coordinates": {
                    "latitude": round(rally_base_df['latitude'].mean(), 4),
                    "longitude": round(rally_base_df['longitude'].mean(), 4)
                }
            }
        }
        
        # ================================
        # 6. CAMPAIGN MANAGER RECOMMENDATIONS
        # ================================
        
        priority_recommendations = []
        resource_allocation = {}
        
        # Prioritize based on urgency and potential impact
        if swing_stats and swing_stats.get("urgency") == "HIGH":
            priority_recommendations.append(f"🚨 URGENT: {num_swing_locations} swing voter locations need immediate attention")
            resource_allocation["swing_voters"] = "40% of Rally Base budget"
        
        if base_intro_stats and base_intro_percentage > 15:
            priority_recommendations.append(f"📢 Base Introduction: {num_base_intro_locations} locations for party alignment messaging")
            resource_allocation["base_introduction"] = "30% of Rally Base budget"
        
        # Issue prioritization
        high_priority_issues = [issue for issue, data in issue_targeting.items() if data["urgency"] == "HIGH"]
        if high_priority_issues:
            priority_recommendations.append(f"📋 Issue Focus: Address {', '.join(high_priority_issues)} in {sum(issue_targeting[i]['total_locations'] for i in high_priority_issues)} locations")
            resource_allocation["issue_campaigns"] = "30% of Rally Base budget"
        
        return {
            "status": "success",
            "message": f"Rally the Base targeting analysis completed for {candidate_name} in {location}",
            "campaign_strategy": "Rally the Base",
            "analysis_metadata": metadata,
            
            # Main targeting segments
            "targeting_segments": {
                "swing_voters": swing_stats,
                "base_introduction": base_intro_stats,
                "issue_targeting": issue_targeting
            },
            
            # Overall Rally the Base insights
            "rally_base_summary": rally_base_summary,
            
            # Campaign manager guidance
            "campaign_recommendations": {
                "priority_actions": priority_recommendations,
                "resource_allocation_suggestions": resource_allocation,
                "total_targetable_locations": num_swing_locations + num_base_intro_locations + sum(data["total_locations"] for data in issue_targeting.values()),
                "campaign_timeline": "2-4 weeks for Rally Base strategy implementation",
                "success_metrics": [
                    "Increase voter turnout in Rally Base segments by 15%",
                    "Improve candidate recognition in base introduction areas",
                    "Address issue concerns in weak performance areas"
                ]
            },
            
            # Next steps
            "next_steps": [
                "Develop targeted messaging for swing voters, base introduction, and issue campaigns. You can ask for detailed targeted campaigns",
            ]
        }
        
    except Exception as e:
        logger.error(f"Error identifying target segments: {e}")
        return {
            "status": "error",
            "message": f"Failed to identify target segments: {str(e)}",
            "error_details": str(e)
        }





async def identify_hidden_goldmine_segments(
    tool_context: ToolContext,
    artifact_filename: Optional[str] = None,
) -> Dict[str, Any]:
    """
    This tool loads the detailed analysis artifact and identifies different subsets/groups to target on Hidden GoldMine segement 
    
    Args:
        artifact_filename (str, optional): Specific artifact file to analyze.
                                         If not provided, uses the most recent analysis from state.
    
    Returns:
        Dict[str, Any]: Comprehensive targeting analysis with actionable segments
    """
    
    try:
        # Get artifact filename from parameter or state
        if artifact_filename is None:
            artifact_filename = tool_context.state.get("temp:candidate_analysis_artifact")
            if not artifact_filename:
                return {
                    "status": "error",
                    "message": "No candidate analysis found. Please run create_candidate_analysis first."
                }
        
        # Load analysis metadata from state
        metadata = tool_context.state.get("temp:analysis_metadata", {})
        
        # Load the analysis DataFrame
        artifact = await tool_context.load_artifact(filename=artifact_filename)
        
        if not artifact or not artifact.inline_data:
            return {
                "status": "error", 
                "message": f"Could not load analysis data from '{artifact_filename}'"
            }
        
        import io
        df = pd.read_pickle(io.BytesIO(artifact.inline_data.data))
        logger.info(f"Loaded analysis data: {df.shape[0]} segments, {df.shape[1]} columns")
        
        candidate_name = metadata.get('candidate_name', 'Candidate')
        opponent_name = metadata.get('opponent_name', 'Opponent')
        location = metadata.get('location', 'Location')
        

        
        #Less Popular than party
        
        
        lpp = df[df['base_popularity_status']=='Less Popular than Party'].copy()
        total_lpp_base = len(lpp)
        total_segments = len(df)
        
        logger.info(f"Hidden Goldmine segment: {lpp.shape[0]} locations ({round((total_lpp_base/total_segments)*100, 1)}%)")
        
        
        hgm_summary = {
            'status':"success",
            "total_hidden_gold_mine_base_locations": total_segments,
            "percentage_of_total_segments_to_target": round((total_lpp_base / total_segments) * 100, 1),
            "message":f"""Target these locations to introduce the candidate among the party base. 
            The candidate is less popular than part in these locations and party is popular in these regions.
            These are votes that will be lost or uncast unless actions is taken""",
            "next_steps": [
                "Introduce the candidate among the party base and make sure they cast vote",
            ]
            
        }
        

        return hgm_summary
        
    except Exception as e:
        logger.error(f"Error identifying target segments: {e}")
        return {
            "status": "error",
            "message": f"Failed to identify target segments: {str(e)}",
            "error_details": str(e)
        }

# Add this function to the end of your tools.py file

async def generate_campaign_report(
    tool_context: ToolContext,
    artifact_filename: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generates a comprehensive campaign analysis report in markdown format.
    
    This tool creates a detailed report covering all campaign segments with top/bottom locations,
    strategic insights, and actionable recommendations. The report is saved as a downloadable
    markdown artifact.
    
    Args:
        artifact_filename (str, optional): Specific artifact file to analyze.
                                         If not provided, uses the most recent analysis from state.
    
    Returns:
        Dict[str, Any]: Report generation status and download information
    """
    
    try:
        # Get artifact filename from parameter or state
        if artifact_filename is None:
            artifact_filename = tool_context.state.get("temp:candidate_analysis_artifact")
            if not artifact_filename:
                return {
                    "status": "error",
                    "message": "No candidate analysis found. Please run create_candidate_analysis first."
                }
        
        # Load analysis metadata from state
        metadata = tool_context.state.get("temp:analysis_metadata", {})
        
        # Load the analysis DataFrame
        artifact = await tool_context.load_artifact(filename=artifact_filename)
        
        if not artifact or not artifact.inline_data:
            return {
                "status": "error", 
                "message": f"Could not load analysis data from '{artifact_filename}'"
            }
        
        import io
        from datetime import datetime
        
        df = pd.read_pickle(io.BytesIO(artifact.inline_data.data))
        logger.info(f"Loaded analysis data: {df.shape[0]} segments, {df.shape[1]} columns")
        
        candidate_name = metadata.get('candidate_name', 'Candidate')
        opponent_name = metadata.get('opponent_name', 'Opponent')
        location = metadata.get('location', 'Location')
        candidate_base = metadata.get('candidate_base', 'Party')
        age = metadata.get('age', 'All Ages')
        gender = metadata.get('gender', 'All Genders')
        tag_names = metadata.get('tag_names', [])
        
        # ================================
        # 1. ANALYZE ALL SEGMENTS
        # ================================
        
        # Get segment breakdowns
        rally_base_df = df[df['strategy'] == 'Rally the Base'].copy()
        hidden_goldmine_df = df[df['strategy'] == 'Hidden Goldmine'].copy()
        bring_them_over_df = df[df['strategy'] == 'Bring Them Over'].copy()
        deep_conversion_df = df[df['strategy'] == 'Deep Conversion'].copy()
        
        total_segments = len(df)
        
        # ================================
        # 2. BUILD MARKDOWN REPORT
        # ================================
        
        report_date = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        
        markdown_report = f"""# Campaign Analysis Report
## {candidate_name} vs {opponent_name}

**Generated:** {report_date}  
**Location:** {location}  
**Party/Base:** {candidate_base}  
**Demographics:** {age}, {gender}  
**Issues Analyzed:** {', '.join(tag_names) if tag_names else 'None'}  

---

## Executive Summary

This comprehensive analysis examined **{total_segments:,} geographic segments** in {location} to identify strategic opportunities for {candidate_name}'s campaign against {opponent_name}.

### Key Findings:
- **Rally the Base:** {len(rally_base_df):,} segments ({len(rally_base_df)/total_segments*100:.1f}%) - High affinity, high popularity
- **Hidden Goldmine:** {len(hidden_goldmine_df):,} segments ({len(hidden_goldmine_df)/total_segments*100:.1f}%) - High affinity, low popularity  
- **Bring Them Over:** {len(bring_them_over_df):,} segments ({len(bring_them_over_df)/total_segments*100:.1f}%) - Low affinity, high popularity
- **Deep Conversion:** {len(deep_conversion_df):,} segments ({len(deep_conversion_df)/total_segments*100:.1f}%) - Low affinity, low popularity

---

"""

        # ================================
        # 3. RALLY THE BASE SECTION
        # ================================
        
        if len(rally_base_df) > 0:
            # Calculate stats
            avg_affinity = rally_base_df['affinity'].mean()
            avg_popularity = rally_base_df['popularity'].mean()
            
            # Top performing areas
            top_rally = rally_base_df.nlargest(5, 'affinity')
            bottom_rally = rally_base_df.nsmallest(5, 'affinity')
            
            # Opponent analysis within Rally Base
            trailing_opponent = rally_base_df[rally_base_df['popularity_status'] == 'Trailing Opponent'] if 'popularity_status' in rally_base_df.columns else pd.DataFrame()
            
            # Base introduction needs
            base_intro_needed = rally_base_df[rally_base_df['base_popularity_status'] == 'Less Popular than Party']
            
            # Issue weaknesses
            issue_weaknesses = {}
            for col in rally_base_df.columns:
                if col.startswith('issue_') and col.endswith('_popularity_status'):
                    issue_name = col.replace('issue_', '').replace('_popularity_status', '')
                    weak_count = len(rally_base_df[rally_base_df[col] == 'Candidate Less Popular with this Issue'])
                    if weak_count > 0:
                        issue_weaknesses[issue_name] = weak_count
            
            markdown_report += f"""## 🎯 Rally the Base Segment

**Overview:** Your core support base with high affinity and popularity. Focus on mobilization and addressing specific concerns.

### Statistics:
- **Total Segments:** {len(rally_base_df):,} ({len(rally_base_df)/total_segments*100:.1f}% of total)
- **Average Affinity:** {avg_affinity:.2f}/1.0
- **Average Popularity:** {avg_popularity:.2f}/1.0
- **Trailing Opponent:** {len(trailing_opponent):,} locations
- **Need Base Introduction:** {len(base_intro_needed):,} locations

### 🏆 Top 5 Strongest Rally Base Locations:
"""
            
            for i, row in top_rally.iterrows():
                markdown_report += f"- **Location {i+1}:** {row['geohash']} (Affinity: {row['affinity']:.2f}, Popularity: {row['popularity']:.2f})\n"
            
            markdown_report += f"""
### ⚠️ Bottom 5 Rally Base Locations (Need Attention):
"""
            
            for i, row in bottom_rally.iterrows():
                markdown_report += f"- **Location {i+1}:** {row['geohash']} (Affinity: {row['affinity']:.2f}, Popularity: {row['popularity']:.2f})\n"
            
            if issue_weaknesses:
                markdown_report += f"""
### 📋 Issue Weaknesses in Rally Base:
"""
                for issue, count in issue_weaknesses.items():
                    percentage = (count / len(rally_base_df)) * 100
                    markdown_report += f"- **{issue.replace('_', ' ').title()}:** {count:,} locations ({percentage:.1f}%)\n"
            
            markdown_report += f"""
### 🎯 Strategic Actions for Rally Base:

#### Immediate Actions (Next 2 weeks):
1. **Voter Mobilization:** Target top 5 strongest locations for voter registration drives
2. **Address Opponent Threat:** Focus on {len(trailing_opponent):,} locations where trailing opponent
3. **Base Introduction:** Introduce candidate in {len(base_intro_needed):,} areas with low party alignment

#### Medium-term Actions (2-4 weeks):
"""
            
            if len(trailing_opponent) > 0:
                markdown_report += f"1. **Swing Voter Outreach:** Develop messaging for {len(trailing_opponent):,} Rally Base locations where opponent is ahead\n"
            
            if len(base_intro_needed) > 0:
                markdown_report += f"2. **Party Alignment Campaign:** Schedule events with party leaders in {len(base_intro_needed):,} underperforming areas\n"
            
            if issue_weaknesses:
                top_issue = max(issue_weaknesses.items(), key=lambda x: x[1])
                markdown_report += f"3. **Issue Focus:** Address {top_issue[0].replace('_', ' ')} concerns in {top_issue[1]:,} locations\n"
            
            markdown_report += f"""
#### Success Metrics:
- Achieve 85%+ voter turnout in top Rally Base locations
- Reduce opponent lead in trailing areas by 50%
- Improve candidate-to-party ratio in base introduction areas

---

"""
        
        # ================================
        # 4. HIDDEN GOLDMINE SECTION
        # ================================
        
        if len(hidden_goldmine_df) > 0:
            avg_affinity_hgm = hidden_goldmine_df['affinity'].mean()
            avg_popularity_hgm = hidden_goldmine_df['popularity'].mean()
            
            top_hgm = hidden_goldmine_df.nlargest(5, 'affinity')
            bottom_hgm = hidden_goldmine_df.nsmallest(5, 'affinity')
            
            # High priority areas (high affinity, very low popularity)
            high_priority_hgm = hidden_goldmine_df[hidden_goldmine_df['affinity'] > 0.7]
            
            markdown_report += f"""## 💎 Hidden Goldmine Segment

**Overview:** High affinity but low popularity - prime candidates need introduction. These supporters just need to know the candidate better.

### Statistics:
- **Total Segments:** {len(hidden_goldmine_df):,} ({len(hidden_goldmine_df)/total_segments*100:.1f}% of total)
- **Average Affinity:** {avg_affinity_hgm:.2f}/1.0
- **Average Popularity:** {avg_popularity_hgm:.2f}/1.0
- **High Priority (>0.7 affinity):** {len(high_priority_hgm):,} locations

### 💰 Top 5 Highest Potential Locations:
"""
            
            for i, row in top_hgm.iterrows():
                markdown_report += f"- **Location {i+1}:** {row['geohash']} (Affinity: {row['affinity']:.2f}, Popularity: {row['popularity']:.2f})\n"
            
            markdown_report += f"""
### 🔍 Lowest Priority Hidden Goldmine:
"""
            
            for i, row in bottom_hgm.iterrows():
                markdown_report += f"- **Location {i+1}:** {row['geohash']} (Affinity: {row['affinity']:.2f}, Popularity: {row['popularity']:.2f})\n"
            
            markdown_report += f"""
### 🚀 Strategic Actions for Hidden Goldmine:

#### Immediate Actions (Next 2 weeks):
1. **Candidate Introduction Campaign:** Focus on top {min(len(high_priority_hgm), 10):,} highest affinity locations
2. **Meet & Greet Events:** Schedule intimate events in top 5 locations
3. **Digital Introduction:** Launch targeted social media campaigns showcasing candidate personality

#### Medium-term Actions (2-6 weeks):
1. **Grassroots Outreach:** Deploy volunteers for door-to-door introductions
2. **Local Media Blitz:** Arrange local interviews and appearances
3. **Community Engagement:** Participate in local events and town halls
4. **Testimonial Campaign:** Collect and share supporter testimonials

#### Success Metrics:
- Increase candidate recognition by 40% in Hidden Goldmine areas
- Convert 60% of high-affinity locations to Rally Base status
- Achieve 70% voter turnout in converted areas

---

"""
        
        # Add sections for Bring Them Over and Deep Conversion (abbreviated for space)
        if len(bring_them_over_df) > 0:
            markdown_report += f"""## 🔄 Bring Them Over Segment

**Overview:** Low affinity but high popularity - focus on converting soft support.

### Statistics:
- **Total Segments:** {len(bring_them_over_df):,} ({len(bring_them_over_df)/total_segments*100:.1f}% of total)
- **Strategy:** Appeal-based messaging and crossover positioning

---

"""
        
        if len(deep_conversion_df) > 0:
            markdown_report += f"""## 🎖️ Deep Conversion Segment

**Overview:** Low affinity and low popularity - long-term relationship building.

### Statistics:
- **Total Segments:** {len(deep_conversion_df):,} ({len(deep_conversion_df)/total_segments*100:.1f}% of total)
- **Strategy:** Community service and authentic presence

---

"""
        
        # ================================
        # OVERALL RECOMMENDATIONS
        # ================================
        
        markdown_report += f"""## 📊 Campaign Resource Allocation

### Recommended Budget Distribution:
- **Rally the Base:** 40% of resources - {len(rally_base_df):,} segments
- **Hidden Goldmine:** 35% of resources - {len(hidden_goldmine_df):,} segments  
- **Bring Them Over:** 20% of resources - {len(bring_them_over_df):,} segments
- **Deep Conversion:** 5% of resources - {len(deep_conversion_df):,} segments

### Priority Timeline:

#### Week 1-2: Foundation
1. Launch voter mobilization in top Rally Base locations
2. Begin candidate introduction in highest-affinity Hidden Goldmine areas
3. Identify cross-party endorsement opportunities

#### Week 3-4: Expansion  
1. Address issue weaknesses in Rally Base
2. Scale Hidden Goldmine introduction campaigns
3. Launch appeal-based messaging in Bring Them Over areas

### Success Indicators:
- **Rally Base:** 85%+ turnout in top locations
- **Hidden Goldmine:** 40%+ recognition increase
- **Bring Them Over:** 30%+ conversion rate
- **Deep Conversion:** Positive sentiment shift

---

*Report generated by Campaign Analysis Platform*  
*Data includes {total_segments:,} geographic segments in {location}*
"""

        # ================================
        # SAVE REPORT AS ARTIFACT
        # ================================
        
        # Create markdown artifact
        report_bytes = markdown_report.encode('utf-8')
        
        artifact_part = types.Part(
            inline_data=types.Blob(
                mime_type="text/markdown",
                data=report_bytes
            )
        )
        
        # Create filename
        safe_candidate = candidate_name.lower().replace(' ', '_').replace('.', '')
        safe_location = location.lower().replace(' ', '_').replace('.', '')
        report_filename = f"campaign_report_{safe_candidate}_{safe_location}.md"
        
        # Save artifact
        version = await tool_context.save_artifact(filename=report_filename, artifact=artifact_part)
        
        logger.info(f"Successfully saved campaign report as artifact '{report_filename}' version {version}")
        
        # Store reference in state
        tool_context.state["temp:campaign_report_artifact"] = report_filename
        
        return {
            "status": "success",
            "message": f"📊 Comprehensive campaign report generated for {candidate_name} vs {opponent_name}",
            "report_filename": report_filename,
            "report_version": version,
            "report_summary": {
                "total_pages": "Multi-page comprehensive analysis",
                "segments_analyzed": 4,
                "total_locations": total_segments,
                "rally_base_locations": len(rally_base_df),
                "hidden_goldmine_locations": len(hidden_goldmine_df),
                "bring_them_over_locations": len(bring_them_over_df),
                "deep_conversion_locations": len(deep_conversion_df)
            },
            "download_info": {
                "file_type": "Markdown (.md)",
                "file_size_kb": round(len(report_bytes) / 1024, 1),
                "contains": [
                    "📋 Executive Summary",
                    "🎯 Detailed Segment Analysis", 
                    "📍 Top/Bottom Locations per Segment",
                    "🚀 Strategic Action Plans",
                    "💰 Resource Allocation Guidance",
                    "📈 Timeline and KPIs"
                ]
            },
            "next_steps": [
                f"📥 Download the report from artifacts: '{report_filename}'",
                "📊 Review strategic recommendations for each segment",
                "⚡ Implement immediate actions for Rally Base and Hidden Goldmine",
                "📅 Schedule follow-up analysis in 2-4 weeks to track progress"
            ]
        }
        
    except Exception as e:
        logger.error(f"Error generating campaign report: {e}")
        return {
            "status": "error",
            "message": f"Failed to generate campaign report: {str(e)}",
            "error_details": str(e)
        }

async def filter_campaign_locations(
    tool_context: ToolContext,
    filter_criteria: Dict[str, Any],
    location_tag: str,
    artifact_filename: Optional[str] = None
) -> Dict[str, Any]:
    """
    Filter campaign locations and save lat/long pairs with consistent naming.
    Shows step-by-step filtering progress in logs.
    
    Args:
        filter_criteria (Dict[str, Any]): Filter criteria with column names and values
        location_tag (str): Simple descriptive tag for this group of locations
        artifact_filename (str, optional): Analysis artifact to use
        
    Returns:
        Dict with filtered locations saved to tool context under "identified_locations"
    """
    
    try:
        # Get artifact filename
        if artifact_filename is None:
            artifact_filename = tool_context.state.get("temp:candidate_analysis_artifact")
            if not artifact_filename:
                return {
                    "status": "error",
                    "message": "No candidate analysis found. Please run create_candidate_analysis first."
                }
        
        # Load the analysis DataFrame
        artifact = await tool_context.load_artifact(filename=artifact_filename)
        
        if not artifact or not artifact.inline_data:
            return {
                "status": "error", 
                "message": f"Could not load analysis data from '{artifact_filename}'"
            }
        
        import io
        df = pd.read_pickle(io.BytesIO(artifact.inline_data.data))
        logger.info(f"Loaded analysis data: {df.shape[0]} segments, {df.shape[1]} columns")
        
        # Start with full dataframe
        filtered_df = df.copy()
        applied_filters = []
        
        logger.info(f"🔍 Starting with {len(filtered_df)} locations")
        logger.info(f"🔍 Filter criteria: {filter_criteria}")
        
        # Apply filters based on criteria
        for column, values in filter_criteria.items():
            step_start_count = len(filtered_df)
            
            if column in ["min_affinity", "min_popularity"]:
                # Handle numeric thresholds
                if column == "min_affinity" and "affinity" in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df['affinity'] >= values]
                    applied_filters.append(f"affinity >= {values}")
                    logger.info(f"🔍 Applied {column} >= {values}: {step_start_count} → {len(filtered_df)} locations")
                elif column == "min_popularity" and "popularity" in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df['popularity'] >= values]
                    applied_filters.append(f"popularity >= {values}")
                    logger.info(f"🔍 Applied {column} >= {values}: {step_start_count} → {len(filtered_df)} locations")
            else:
                # Handle categorical filters
                if column in filtered_df.columns:
                    if isinstance(values, str):
                        values = [values]  # Convert single value to list
                    
                    filtered_df = filtered_df[filtered_df[column].isin(values)]
                    applied_filters.append(f"{column}: {values}")
                    
                    logger.info(f"🔍 Applied {column} filter: {step_start_count} → {len(filtered_df)} locations")
                else:
                    logger.warning(f"🔍 Column '{column}' not found in data")
        
        logger.info(f"🔍 Final result: {len(filtered_df)} locations after all filters")
        
        if filtered_df.empty:
            return {
                "status": "warning",
                "message": "No locations found matching the filter criteria",
                "filters_applied": applied_filters,
                "available_columns": list(df.columns)
            }
        
        # Extract lat/long pairs
        coordinates = []
        for idx, row in filtered_df.iterrows():
            coordinates.append([float(row['latitude']), float(row['longitude'])])
        
        # Create simple location group data
        identified_locations = {
            "tag": location_tag,
            "description": f"Filtered locations: {', '.join(applied_filters)}",
            "coordinates": coordinates,
            "total_locations": len(coordinates),
            "filters_applied": applied_filters,
            "created_timestamp": pd.Timestamp.now().isoformat()
        }
        
        # Save to the standard key
        tool_context.state["identified_locations"] = identified_locations
        
        # Keep a simple log of location group names for reference
        location_history = tool_context.state.get("location_history", [])
        location_history.append({
            "tag": location_tag,
            "total_locations": len(coordinates),
            "created": identified_locations["created_timestamp"]
        })
        tool_context.state["location_history"] = location_history[-10:]  # Keep last 10 only
        
        logger.info(f"✅ Saved {len(coordinates)} locations with tag '{location_tag}' to 'identified_locations'")
        
        return {
            "status": "success",
            "message": f"Successfully identified and saved {len(coordinates)} locations with tag '{location_tag}'",
            "location_group": {
                "tag": location_tag,
                "total_locations": len(coordinates),
                "description": identified_locations["description"]
            },
            "filters_applied": applied_filters,
            "coordinates_sample": coordinates[:3] if len(coordinates) > 3 else coordinates,
            "saved_as": "identified_locations",
            "ready_for_content_creation": True
        }
        
    except Exception as e:
        logger.error(f"Error filtering campaign locations: {e}")
        return {
            "status": "error",
            "message": f"Failed to filter locations: {str(e)}",
            "error_details": str(e)
        }


async def get_identified_locations(
    tool_context: ToolContext
) -> Dict[str, Any]:
    """
    Retrieve the currently identified locations (always stored under 'identified_locations').
    
    Returns:
        Dict with location group data including coordinates array
    """
    
    try:
        identified_locations = tool_context.state.get("identified_locations")
        
        if not identified_locations:
            return {
                "status": "error",
                "message": "No identified locations found. Please run filter_campaign_locations first.",
                "identified_locations": None
            }
        
        logger.info(f"📍 Retrieved {identified_locations['total_locations']} locations tagged as '{identified_locations['tag']}'")
        
        return {
            "status": "success",
            "message": f"Retrieved identified locations: '{identified_locations['tag']}'",
            "identified_locations": identified_locations,
            "coordinates": identified_locations["coordinates"],
            "total_locations": identified_locations["total_locations"],
            "tag": identified_locations["tag"],
            "description": identified_locations["description"]
        }
        
    except Exception as e:
        logger.error(f"Error retrieving identified locations: {e}")
        return {
            "status": "error",
            "message": f"Failed to retrieve identified locations: {str(e)}"
        }


async def list_location_history(
    tool_context: ToolContext
) -> Dict[str, Any]:
    """
    List the history of location groups that have been identified.
    
    Returns:
        Dict with history of location groups
    """
    
    try:
        location_history = tool_context.state.get("location_history", [])
        
        if not location_history:
            return {
                "status": "info",
                "message": "No location history found",
                "location_history": []
            }
        
        logger.info(f"📋 Found {len(location_history)} recent location groups in history")
        
        return {
            "status": "success",
            "message": f"Found {len(location_history)} recent location groups",
            "location_history": location_history,
            "current_identified": tool_context.state.get("identified_locations", {}).get("tag", "None")
        }
        
    except Exception as e:
        logger.error(f"Error listing location history: {e}")
        return {
            "status": "error",
            "message": f"Failed to list location history: {str(e)}"
        }


async def check_analysis_data_structure(
    tool_context: ToolContext,
    artifact_filename: Optional[str] = None
) -> Dict[str, Any]:
    """
    Check the structure of the analysis data to understand available columns and values.
    """
    
    try:
        # Get artifact filename
        if artifact_filename is None:
            artifact_filename = tool_context.state.get("temp:candidate_analysis_artifact")
            if not artifact_filename:
                return {
                    "status": "error",
                    "message": "No candidate analysis found."
                }
        
        # Load the analysis DataFrame
        artifact = await tool_context.load_artifact(filename=artifact_filename)
        
        if not artifact or not artifact.inline_data:
            return {
                "status": "error", 
                "message": f"Could not load analysis data from '{artifact_filename}'"
            }
        
        import io
        df = pd.read_pickle(io.BytesIO(artifact.inline_data.data))
        # CRITICAL FIX: Clean DataFrame before any dictionary conversion
        df = df.fillna(value=None)  # Replace all NaN with None
        
        logger.info(f"📊 Data structure: {len(df)} rows, {len(df.columns)} columns")
        
        # Analyze data structure
        data_info = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "columns": list(df.columns),
            "column_value_counts": {}
        }
        
        # Get value counts for key columns
        key_columns = ['strategy', 'popularity_status', 'base_popularity_status']
        for col in key_columns:
            if col in df.columns:
                data_info["column_value_counts"][col] = df[col].value_counts().to_dict()
                logger.info(f"📊 {col}: {df[col].value_counts().to_dict()}")
        
        # Show issue columns if they exist
        issue_columns = [col for col in df.columns if col.startswith('issue_') and col.endswith('_popularity_status')]
        for col in issue_columns:
            data_info["column_value_counts"][col] = df[col].value_counts().to_dict()
            logger.info(f"📊 {col}: {df[col].value_counts().to_dict()}")
        
        # Get sample data
        data_info["sample_rows"] = df.head(3).to_dict('records')
        
        return {
            "status": "success",
            "message": f"Analysis data structure checked - {len(df)} rows, {len(df.columns)} columns",
            "data_info": data_info
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to check data structure: {str(e)}"
        }