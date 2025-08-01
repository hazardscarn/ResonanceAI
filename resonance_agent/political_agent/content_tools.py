# campaign_content_agent/content_tools.py
import logging
from typing import Dict, List, Optional, Any
from google.adk.tools import ToolContext
from google.genai import types
import json

# Import your existing functions
from .qlootools import (
    get_entity_brand_insights,
    get_entity_movie_insights,
    get_entity_artist_insights,
    get_entity_tv_show_insights,
    get_tag_insights,
    get_entity_place_insights
)

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.qloo import QlooAPIClient, QlooSignals
from src.secret_manager import SecretConfig



qloo_api_key = SecretConfig.get_qloo_api_key()
client = QlooAPIClient(api_key=qloo_api_key)
logger = logging.getLogger(__name__)


def create_polygon(coordinates):
    """
    Create WKT polygon from coordinates
    Args: coordinates - list of [longitude, latitude] pairs
    Returns: WKT polygon string if >=3 points, otherwise returns original coordinates array
    """
    if len(coordinates) < 3:
        return coordinates
        
    longs = [coord[0] for coord in coordinates]
    lats = [coord[1] for coord in coordinates]
        
    min_long, max_long = min(longs), max(longs)
    min_lat, max_lat = min(lats), max(lats)
        
    return f"POLYGON(({min_long} {min_lat}, {max_long} {min_lat}, {max_long} {max_lat}, {min_long} {max_lat}, {min_long} {min_lat}))"


def convert_age_for_insights(age: Optional[str]) -> Optional[str]:
    """Convert political analysis age to insights API format"""
    if not age:
        return None
        
    age_mapping = {
        "24_and_younger": "35_and_younger",
        "25_to_29": "35_and_younger", 
        "30_to_34": "35_and_younger",
        "35_and_younger": "35_and_younger",
        "35_to_44": "36_to_55",
        "45_to_54": "36_to_55", 
        "36_to_55": "36_to_55",
        "55_and_older": "55_and_older"
    }
    
    return age_mapping.get(age, None)


async def generate_content_insights(
    tool_context: ToolContext,
    age: Optional[str] = None,
    gender: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate content insights for identified locations using your existing functions.
    
    Args:
        age: Age demographic from political analysis
        gender: Gender filter (male/female)
    
    Returns:
        Dict with content insights and downloadable report
    """
    
    try:
        # Get identified locations
        identified_locations = tool_context.state.get("identified_locations")
        if not identified_locations:
            return {
                "status": "error", 
                "message": "No identified locations found. Run political analysis and location filtering first."
            }
        
        coordinates = identified_locations.get("coordinates", [])
        if not coordinates:
            return {
                "status": "error",
                "message": "No coordinates found in identified locations"
            }
        
        logger.info(f"🎯 Generating content insights for {len(coordinates)} locations")
        
        # Create polygon from coordinates
        # Note: coordinates are [lat, lon] but we need [lon, lat] for WKT
        lon_lat_coords = [[coord[1], coord[0]] for coord in coordinates]
        polygon_wkt = create_polygon(lon_lat_coords)
        
        if isinstance(polygon_wkt, list):
            return {
                "status": "error",
                "message": f"Need at least 3 coordinates to create polygon, got {len(coordinates)}"
            }
        
        logger.info(f"📍 Created polygon: {polygon_wkt}")
        
        # Convert age for insights API
        insights_age = convert_age_for_insights(age)
        
        # Create demographics dict
        demographics = {}
        if insights_age:
            demographics["age"] = insights_age
        if gender and gender.lower() in ["male", "female"]:
            demographics["gender"] = gender.lower()
        
        # FIXED: Create QlooSignals with proper location format
        # For WKT polygons, use direct WKT string, not wrapped in query object
        signals = QlooSignals(
            demographics=demographics if demographics else None,
            location=polygon_wkt,  # Direct WKT string for signal.location
            audience_ids=None,  # No audience signals needed
            entity_queries=None
        )
        
        logger.info(f"🔍 Using signals: demographics={demographics}, location=polygon (no audiences)")
        
        # Use empty audience_ids list since we don't want audience signals
        audience_ids = []
        
        # Use your existing functions to get insights WITHOUT audience_ids
        all_insights = {}
        
        # DEBUGGING: Let's see what's being passed to the API
        logger.info(f"🔍 Signals being used: demographics={signals.demographics}, location='{signals.location}', audience_ids={audience_ids}")
        
        # Get brand insights
        logger.info(f"🏢 Getting brand insights...")
        brand_result = get_entity_brand_insights(signals, audience_ids=audience_ids, limit=10)
        all_insights["brands"] = brand_result
        # Save to tool context state
        tool_context.state['brand_insights'] = brand_result
        
        # Get movie insights  
        logger.info(f"🎬 Getting movie insights...")
        movie_result = get_entity_movie_insights(signals, audience_ids=audience_ids, limit=5)
        all_insights["movies"] = movie_result
        # Save to tool context state
        tool_context.state['movie_insights'] = movie_result
        
        # Get TV show insights
        logger.info(f"📺 Getting TV show insights...")
        tv_result = get_entity_tv_show_insights(signals, audience_ids=audience_ids, limit=5)
        all_insights["tv_shows"] = tv_result
        # Save to tool context state
        tool_context.state['tv_show_insights'] = tv_result
        
        # Get artist insights
        logger.info(f"🎵 Getting artist insights...")
        artist_result = get_entity_artist_insights(signals, audience_ids=audience_ids, limit=5)
        all_insights["artists"] = artist_result
        # Save to tool context state
        tool_context.state['artist_insights'] = artist_result
        
        # Get tag insights
        logger.info(f"🏷️ Getting tag insights...")
        tag_result = get_tag_insights(signals, audience_ids=audience_ids, tag_filter=None, limit=15)
        all_insights["tags"] = tag_result
        # Save to tool context state
        tool_context.state['tag_insights'] = tag_result
        

        # Get Places insights
        logger.info(f"🏷️ Getting Places insights...")
        place_result = get_entity_place_insights(signals, audience_ids=audience_ids, limit=10)
        all_insights["place_insights"] = place_result
        # Save to tool context state
        tool_context.state['place_insights'] = place_result


        # Save all insights to state for later use
        tool_context.state['all_content_insights'] = all_insights
        
        # Generate report
        report_result = await _create_simple_report(
            tool_context, all_insights, identified_locations, age, gender, polygon_wkt
        )
        
        return {
            "status": "success",
            "message": f"Generated content insights for {len(coordinates)} locations using polygon area",
            "polygon_used": polygon_wkt,
            "demographics": {
                "original_age": age,
                "insights_age": insights_age,
                "gender": gender
            },
            "insights_generated": {
                "brands": "✅" if "No brand results found" not in all_insights["brands"] else "❌",
                "movies": "✅" if "No movie results found" not in all_insights["movies"] else "❌", 
                "tv_shows": "✅" if "No TV show results found" not in all_insights["tv_shows"] else "❌",
                "artists": "✅" if "No artist results found" not in all_insights["artists"] else "❌",
                "tags": "✅" if "No tag results found" not in all_insights["tags"] else "❌",
                "places": "✅" if "No place results found" not in all_insights["place_insights"] else "❌"
            },
            "report_info": report_result,
            "location_info": {
                "tag": identified_locations.get("tag"),
                "total_coordinates": len(coordinates)
            },
            "audience_ids_used": "None (location + demographics only)"
        }
        
    except Exception as e:
        logger.error(f"Error generating content insights: {e}")
        return {
            "status": "error",
            "message": f"Content insights generation failed: {str(e)}"
        }

async def _create_simple_report(
    tool_context: ToolContext,
    all_insights: Dict[str, str],
    identified_locations: Dict[str, Any],
    age: Optional[str],
    gender: Optional[str],
    polygon_wkt: str
) -> Dict[str, Any]:
    """Create simple markdown report from insights"""
    
    try:
        from datetime import datetime
        report_date = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        
        location_tag = identified_locations.get("tag", "Campaign Locations")
        
        # Build simple report using your existing formatted strings
        markdown_report = f"""# Campaign Content Strategy Report
## {location_tag}

**Generated:** {report_date}  
**Target Area:** {len(identified_locations.get('coordinates', []))} locations (polygon area)  
**Demographics:** {age or 'All Ages'}, {gender or 'All Genders'}  
**Polygon:** {polygon_wkt}

---

## Cultural Intelligence Insights

Based on demographic and geographic signals for the target campaign area.

---

## Brand Preferences

{all_insights.get("brands", "No brand insights available")}

---

## Movie Preferences  

{all_insights.get("movies", "No movie insights available")}

---

## TV Show Preferences

{all_insights.get("tv_shows", "No TV show insights available")}

---

## Music & Artist Preferences

{all_insights.get("artists", "No artist insights available")}

---

## Interest Tags & Categories

{all_insights.get("tags", "No tag insights available")}

---

## Content Strategy Recommendations

### Immediate Actions:
1. **Cultural Alignment**: Review campaign materials against identified preferences
2. **Brand Integration**: Consider partnerships with top-performing brands
3. **Entertainment References**: Use popular movies/shows for relatable messaging
4. **Music Strategy**: Incorporate preferred artists in campaign content

### Message Targeting:
- Use entertainment preferences to create relatable campaign content
- Reference popular brands for authentic cultural connections  
- Align messaging with interest categories shown in tags
- Consider music preferences for rally/event soundtracks

---

*Generated by Campaign Content Intelligence Platform using Qloo Cultural Data*
"""
        
        # Save as artifact
        report_bytes = markdown_report.encode('utf-8')
        artifact_part = types.Part(
            inline_data=types.Blob(
                mime_type="text/markdown",
                data=report_bytes
            )
        )
        
        safe_tag = location_tag.lower().replace(' ', '_').replace('.', '')
        report_filename = f"content_strategy_{safe_tag}.md"
        
        version = await tool_context.save_artifact(filename=report_filename, artifact=artifact_part)
        
        logger.info(f"✅ Saved content report: {report_filename}")
        
        return {
            "report_filename": report_filename,
            "report_version": version,
            "report_size_kb": round(len(report_bytes) / 1024, 1)
        }
        
    except Exception as e:
        logger.error(f"Error creating report: {e}")
        return {
            "error": f"Failed to create report: {str(e)}"
        }


async def get_current_locations(
    tool_context: ToolContext
) -> Dict[str, Any]:
    """Get the identified locations saved by the main political analysis agent."""
    
    try:
        identified_locations = tool_context.state.get("identified_locations")
        
        if not identified_locations:
            return {
                "status": "error",
                "message": "No identified locations found. Run political analysis and location filtering first."
            }
        
        coordinates = identified_locations.get("coordinates", [])
        
        return {
            "status": "success",
            "message": f"Retrieved {len(coordinates)} identified locations",
            "coordinates": coordinates,
            "total_locations": len(coordinates),
            "location_tag": identified_locations.get("tag", "Unknown"),
            "description": identified_locations.get("description", "")
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to retrieve locations: {str(e)}"
        }