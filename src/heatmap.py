import requests
import json
from typing import Dict, List, Optional, Any,Union
from dataclasses import dataclass
from pprint import pprint
import logging
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv

# Copy the QlooAPIClient class from the provided implementation
from .qloo import QlooAPIClient, QlooSignals, QlooAudience
#from .secret_manager import SecretManager,SecretConfig


load_dotenv()

project_id = os.getenv('PROJECT_ID', 'energyagentai')  # Default fallback
location = os.getenv('LOCATION', 'us-central1')        # Default fallback
qloo_api_key = os.getenv('QLOO_API_KEY')
client = QlooAPIClient(api_key=qloo_api_key)
logger = logging.getLogger(__name__)



def get_complete_heatmap_analysis(
    location_query: str,
    entity_names: Optional[Union[str, List[str]]] = None,
    tag_names: Optional[Union[str, List[str]]] = None,
    audience_ids: Optional[List[str]] = None,  # NEW: Add audience_ids parameter
    audience_weight: Optional[float] = None,   # NEW: Add audience_weight parameter
    age: Optional[str] = None,
    gender: Optional[str] = None,
    boundary: Optional[str] = None,
    bias_trends: Optional[str] = None,
    limit: int = 50
) -> Dict[str, Any]:
    """
    Core heatmap function - resolves names to IDs and returns raw JSON data
    Uses global client variable
    
    Args:
        location_query: Location to analyze (e.g., "New York", "Toronto", "London")
        entity_names: Optional Entity name(s) for context - can be string or list
        tag_names: Optional Tag name(s) for context - can be string or list
        audience_ids: Optional list of audience IDs for context  # NEW
        audience_weight: Optional weight for audience influence (0-1)  # NEW
        age: Optional Age demographic ("35_and_younger", "36_to_55", "55_and_older")
        gender: Optional Gender demographic ("male", "female")
        boundary: Optional Heatmap boundary type ("geohashes", "city", "neighborhood")
        bias_trends: Optional bias trends parameter
        limit: Max data points to retrieve (1-50)
        
    Returns:
        DataFrame with raw heatmap data points and resolution info
    """
    
    # Convert single strings to lists for consistency
    if isinstance(entity_names, str):
        entity_names = [entity_names]
    if isinstance(tag_names, str):
        tag_names = [tag_names]
    
    result = {
        "success": False,
        "location_query": location_query,
        "input_parameters": {
            "entity_names": entity_names or [],
            "tag_names": tag_names or [],
            "audience_ids": audience_ids or [],  # NEW
            "audience_weight": audience_weight,  # NEW
            "age": age,
            "gender": gender,
            "boundary": boundary,
            "bias_trends": bias_trends,
            "limit": limit
        },
        "resolution": {
            "entities": {},
            "tags": {},
            "entity_ids": [],
            "tag_ids": []
        },
        "data_points": [],
        "total_points": 0,
        "errors": []
    }
    
    try:
        # Step 1: Resolve entity names to IDs (existing code remains the same)
        logger.info(f"🔍 Resolving entities for heatmap in {location_query}")
        
        if entity_names:
            for entity_name in entity_names:
                try:
                    search_result = client.search_entities(
                        query=entity_name,
                        limit=2,
                        sort_by="match"
                    )
                    
                    if search_result.get("success") and search_result.get("entities"):
                        entity = search_result["entities"][0]
                        result["resolution"]["entity_ids"].append(entity.id)
                        result["resolution"]["entities"][entity_name] = {
                            "entity_id": entity.id,
                            "matched_name": entity.name,
                            "entity_type": entity.entity_type,
                            "affinity_score": entity.affinity_score,
                            "success": True
                        }
                        logger.info(f"  ✅ {entity_name} → {entity.name} ({entity.id})")
                    else:
                        result["resolution"]["entities"][entity_name] = {
                            "entity_id": None,
                            "error": "Entity not found",
                            "success": False
                        }
                        result["errors"].append(f"Could not resolve entity: {entity_name}")
                        logger.warning(f"  ❌ {entity_name} → Not found")
                        
                except Exception as e:
                    result["resolution"]["entities"][entity_name] = {
                        "entity_id": None,
                        "error": str(e),
                        "success": False
                    }
                    result["errors"].append(f"Error resolving entity '{entity_name}': {str(e)}")
                    logger.error(f"  ❌ {entity_name} → Error: {e}")
        
        # Step 2: Resolve tag names to IDs (existing code remains the same)
        logger.info(f"🏷️ Resolving tags for heatmap in {location_query}")
        
        if tag_names:
            for tag_name in tag_names:
                try:
                    tag_search_result = client.search_tags(
                        query=tag_name,
                        limit=5
                    )
                    
                    if tag_search_result.get("success") and tag_search_result.get("tags"):
                        found_tags = []
                        for i, tag in enumerate(tag_search_result["tags"]):
                            tag_id = tag.get("id")
                            if tag_id:
                                result["resolution"]["tag_ids"].append(tag_id)
                                
                                tag_key = f"{tag_name}_{i}" if i > 0 else tag_name
                                result["resolution"]["tags"][tag_key] = {
                                    "tag_id": tag_id,
                                    "matched_name": tag.get("name"),
                                    "subtype": tag.get("subtype"),
                                    "types": tag.get("types", []),
                                    "success": True,
                                    "search_term": tag_name,
                                    "match_rank": i + 1
                                }
                                found_tags.append(tag.get("name"))
                                logger.info(f"  ✅ {tag_name} → {tag.get('name')} ({tag_id})")
                        
                        if not found_tags:
                            result["resolution"]["tags"][tag_name] = {
                                "tag_id": None,
                                "error": "No valid tag ID found in results",
                                "success": False
                            }
                            result["errors"].append(f"Tag found but no valid ID: {tag_name}")
                        else:
                            logger.info(f"  📋 Found {len(found_tags)} tags for '{tag_name}': {', '.join(found_tags)}")
                    else:
                        result["resolution"]["tags"][tag_name] = {
                            "tag_id": None,
                            "error": tag_search_result.get("error", "Tag not found"),
                            "success": False
                        }
                        result["errors"].append(f"Could not resolve tag: {tag_name}")
                        logger.warning(f"  ❌ {tag_name} → Not found")
                        
                except Exception as e:
                    result["resolution"]["tags"][tag_name] = {
                        "tag_id": None,
                        "error": str(e),
                        "success": False
                    }
                    result["errors"].append(f"Error resolving tag '{tag_name}': {str(e)}")
                    logger.error(f"  ❌ {tag_name} → Error: {e}")
        
        # Step 3: Prepare signals with demographics AND audiences
        logger.info(f"🗺️ Getting heatmap data for {location_query}")
        
        entity_ids = result["resolution"]["entity_ids"] if result["resolution"]["entity_ids"] else None
        tag_ids = result["resolution"]["tag_ids"] if result["resolution"]["tag_ids"] else None
        
        # Create signals object if demographics OR audiences are provided
        signals = None
        if age or gender or audience_ids:  # MODIFIED: Include audience_ids check
            demographics = {}
            if age:
                demographics["age"] = age
            if gender:
                demographics["gender"] = gender
                
            signals = QlooSignals(
                demographics=demographics if demographics else None,
                location={"query": location_query},
                entity_ids=entity_ids,
                tag_ids=tag_ids,
                audience_ids=audience_ids,          # NEW: Pass audience_ids
                audience_weight=audience_weight     # NEW: Pass audience_weight
            )
        
        # Step 4: Call heatmap API once with all parameters
        try:
            heatmap_result = client.get_heatmap_analysis(
                location_query=location_query,
                entity_ids=entity_ids,
                tag_ids=tag_ids,
                signals=signals,  # Now includes audience signals
                bias_trends=bias_trends,
                boundary=boundary,
                limit=limit
            )
            
            logger.info(f"Heatmap API call result: {heatmap_result.get('success', False)}")
            
        except Exception as heatmap_error:
            logger.error(f"Heatmap API call failed: {heatmap_error}")
            result["errors"].append(f"Heatmap API timeout/error: {str(heatmap_error)}")
            result["message"] = f"Heatmap API failed: {str(heatmap_error)}"
            return result
        
        # Step 5: Process heatmap results (existing code remains the same)
        if heatmap_result.get("success"):
            heatmap_data = heatmap_result.get("results", {}).get("heatmap", [])
            
            data_points = []
            for point in heatmap_data:
                location_data = point.get("location", {})
                query_data = point.get("query", {})
                
                data_points.append({
                    "latitude": location_data.get("latitude"),
                    "longitude": location_data.get("longitude"),
                    "geohash": location_data.get("geohash"),
                    "affinity": query_data.get("affinity", 0),
                    "popularity": query_data.get("popularity", 0),
                    "affinity_rank": query_data.get("affinity_rank", 0),
                    "hotspot_score": (query_data.get("affinity", 0) * 0.6) + (query_data.get("popularity", 0) * 0.4)
                })
            
            result["data_points"] = data_points
            result["total_points"] = len(data_points)
            result["success"] = True
            result["message"] = f"Retrieved {len(data_points)} heatmap data points for {location_query}"
            result["raw_heatmap_response"] = heatmap_result
            
            logger.info(f"  ✅ Retrieved {len(data_points)} heatmap data points")

            df = pd.DataFrame(result['data_points'])
            df['affinity']=pd.to_numeric(df['affinity'],errors='coerce')
            df['popularity']=pd.to_numeric(df['popularity'],errors='coerce')

            # Add segment analysis (existing code)
            conditions = [
                (df['affinity'] >= 0.6) & (df['popularity'] >= 0.6),
                (df['affinity'] >= 0.6) & (df['popularity'] < 0.6),
                (df['affinity'] < 0.6) & (df['popularity'] >= 0.6),
                (df['affinity'] < 0.6) & (df['popularity'] < 0.6)
            ]
            choices = ['HA-HP', 'HA-LP', 'LA-HP', 'LA-LP']
            df['segment'] = np.select(conditions, choices, default='Unknown')

            strategy_map = {
                'HA-HP': 'Rally the Base',
                'HA-LP': 'Hidden Goldmine',
                'LA-HP': 'Bring Them Over',
                'LA-LP': 'Deep Conversion'
            }

            df['strategy'] = df['segment'].map(strategy_map)
            
        else:
            error_msg = heatmap_result.get("error", "Unknown error")
            result["errors"].append(f"Heatmap API call failed: {error_msg}")
            result["message"] = f"Heatmap data retrieval failed: {error_msg}"
            logger.error(f"Heatmap API failed: {error_msg}")
            
    except Exception as e:
        logger.error(f"Complete heatmap analysis failed: {e}")
        result["errors"].append(f"Function error: {str(e)}")
        result["message"] = f"Analysis failed: {str(e)}"
    
    return df

def get_heatmap_analysis_summary(
    location_query: str,
    entity_names: Optional[Union[str, List[str]]] = None,
    tag_names: Optional[Union[str, List[str]]] = None,
    age: Optional[str] = None,
    gender: Optional[str] = None,
    boundary: Optional[str] = None,
    limit: int = 50
) -> Dict[str, Any]:
    """
    Get heatmap analysis with summary, demographics, and strategic insights
    
    Args:
        location_query: Location query
        entity_names: Optional entity names for context
        tag_names: Optional tag names for context
        age: Optional age demographic
        gender: Optional gender demographic
        boundary: Optional heatmap boundary type
        limit: Max data points
        
    Returns:
        Dict with comprehensive heatmap analysis and summary
    """
    try:
        # Get raw heatmap data first using core function
        heatmap_data_result = get_complete_heatmap_analysis(
            location_query=location_query,
            entity_names=entity_names,
            tag_names=tag_names,
            age=age,
            gender=gender,
            boundary=boundary,
            limit=limit
        )
        
        if not heatmap_data_result.get("success"):
            return {
                "success": False,
                "location_query": location_query,
                "error": heatmap_data_result.get("message", "Failed to get heatmap data"),
                "analysis": {}
            }
        
        data_points = heatmap_data_result["data_points"]
        
        if not data_points:
            return {
                "success": False,
                "location_query": location_query,
                "error": "No heatmap data points found",
                "analysis": {}
            }
        
        # Calculate statistics
        affinities = [point["affinity"] for point in data_points if point["affinity"]]
        popularities = [point["popularity"] for point in data_points if point["popularity"]]
        hotspot_scores = [point["hotspot_score"] for point in data_points if point["hotspot_score"]]
        
        # Categorize hotspots
        premium_hotspots = [p for p in data_points if p["affinity"] > 0.8 and p["popularity"] > 0.8]
        good_hotspots = [p for p in data_points if p["hotspot_score"] > 0.6 and p not in premium_hotspots]
        moderate_hotspots = [p for p in data_points if 0.4 <= p["hotspot_score"] <= 0.6]
        worst_hotspots = [p for p in data_points if p["hotspot_score"] < 0.4]
        
        # Sort by hotspot score
        premium_hotspots.sort(key=lambda x: x["hotspot_score"], reverse=True)
        good_hotspots.sort(key=lambda x: x["hotspot_score"], reverse=True)
        moderate_hotspots.sort(key=lambda x: x["hotspot_score"], reverse=True)
        worst_hotspots.sort(key=lambda x: x["hotspot_score"], reverse=False)  # Worst first
        
        # Generate analysis
        analysis = {
            "location_analyzed": location_query,
            "context": {
                "entity_names": entity_names or [],
                "tag_names": tag_names or [],
                "demographics": {
                    "age": age,
                    "gender": gender
                } if age or gender else {},
                "resolved_entities": heatmap_data_result.get("resolution", {}).get("entities", {}),
                "resolved_tags": heatmap_data_result.get("resolution", {}).get("tags", {})
            },
            "data_summary": {
                "total_data_points": len(data_points),
                "entities_resolved": len(heatmap_data_result.get("resolution", {}).get("entity_ids", [])),
                "tags_resolved": len(heatmap_data_result.get("resolution", {}).get("tag_ids", [])),
                "affinity_stats": {
                    "min": min(affinities) if affinities else 0,
                    "max": max(affinities) if affinities else 0,
                    "avg": sum(affinities) / len(affinities) if affinities else 0,
                    "high_affinity_count": len([a for a in affinities if a > 0.8])
                },
                "popularity_stats": {
                    "min": min(popularities) if popularities else 0,
                    "max": max(popularities) if popularities else 0,
                    "avg": sum(popularities) / len(popularities) if popularities else 0,
                    "high_popularity_count": len([p for p in popularities if p > 0.8])
                }
            },
            "hotspot_analysis": {
                "premium_hotspots": {
                    "count": len(premium_hotspots),
                    "description": "High affinity (>0.8) AND high popularity (>0.8)",
                    "top_locations": premium_hotspots[:5]  # Top 5
                },
                "good_hotspots": {
                    "count": len(good_hotspots),
                    "description": "Strong combined performance (score >0.6)",
                    "top_locations": good_hotspots[:5]  # Top 5
                },
                "moderate_hotspots": {
                    "count": len(moderate_hotspots),
                    "description": "Moderate performance (score 0.4-0.6)",
                    "top_locations": moderate_hotspots[:3]  # Top 3
                },
                "worst_hotspots": {
                    "count": len(worst_hotspots),
                    "description": "Poor performance (score <0.4) - areas to avoid or improve",
                    "worst_locations": worst_hotspots[:5]  # Worst 5
                }
            },
            "strategic_insights": {
                "market_coverage": {
                    "significant_locations": len(premium_hotspots) + len(good_hotspots),
                    "coverage_percentage": ((len(premium_hotspots) + len(good_hotspots)) / len(data_points)) * 100 if data_points else 0
                },
                "recommendations": []
            }
        }
        
        # Generate recommendations
        recommendations = []
        if premium_hotspots:
            recommendations.append(f"🎯 PRIORITY FOCUS: {len(premium_hotspots)} premium locations identified for immediate attention")
        if good_hotspots:
            recommendations.append(f"📈 SECONDARY TARGETS: {len(good_hotspots)} good locations for expansion or optimization")
        if moderate_hotspots:
            recommendations.append(f"👀 MONITORING: {len(moderate_hotspots)} moderate locations for future consideration")
        if worst_hotspots:
            recommendations.append(f"⚠️ AVOID/IMPROVE: {len(worst_hotspots)} poor performing locations that need attention or should be avoided")
        
        coverage_pct = analysis["strategic_insights"]["market_coverage"]["coverage_percentage"]
        recommendations.append(f"📊 MARKET COVERAGE: {coverage_pct:.1f}% of analyzed area shows significant potential")
        
        # Add insights about worst areas
        if worst_hotspots:
            worst_pct = (len(worst_hotspots) / len(data_points)) * 100
            recommendations.append(f"⛔ RISK AREAS: {worst_pct:.1f}% of locations show poor performance and should be carefully evaluated")
        
        analysis["strategic_insights"]["recommendations"] = recommendations
        
        return {
            "success": True,
            "location_query": location_query,
            "analysis": analysis,
            "raw_data_points": data_points,
            "resolution_info": heatmap_data_result.get("resolution", {}),
            "message": f"Analyzed {len(data_points)} locations in {location_query}"
        }
        
    except Exception as e:
        logger.error(f"Heatmap analysis summary failed: {e}")
        return {
            "success": False,
            "location_query": location_query,
            "error": str(e),
            "analysis": {}
        }

def get_heatmap_top_locations(
    location_query: str,
    entity_names: Optional[Union[str, List[str]]] = None,
    tag_names: Optional[Union[str, List[str]]] = None,
    age: Optional[str] = None,
    gender: Optional[str] = None,
    top_n: int = 10,
    min_score: float = 0.5
) -> Dict[str, Any]:
    """
    Get top N locations from heatmap analysis (simplified version)
    
    Args:
        location_query: Location query
        entity_names: Optional entity names for context
        tag_names: Optional tag names for context
        age: Optional age demographic
        gender: Optional gender demographic
        top_n: Number of top locations to return
        min_score: Minimum hotspot score threshold
        
    Returns:
        Dict with top locations and their scores
    """
    try:
        # Get raw heatmap data using core function
        heatmap_result = get_complete_heatmap_analysis(
            location_query=location_query,
            entity_names=entity_names,
            tag_names=tag_names,
            age=age,
            gender=gender,
            limit=50
        )
        
        if not heatmap_result.get("success"):
            return {
                "success": False,
                "location_query": location_query,
                "error": heatmap_result.get("message", "Failed to get heatmap data"),
                "top_locations": []
            }
        
        data_points = heatmap_result["data_points"]
        
        # Filter by minimum score and sort
        filtered_points = [p for p in data_points if p["hotspot_score"] >= min_score]
        top_locations = sorted(filtered_points, key=lambda x: x["hotspot_score"], reverse=True)[:top_n]
        
        return {
            "success": True,
            "location_query": location_query,
            "entity_context": entity_names or [],
            "tag_context": tag_names or [],
            "demographics": {
                "age": age,
                "gender": gender
            } if age or gender else {},
            "total_locations_analyzed": len(data_points),
            "locations_above_threshold": len(filtered_points),
            "min_score_threshold": min_score,
            "entities_resolved": len(heatmap_result.get("resolution", {}).get("entity_ids", [])),
            "tags_resolved": len(heatmap_result.get("resolution", {}).get("tag_ids", [])),
            "top_locations": [
                {
                    "rank": i + 1,
                    "latitude": loc["latitude"],
                    "longitude": loc["longitude"],
                    "hotspot_score": round(loc["hotspot_score"], 3),
                    "affinity": round(loc["affinity"], 3),
                    "popularity": round(loc["popularity"], 3),
                    "geohash": loc["geohash"]
                }
                for i, loc in enumerate(top_locations)
            ],
            "resolution_info": heatmap_result.get("resolution", {}),
            "message": f"Found {len(top_locations)} top locations in {location_query}"
        }
        
    except Exception as e:
        logger.error(f"Top locations extraction failed: {e}")
        return {
            "success": False,
            "location_query": location_query,
            "error": str(e),
            "top_locations": []
        }

def get_heatmap_bottom_locations(
    location_query: str,
    entity_names: Optional[Union[str, List[str]]] = None,
    tag_names: Optional[Union[str, List[str]]] = None,
    age: Optional[str] = None,
    gender: Optional[str] = None,
    bottom_n: int = 10,
    max_score: float = 0.5
) -> Dict[str, Any]:
    """
    Get bottom N (worst performing) locations from heatmap analysis
    
    Args:
        location_query: Location query
        entity_names: Optional entity names for context
        tag_names: Optional tag names for context
        age: Optional age demographic
        gender: Optional gender demographic
        bottom_n: Number of bottom locations to return
        max_score: Maximum hotspot score threshold (locations below this score)
        
    Returns:
        Dict with bottom locations and their scores
    """
    try:
        # Get raw heatmap data using core function
        heatmap_result = get_complete_heatmap_analysis(
            location_query=location_query,
            entity_names=entity_names,
            tag_names=tag_names,
            age=age,
            gender=gender,
            limit=50
        )
        
        if not heatmap_result.get("success"):
            return {
                "success": False,
                "location_query": location_query,
                "error": heatmap_result.get("message", "Failed to get heatmap data"),
                "bottom_locations": []
            }
        
        data_points = heatmap_result["data_points"]
        
        # Filter by maximum score and sort (lowest scores first)
        filtered_points = [p for p in data_points if p["hotspot_score"] <= max_score]
        bottom_locations = sorted(filtered_points, key=lambda x: x["hotspot_score"], reverse=False)[:bottom_n]
        
        return {
            "success": True,
            "location_query": location_query,
            "entity_context": entity_names or [],
            "tag_context": tag_names or [],
            "demographics": {
                "age": age,
                "gender": gender
            } if age or gender else {},
            "total_locations_analyzed": len(data_points),
            "locations_below_threshold": len(filtered_points),
            "max_score_threshold": max_score,
            "entities_resolved": len(heatmap_result.get("resolution", {}).get("entity_ids", [])),
            "tags_resolved": len(heatmap_result.get("resolution", {}).get("tag_ids", [])),
            "bottom_locations": [
                {
                    "rank": i + 1,
                    "latitude": loc["latitude"],
                    "longitude": loc["longitude"],
                    "hotspot_score": round(loc["hotspot_score"], 3),
                    "affinity": round(loc["affinity"], 3),
                    "popularity": round(loc["popularity"], 3),
                    "geohash": loc["geohash"]
                }
                for i, loc in enumerate(bottom_locations)
            ],
            "resolution_info": heatmap_result.get("resolution", {}),
            "message": f"Found {len(bottom_locations)} worst performing locations in {location_query}"
        }
        
    except Exception as e:
        logger.error(f"Bottom locations extraction failed: {e}")
        return {
            "success": False,
            "location_query": location_query,
            "error": str(e),
            "bottom_locations": []
        }
