import requests
import json
import sys
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from dotenv import load_dotenv
import os
import logging
from google.adk.tools import ToolContext
from datetime import datetime

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.qloo import QlooAPIClient, QlooSignals, QlooAudience
from .config import SecretConfig,Modelconfig
from vertexai.generative_models import GenerativeModel,GenerationConfig
from vertexai.generative_models import (
    GenerativeModel,
    GenerationConfig,
    SafetySetting,
    HarmCategory,
    HarmBlockThreshold,
)


# Initialize Qloo API Client with Secret Manager
qloo_api_key = SecretConfig.get_qloo_api_key()
client = QlooAPIClient(api_key=qloo_api_key)

project_id = SecretConfig.get_google_cloud_project()
location = SecretConfig.get_google_cloud_location()
logging.info(f"project id {project_id}")
logging.info(f"location id {location}")
step_logger = logging.getLogger("AGENT_STEPS")
logger = logging.getLogger(__name__)

def detect_signals_function(request: str, tool_context: ToolContext) -> Dict[str, Any]:
    """Direct signal detection function using Gemini API"""
    
    step_logger.info("STEP 1: 🎯 Detecting demographic signals...")
    model = GenerativeModel(
        Modelconfig.flash_model,
        generation_config=GenerationConfig(
            temperature=0.1,
            max_output_tokens=2000,
            response_mime_type="application/json"  # Force JSON response
        ),
                    safety_settings={
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

    )
    
    # Double all curly braces to escape them, then use format for the request
    prompt = """Extract Qloo signals from user queries and respond ONLY with valid JSON.

    Detect and return JSON with these fields as arrays:
    - age: array of age ranges that match the query
    - gender: array of genders mentioned 
    - location: array of locations/regions mentioned

    Valid age ranges: 35_and_younger, 36_to_55, 55_and_older

    Age mapping examples:
    - "young fans" → ["35_and_younger"]
    - "under 50" → ["35_and_younger", "36_to_55"]
    - "teenagers" → ["35_and_younger"]

    Gender examples:
    - "men" → ["male"]
    - "women" → ["female"]

    Example outputs:
    {{"age": ["36_to_55"], "gender": ["male"], "location": ["New York"]}}
    {{"age": ["55_and_older"], "gender": ["male"], "location": ["south Texas"]}}

    user_query: {}
    """.format(request)  # Only the request gets formatted
    
    try:
        response = model.generate_content(prompt)
        
        if not response.text:
            return {"error": "Empty response from model"}
            
        signals = json.loads(response.text.strip())
        tool_context.state['detected_signals'] = signals
        # Clean result logging
        age_groups = ', '.join(signals.get('age', []))
        location = ', '.join(signals.get('location', []))
        step_logger.info(f"   ✅ Found: Age({age_groups}) Location({location})")
        
        return {
            "success": True,
            "detected_signals": signals,
            "message": "Signals detected successfully"
        }
        
    except Exception as e:
        return {"error": f"Detection failed: {str(e)}"}
    
def detect_theme_function(request: str, tool_context: ToolContext) -> Dict[str, Any]:
    """Direct theme function using Gemini API"""
    
    step_logger.info("🎯 Detecting Theme...")
    model = GenerativeModel(
        Modelconfig.flash_lite_model,
        generation_config=GenerationConfig(
            temperature=0.1,
            max_output_tokens=2000,
            response_mime_type="application/json"  # Force JSON response
        ),
        safety_settings={
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
    )
    
    # Use f-string instead of .format() to avoid KeyError with special characters
    prompt = f"""Identify the theme/event behind the theme of the story user wants to create.
    Make sure you detecte the theme/event behind the theme user have asked in full.
    

    Example outputs:
    {{"theme": ["Chelsea FC CWC 2025 victory"]}}

    user_query: {request}
    """
    
    try:
        response = model.generate_content(prompt)
        
        if not response.text:
            return {"error": "Empty response from model"}
        

        theme = json.loads(response.text.strip())
        # If theme is empty, use default Chelsea FC theme
        if not theme.get('theme') or len(theme.get('theme', [])) == 0:
            theme = {"theme": ["Chelsea FC CWC 2025 victory"]}

        tool_context.state['theme_detected'] = theme
        
        return {
            "success": True,
            "detected_theme": theme,
            "message": "theme detected successfully"
        }
        
    except Exception as e:
        return {"error": f"Detection failed: {str(e)}"}

def detect_specific_audiences(request: str, tool_context: ToolContext) -> Dict[str, Any]:
    """Detect specific audiences from the 104 available audiences based on user query"""
    
    # Full list of 104 audiences
    AVAILABLE_AUDIENCES = [
    {"id": "urn:audience:hobbies_and_interests:health_and_beauty", "name": "Health And Beauty"},
    {"id": "urn:audience:hobbies_and_interests:adventuring", "name": "Adventuring"},
    {"id": "urn:audience:hobbies_and_interests:photography", "name": "Photography"},
    # {"id": "urn:audience:hobbies_and_interests:tattoos", "name": "Tattoos"},
    {"id": "urn:audience:hobbies_and_interests:meditation", "name": "Meditation"},
    {"id": "urn:audience:hobbies_and_interests:hockey", "name": "Hockey"},
    {"id": "urn:audience:hobbies_and_interests:american_football", "name": "American Football"},
    {"id": "urn:audience:hobbies_and_interests:martial_arts", "name": "Martial Arts"},
    {"id": "urn:audience:hobbies_and_interests:running", "name": "Running"},
    # {"id": "urn:audience:hobbies_and_interests:secret_unravelers", "name": "Secret Unravelers"},
    # {"id": "urn:audience:hobbies_and_interests:motorcycles", "name": "Motorcycles"},
    {"id": "urn:audience:hobbies_and_interests:jewellery", "name": "Jewellery"},
    {"id": "urn:audience:hobbies_and_interests:hiking", "name": "Hiking"},
    {"id": "urn:audience:hobbies_and_interests:watches", "name": "Watches"},
    # {"id": "urn:audience:hobbies_and_interests:spy_enthusiast", "name": "Spy Enthusiast"},
    # {"id": "urn:audience:hobbies_and_interests:arts_crafts", "name": "Arts Crafts"},
    {"id": "urn:audience:hobbies_and_interests:wrestling", "name": "Wrestling"},
    # {"id": "urn:audience:hobbies_and_interests:home_organization", "name": "Home Organization"},
    {"id": "urn:audience:hobbies_and_interests:sneakerheads", "name": "Sneakerheads"},
    {"id": "urn:audience:hobbies_and_interests:travel", "name": "Travel"},
    {"id": "urn:audience:hobbies_and_interests:golf", "name": "Golf"},
    {"id": "urn:audience:hobbies_and_interests:high_fashion", "name": "High Fashion"},
    {"id": "urn:audience:hobbies_and_interests:home_decor", "name": "Home Decor"},
    {"id": "urn:audience:hobbies_and_interests:street_fashion", "name": "Street Fashion"},
    {"id": "urn:audience:hobbies_and_interests:video_gamer", "name": "Video Gamer"},
    {"id": "urn:audience:hobbies_and_interests:tennis", "name": "Tennis"},
    {"id": "urn:audience:hobbies_and_interests:swimming", "name": "Swimming"},
    {"id": "urn:audience:hobbies_and_interests:automotive", "name": "Automotive"},
    {"id": "urn:audience:hobbies_and_interests:musician", "name": "Musician"},
    {"id": "urn:audience:hobbies_and_interests:perfume", "name": "Perfume"},
    {"id": "urn:audience:hobbies_and_interests:casual_escapists", "name": "Casual Escapists"},
    {"id": "urn:audience:hobbies_and_interests:racing", "name": "Racing"},
    {"id": "urn:audience:hobbies_and_interests:adrenaline_rushers", "name": "Adrenaline Rushers"},
    {"id": "urn:audience:hobbies_and_interests:soccer", "name": "Soccer"},
    {"id": "urn:audience:hobbies_and_interests:yoga", "name": "Yoga"},
    {"id": "urn:audience:hobbies_and_interests:baseball", "name": "Baseball"},
    {"id": "urn:audience:hobbies_and_interests:architecture", "name": "Architecture"},
    {"id": "urn:audience:hobbies_and_interests:wine_enthusiast", "name": "Wine Enthusiast"},
    {"id": "urn:audience:hobbies_and_interests:basketball", "name": "Basketball"},
    {"id": "urn:audience:hobbies_and_interests:outdoors", "name": "Outdoors"},
    {"id": "urn:audience:hobbies_and_interests:fishing", "name": "Fishing"},
    {"id": "urn:audience:hobbies_and_interests:dance", "name": "Dance"},
    {"id": "urn:audience:spending_habits:vintage_apparel", "name": "Vintage Apparel"},
    {"id": "urn:audience:spending_habits:technology_enthusiast", "name": "Technology Enthusiast"},
    {"id": "urn:audience:spending_habits:watch_collecting", "name": "Watch Collecting"},
    {"id": "urn:audience:spending_habits:boutique_hotels", "name": "Boutique Hotels"},
    {"id": "urn:audience:spending_habits:discount_shoppers", "name": "Discount Shoppers"},
    {"id": "urn:audience:spending_habits:gourmand_fine_dining", "name": "Gourmand Fine Dining"},
    # {"id": "urn:audience:communities:aapi", "name": "AAPI"},
    # {"id": "urn:audience:communities:latino", "name": "Latino"},
    # {"id": "urn:audience:communities:lgbtq", "name": "LGBTQ"},
    # {"id": "urn:audience:communities:black", "name": "Black"},
    # {"id": "urn:audience:global_issues:climate_activism", "name": "Climate Activism"},
    # {"id": "urn:audience:global_issues:gender_equality", "name": "Gender Equality"},
    # {"id": "urn:audience:global_issues:racial_justice", "name": "Racial Justice"},
    # {"id": "urn:audience:global_issues:education_issues", "name": "Education Issues"},
    # {"id": "urn:audience:global_issues:ocean_health", "name": "Ocean Health"},
    # {"id": "urn:audience:global_issues:foreign_affairs", "name": "Foreign Affairs"},
    # {"id": "urn:audience:global_issues:wealth_inequality", "name": "Wealth Inequality"},
    # {"id": "urn:audience:global_issues:animal_issues", "name": "Animal Issues"},
    # {"id": "urn:audience:global_issues:mental_health", "name": "Mental Health"},
    # {"id": "urn:audience:global_issues:sustainability", "name": "Sustainability"},
    # {"id": "urn:audience:global_issues:fair_wages", "name": "Fair Wages"},
    # {"id": "urn:audience:investing_interests:stocks_bonds", "name": "Stocks Bonds"},
    # {"id": "urn:audience:investing_interests:angel_start_up_investing", "name": "Angel Start Up Investing"},
    # {"id": "urn:audience:investing_interests:real_estate", "name": "Real Estate"},
    {"id": "urn:audience:investing_interests:nft_collectors", "name": "NFT Collectors"},
    {"id": "urn:audience:investing_interests:cryptocurrency_enthusiasts", "name": "Cryptocurrency Enthusiasts"},
    {"id": "urn:audience:investing_interests:art_collectibles", "name": "Art Collectibles"},
    # {"id": "urn:audience:leisure:political_junkie", "name": "Political Junkie"},
    {"id": "urn:audience:leisure:arts_culture", "name": "Arts Culture"},
    {"id": "urn:audience:leisure:foodie", "name": "Foodie"},
    {"id": "urn:audience:leisure:cooking", "name": "Cooking"},
    {"id": "urn:audience:leisure:museums", "name": "Museums"},
    {"id": "urn:audience:leisure:cinephile", "name": "Cinephile"},
    {"id": "urn:audience:leisure:music_festivals", "name": "Music Festivals"},
    {"id": "urn:audience:leisure:avid_reader", "name": "Avid Reader"},
    {"id": "urn:audience:leisure:exercising", "name": "Exercising"},
    {"id": "urn:audience:leisure:coffee", "name": "Coffee"},
    {"id": "urn:audience:leisure:news_junkie", "name": "News Junkie"},
    {"id": "urn:audience:life_stage:parents_with_young_children", "name": "Parents With Young Children"},
    {"id": "urn:audience:life_stage:engaged", "name": "Engaged"},
    {"id": "urn:audience:life_stage:single", "name": "Single"},
    {"id": "urn:audience:life_stage:retirement", "name": "Retirement"},
    # {"id": "urn:audience:lifestyle_preferences_beliefs:astrology", "name": "Astrology"},
    # {"id": "urn:audience:lifestyle_preferences_beliefs:organic_ingredients", "name": "Organic Ingredients"},
    # {"id": "urn:audience:lifestyle_preferences_beliefs:christianity", "name": "Christianity"},
    # {"id": "urn:audience:lifestyle_preferences_beliefs:healthy_eating", "name": "Healthy Eating"},
    # {"id": "urn:audience:lifestyle_preferences_beliefs:judaism", "name": "Judaism"},
    # {"id": "urn:audience:lifestyle_preferences_beliefs:veganism", "name": "Veganism"},
    # {"id": "urn:audience:lifestyle_preferences_beliefs:spirituality", "name": "Spirituality"},
    # {"id": "urn:audience:lifestyle_preferences_beliefs:islam", "name": "Islam"},
    # {"id": "urn:audience:political_preferences:politically_progressive", "name": "Politically Progressive"},
    # {"id": "urn:audience:political_preferences:politically_center", "name": "Politically Center"},
    # {"id": "urn:audience:political_preferences:politically_conservative", "name": "Politically Conservative"},
    {"id": "urn:audience:professional_area:business_professional", "name": "Business Professional"},
    {"id": "urn:audience:professional_area:hospitality_professional", "name": "Hospitality Professional"},
    {"id": "urn:audience:professional_area:medical_professional", "name": "Medical Professional"},
    {"id": "urn:audience:professional_area:advertising_design", "name": "Advertising Design"},
    {"id": "urn:audience:professional_area:technology_professional", "name": "Technology Professional"},
    {"id": "urn:audience:professional_area:marketing_professional", "name": "Marketing Professional"},
    {"id": "urn:audience:professional_area:retail_professional", "name": "Retail Professional"},
    {"id": "urn:audience:professional_area:finance_professional", "name": "Finance Professional"},
    {"id": "urn:audience:professional_area:sales_professional", "name": "Sales Professional"}
]
    
    # Create audience list for LLM
    audience_list = []
    for aud in AVAILABLE_AUDIENCES:
        audience_list.append(f'"{aud["name"]}"')
    
    audience_text = ", ".join(audience_list)
        
    prompt = f"""Analyze this user query and identify ALL specific audiences they mentioned or strongly implied or are similar/related to what user is asking for.

User query: "{request}"

Available audiences: {[aud["name"] for aud in AVAILABLE_AUDIENCES]}

Instructions:
- Look for explicit mentions (e.g., "gamers", "runners", "fashionistas")
- Include strongly implied audiences (e.g., "fitness gear" implies "Running", "Health And Beauty")
- Include related audiences (e.g., "sports" could be "American Football")
- Return multiple audiences if multiple interests are mentioned
- Return empty list if no specific interests are mentioned
- Be inclusive rather than restrictive
- Note soccer is American word for football. Include these similar audiences unless said by user not to
- user asks for job,occupation,profession etc return all the valid audience from urn:audience:professional_area to users query
- any kind of spending characters/money habits use urn:spending_habits audience



Examples:
- "products for gamers and sneaker lovers" → ["Video Gamer", "Sneakerheads"]
- "merch for fitness enthusiasts" → ["Running", "Health And Beauty", "Martial Arts"]
- "items for fashion-forward travelers" → ["High Fashion", "Street Fashion", "Travel"]
- "products for 25 year old men" → []

Return JSON: {{"audience_names": ["Name1", "Name2"]}}
"""
    

    model = GenerativeModel(Modelconfig.flash_model)
    try:
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        if response_text.startswith('```json'):
            response_text = response_text.replace('```json', '').replace('```', '').strip()
        
        result = json.loads(response_text)
        detected_names = result.get("audience_names", [])
        
        # Map names to IDs
        detected_audiences = []
        name_to_audience = {aud["name"]: aud for aud in AVAILABLE_AUDIENCES}
        
        for name in detected_names:
            if name in name_to_audience:
                detected_audiences.append(name_to_audience[name])
        
        # Store in state
        if detected_audiences:
            audience_ids = [aud["id"] for aud in detected_audiences]
            tool_context.state['detected_audience_ids'] = audience_ids
            tool_context.state['detected_audience_names'] = [aud["name"] for aud in detected_audiences]
        else:
            tool_context.state['detected_audience_ids'] = []
            tool_context.state['detected_audience_names'] = []
        
        return {
            "success": True,
            "audiences_detected": len(detected_audiences),
            "audience_names": [aud["name"] for aud in detected_audiences],
            "audience_ids": [aud["id"] for aud in detected_audiences],
            "message": f"Detected {len(detected_audiences)} specific audiences" if detected_audiences else "No specific audiences mentioned"
        }
        
    except Exception as e:
        return {"error": f"Audience detection failed: {str(e)}"}




def convert_and_create_signals(tool_context: ToolContext) -> Dict[str, Any]:
    """Convert detected signals and call create_qloo_signals"""
    
    # Get detected signals from state
    detected = tool_context.state.get('detected_signals')
    if not detected:
        return {"error": "No signals detected"}
    
    # Get detected audiences (if any)
    audience_ids = tool_context.state.get('detected_audience_ids', [])
    
    # Convert to Qloo format - handle multiple values
    demographics = {}
    
    # Handle multiple age ranges - only add if not empty
    if detected.get('age') and len(detected['age']) > 0:
        demographics['age'] = ','.join(detected['age'])
    
    # Handle multiple genders - only add if not empty
    if detected.get('gender') and len(detected['gender']) > 0:
        demographics['gender'] = ','.join(detected['gender'])
    
    location = {}
    # Handle multiple locations - only add if not empty
    if detected.get('location') and len(detected['location']) > 0:
        location['query'] = ','.join(detected['location'])
    
    try:
        # Convert to JSON strings as required by create_qloo_signals
        demographics_json = json.dumps(demographics) if demographics else "{}"
        location_json = json.dumps(location) if location else "{}"

        # Call create_qloo_signals with audience support
        result = create_qloo_signals(
            demographics=demographics_json,
            location=location_json,
            audience_ids=audience_ids  # Pass audience IDs
        )

        return result
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to create QlooSignals object"
        }


def create_qloo_signals(
    demographics: str, 
    location: str,
    audience_ids: List[str] = None  # Add audience_ids parameter
) -> Dict[str, Any]:
    """
    Creates QlooSignals object with demographics, location, and audiences.
    """
    if not client:
        return {
            "success": False,
            "error": "Qloo client not available. Check QLOO_API_KEY configuration.",
            "message": "Failed to create QlooSignals - client not initialized"
        }
    
    try:
        # Parse input parameters
        demo_dict = json.loads(demographics) if demographics else {}
        location_dict = json.loads(location) if location else {}
        
        # Create QlooSignals object with all parameters
        signals = QlooSignals(
            demographics=demo_dict if demo_dict else None,
            location=location_dict if location_dict else None,
            audience_ids=audience_ids if audience_ids else None,  # Include audience IDs
            entity_queries=None
        )
        
        return {
            "success": True,
            "signals_created": True,
            "demographics_provided": bool(demo_dict),
            "location_provided": bool(location_dict),
            "audience_ids_provided": bool(audience_ids),
            "audience_count": len(audience_ids) if audience_ids else 0,
            "qloo_signal": signals,
            "demographics": demo_dict,
            "location": location_dict,
            "audience_ids": audience_ids or [],
            "message": "QlooSignals object created successfully",
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to create QlooSignals object"
        }




def get_insights_function(tool_context: ToolContext) -> Dict[str, Any]:
    """
    Get insights for the signals provided.
    
    ["movie", "tv_show", "artist", "brand", "tag"]
    :return: Formatted string containing insights
    """
    step_logger.info("STEP 3: 🧠 Collecting cultural insights from Qloo...")
    insight_summary = []
    # Check if we have detected signals
    detected_signals = tool_context.state.get('detected_signals')
    if not detected_signals:
        step_logger.error("   ❌ No signals found")
        return {"error": "No signals detected. Run signal detection first."}
    
    # Get detected audiences (if any)
    audience_ids = tool_context.state.get('detected_audience_ids', [])
    if not audience_ids:
        step_logger.error("   ❌ No audiences  found")
        return {"error": "Audience detection is not done. Run Audience detection detection first."}
    step_logger.info(f"   📊 Working with {len(audience_ids)} audiences")
    # Convert and create QlooSignals
    # Add Qloo signal to state context
    qloo_result = convert_and_create_signals(tool_context)  # Fixed: Added tool_context
    #tool_context.state['qloo_signal'] = qloo_result.get('qloo_signal')

    signals=qloo_result.get('qloo_signal')
    result = get_entity_brand_insights(signals,limit=15)
    tool_context.state['brand_insight']=result
    if result:
        insight_summary.append(result)
        step_logger.info("✅ Brand insights collected")
    else:
        step_logger.warning(" ⚠️ No Brand insights found")
            
    result = get_entity_movie_insights(signals,limit=5)
    tool_context.state['movie_insight']=result
    if result:
        insight_summary.append(result)
        step_logger.info("✅ Movie insights collected")
    else:
        step_logger.warning(" ⚠️  No movie insights found")

            
    result = get_entity_tv_show_insights(signals,limit=5)
    tool_context.state['tv_show_insight']=result
    if result:
        insight_summary.append(result)
        step_logger.info("✅ TV Show insights collected")
            
            
    result = get_entity_artist_insights(signals,limit=5)
    tool_context.state['artist_insight']=result
    if result:
        insight_summary.append(result)
        step_logger.info("✅ Artist insights collected")
    else:
        step_logger.warning(" ⚠️ No Artist insights found")
            

    result = get_tag_insights(signals,limit=10)
    tool_context.state['tag_insight']=result
    if result:
        insight_summary.append(result)
        step_logger.info("✅ tag insights collected")
    else:
        step_logger.warning(" ⚠️ No tag insights found")
            
    # result = get_entity_place_insights(signals)  # Fixed: should be place insights
    # if result:
    #     insight_summary.append(result)
    step_logger.info(f"Insights Summary:{insight_summary}")
    if insight_summary:
    
        # Join all results with separators
        separator = f"\n\n{'='*80}\n\n"
        insight_report=separator.join(insight_summary)

        ##Add the report to state context
        # tool_context.state['insight_summary']= insight_report
        return {
                "success": True,
                "message": "Insight report created for audience"
            }
    else:
        return "No insights available for the selected entity."




def get_entity_movie_insights(signals: Optional[QlooSignals], audience_ids: Optional[List[str]] = None,limit=3) -> str:
    """
    Get movie insights for any signals and or audience ID list passed.
    
    :param signals: Optional QlooSignals object containing signals for the query
    :param audience_ids: Optional list of audience IDs to filter results
    :return: Formatted string containing movie insights
    """
    insights = client.get_entity_insights(
        audience_ids=audience_ids or [],
        entity_type="movie",
        signals=signals,
        limit=limit
    )
    
    entities = insights.get('results', {}).get('entities', [])
    postman_url = insights.get('postman_url')

    if not entities:
        return "No movie results found."
    
    # Build formatted string result
    result = [f"""Top {limit} movies liked by the audience\n
              """]
    result.append("")

    # Process each movie entity
    for i, entity in enumerate(entities, 1):
        
        # Add movie header for multiple movies
        if len(entities) > 1:
            result.append(f"--- MOVIE RANK {i} ---")
        
        # Basic movie information
        result.append(f"MOVIE Name: {entity['name']}\n")
        # result.append(f"   Entity ID: {entity['entity_id']}")
        # Properties section
        props = entity.get('properties', {})
        
        # Release information
        # if 'release_year' in props:
        #     result.append(f"   Release Year: {props['release_year']}")
        # if 'release_date' in props:
        #     result.append(f"   Release Date: {props['release_date']}")
        
        # # Duration
        # if 'duration' in props:
        #     result.append(f"   Duration: {props['duration']} minutes")
        
        # Rating
        if 'content_rating' in props:
            result.append(f"Content Rating: {props['content_rating']}\n")
        
        # Description
        if 'description' in props:
            description = props['description']
            result.append(f"Plot: {description}\n")

        # Affinity and popularity - handle different possible locations
        affinity = 0
        if 'query' in entity and 'affinity' in entity['query']:
            affinity = entity['query']['affinity']
        elif 'affinity_score' in entity:
            affinity = entity['affinity_score']
        elif 'affinity' in entity:
            affinity = entity['affinity']
        
        popularity = entity.get('popularity', 0)
        result.append(f"Affinity: {affinity:.3f}\n")
        # result.append(f"   Popularity: {popularity:.3f}")


        # # Production details
        # if 'production_companies' in props and props['production_companies']:
        #     companies = props['production_companies'][:3]
        #     result.append(f"   Production: {', '.join(companies)}")
        
        # # Filming location
        # if 'filming_location' in props:
        #     result.append(f"   Filmed in: {props['filming_location']}")
        
        # # Release countries
        # if 'release_country' in props and props['release_country']:
        #     countries = props['release_country'][:3]
        #     result.append(f"   Released in: {', '.join(countries)}")
        
        # # Image
        # if 'image' in props and 'url' in props['image']:
        #     result.append(f"   Image: {props['image']['url']}")
        
        
        # Add spacing between movies
        if i < len(entities):
            result.append("")
    
    # Postman URL (only show once at the end)
    # if postman_url:
    #     result.append("")
    #     result.append(f"Postman URL: {postman_url}")
    #     result.append("   " + "="*60)
    
    return "\n".join(result)

def get_entity_brand_insights(signals: Optional[QlooSignals], audience_ids: Optional[List[str]] = None, limit=3) -> str:
    """
    Get brand insights for any signals and/or audience ID list passed.
    FIXED: Properly handle location signals for WKT polygons
    """
    
    # DEBUGGING: Let's see what's being passed to the API
    logger.info(f"🔍 Brand insights INPUT - signals type: {type(signals)}")
    if signals:
        logger.info(f"🔍 Brand insights INPUT - demographics: {signals.demographics}")
        logger.info(f"🔍 Brand insights INPUT - location: '{signals.location}' (type: {type(signals.location)})")
        logger.info(f"🔍 Brand insights INPUT - audience_ids: {signals.audience_ids}")
    logger.info(f"🔍 Brand insights INPUT - passed audience_ids: {audience_ids}")
    
    insights = client.get_entity_insights(
        audience_ids=audience_ids or [],
        entity_type="brand",
        signals=signals,
        limit=limit
    )
    
    # DEBUGGING: Let's see what the API returns
    logger.info(f"🔍 Brand insights API response success: {insights.get('success')}")
    logger.info(f"🔍 Brand insights API response keys: {list(insights.keys())}")
    
    if not insights.get('success'):
        logger.error(f"🔍 Brand insights API error: {insights.get('error')}")
        logger.error(f"🔍 Brand insights API status code: {insights.get('status_code')}")
        logger.error(f"🔍 Brand insights API postman URL: {insights.get('postman_url')}")
        return f"API Error: {insights.get('error', 'Unknown error')}"
    
    entities = insights.get('results', {}).get('entities', [])
    logger.info(f"🔍 Brand insights entities found: {len(entities)}")
    
    if entities:
        logger.info(f"🔍 First entity sample: {entities[0].get('name', 'No name')} - {entities[0].get('entity_id', 'No ID')}")

    if not entities:
        # Check if results structure is different
        logger.info(f"🔍 Full results structure: {insights.get('results', {})}")
        return "No brand results found."
    
    # Build formatted string result
    result = [f"""Top {limit} brands liked by the audience\n"""]
    result.append("")

    # Process each brand entity
    for i, entity in enumerate(entities, 1):
        
        # Add brand header for multiple brands
        if len(entities) > 1:
            result.append(f"--- BRAND Rank {i} ---\n")
        
        # Basic brand information
        result.append(f"BRAND Name: {entity['name']}\n")
        
        # Properties section
        props = entity.get('properties', {})
        
        # Brand description
        if 'short_description' in props:
            result.append(f"Brand Description: {props['short_description']}\n")
        
        # Affinity and popularity - handle different possible locations
        affinity = 0
        if 'query' in entity and 'affinity' in entity['query']:
            affinity = entity['query']['affinity']
        elif 'affinity_score' in entity:
            affinity = entity['affinity_score']
        elif 'affinity' in entity:
            affinity = entity['affinity']
        
        result.append(f"   Affinity: {affinity:.3f}\n")
        
        # Add spacing between brands
        if i < len(entities):
            result.append("")
    
    return "\n".join(result)


def get_entity_place_insights(signals: Optional[QlooSignals], audience_ids: Optional[List[str]] = None,limit=5) -> str:
    """
    Get place insights for any signals and/or audience ID list passed.
    
    :param signals: Optional QlooSignals object containing signals for the query
    :param audience_ids: Optional list of audience IDs to filter results
    :return: Formatted string containing place insights
    """
    insights = client.get_entity_insights(
        audience_ids=audience_ids or [],
        entity_type="place",
        signals=signals,
        limit=limit
    )
    
    entities = insights.get('results', {}).get('entities', [])
    postman_url = insights.get('postman_url')

    if not entities:
        return "No place results found."
    
    # Build formatted string result
    result = [f"""Below are the details of the top 10 places liked by the audience based on the signals provided.
              This can help you understand the places the audience likes to visit or engage with most"""]
    result.append("")

    # Process each place entity
    for i, entity in enumerate(entities, 1):
        
        # Add place header for multiple places
        if len(entities) > 1:
            result.append(f"--- PLACE {i} ---")
        
        # Basic place information
        result.append(f"PLACE: {entity['name']}")
        result.append(f"   Entity ID: {entity['entity_id']}")
        
        # Affinity and popularity - handle different possible locations
        affinity = 0
        if 'query' in entity and 'affinity' in entity['query']:
            affinity = entity['query']['affinity']
        elif 'affinity_score' in entity:
            affinity = entity['affinity_score']
        elif 'affinity' in entity:
            affinity = entity['affinity']
        
        result.append(f"   Affinity: {affinity:.3f}")
        
        # Audience Growth
        growth = 0
        if 'query' in entity and 'measurements' in entity['query']:
            growth = entity['query']['measurements'].get('audience_growth', 0)
        
        if growth > 0:
            growth_indicator = "(up)"
        elif growth < 0:
            growth_indicator = "(down)"
        else:
            growth_indicator = "(flat)"
        
        result.append(f"   Audience Growth: {growth:.2f} {growth_indicator}")
        
        props = entity.get('properties', {})
        
        # Address
        if 'address' in props:
            result.append(f"   Address: {props['address']}")
        
        # Business details
        if 'business_rating' in props:
            result.append(f"   Business Rating: {props['business_rating']}/5.0")
        
        if 'phone' in props:
            result.append(f"   Phone: {props['phone']}")
        
        if 'website' in props:
            result.append(f"   Website: {props['website']}")
        
        # Neighborhood/Location
        if 'neighborhood' in props:
            result.append(f"   Neighborhood: {props['neighborhood']}")
        
        # Status
        if 'is_closed' in props:
            status = "Closed" if props['is_closed'] else "Open"
            result.append(f"   Status: {status}")
        
        # Keywords/Popular terms
        if 'keywords' in props and props['keywords']:
            top_keywords = [kw.get('name', str(kw)) for kw in props['keywords'][:5]]
            result.append(f"   Keywords: {', '.join(top_keywords)}")
        
        # Coordinates
        if 'location' in entity:
            loc = entity['location']
            lat = loc.get('lat', 'N/A')
            lon = loc.get('lon', 'N/A')
            result.append(f"   Coordinates: {lat}, {lon}")
        
        # Add spacing between places
        if i < len(entities):
            result.append("")
    
    # Postman URL (only show once at the end)
    # if postman_url:
    #     result.append("")
    #     result.append(f"Postman URL: {postman_url}")
    #     result.append("   " + "="*60)
    
    return "\n".join(result)

def get_entity_artist_insights(signals: Optional[QlooSignals], audience_ids: Optional[List[str]] = None,limit=3) -> str:
    """
    Get artist insights for any signals and/or audience ID list passed.
    
    :param signals: Optional QlooSignals object containing signals for the query
    :param audience_ids: Optional list of audience IDs to filter results
    :return: Formatted string containing artist insights
    """
    insights = client.get_entity_insights(
        audience_ids=audience_ids or [],
        entity_type="artist",
        signals=signals,
        limit=3
    )
    
    entities = insights.get('results', {}).get('entities', [])
    postman_url = insights.get('postman_url')

    if not entities:
        return "No artist results found."
    
    # Build formatted string result
    result = [f"""Top {limit} artists/musicians audience likes. You can use these insights to understand the taste of music of the audience"""]
    result.append("\n")

    # Process each artist entity
    for i, entity in enumerate(entities, 1):
        
        # Add artist header for multiple artists
        if len(entities) > 1:
            result.append(f"--- ARTIST Rank {i} ---")
        
        # Basic artist information
        result.append(f"ARTIST Name: {entity['name']}\n")
        # result.append(f"   Entity ID: {entity['entity_id']}")
        props = entity.get('properties', {})
        
        descriptions = props.get('short_descriptions') or props.get('short_description')
        if descriptions:
            if isinstance(descriptions, list) and descriptions:
                desc = descriptions[0]
            else:
                desc = descriptions            
            description_text = desc.get('value') if isinstance(desc, dict) else desc
            if description_text:
                result.append(f"Description: {description_text}\n")
        
        # Affinity and popularity - handle different possible locations
        affinity = 0
        if 'query' in entity and 'affinity' in entity['query']:
            affinity = entity['query']['affinity']
        elif 'affinity_score' in entity:
            affinity = entity['affinity_score']
        elif 'affinity' in entity:
            affinity = entity['affinity']
        
        result.append(f"   Affinity: {affinity:.3f}\n")
        
        # # Audience Growth
        # growth = 0
        # if 'query' in entity and 'measurements' in entity['query']:
        #     growth = entity['query']['measurements'].get('audience_growth', 0)
        
        # if growth > 0:
        #     growth_indicator = "(up)"
        # elif growth < 0:
        #     growth_indicator = "(down)"
        # else:
        #     growth_indicator = "(flat)"
        
        # result.append(f"   Audience Growth: {growth:.2f} {growth_indicator}")
        

        
        # # Birth info
        # if 'date_of_birth' in props:
        #     result.append(f"   Born: {props['date_of_birth']}")
        # if 'place_of_birth' in props:
        #     result.append(f"   Birthplace: {props['place_of_birth']}")
        
        # # Alternative names
        # if 'akas' in props and props['akas']:
        #     akas = [aka.get('value', aka) for aka in props['akas'][:3] if aka]
        #     result.append(f"   Also known as: {', '.join(akas)}")
        
        # # Image
        # if 'image' in props and 'url' in props['image']:
        #     result.append(f"   Image: {props['image']['url']}")
        
        # External platforms
        if 'external' in entity:
            for platform, data in entity['external'].items():
                if data and isinstance(data, list) and len(data) > 0:
                    platform_data = data[0]
                    if platform == 'spotify':
                        followers = platform_data.get('followers', 'N/A')
                        listeners = platform_data.get('monthly_listeners', 'N/A')
                        result.append(f"Popularity in Spotify: {followers} followers, {listeners} monthly listeners\n")
                    # else:
                    #     result.append(f"   {platform.title()}: Available")
        
        # Add spacing between artists
        if i < len(entities):
            result.append("")
    
    # Postman URL (only show once at the end)
    # if postman_url:
    #     result.append("")
    #     result.append(f"Postman URL: {postman_url}")
    #     result.append("   " + "="*60)
    
    return "\n".join(result)



def get_entity_tv_show_insights(signals: Optional[QlooSignals], audience_ids: Optional[List[str]] = None,limit=3) -> str:
    """
    Get TV show insights for any signals and/or audience ID list passed.
    
    :param signals: Optional QlooSignals object containing signals for the query
    :param audience_ids: Optional list of audience IDs to filter results
    :return: Formatted string containing TV show insights
    """
    insights = client.get_entity_insights(
        audience_ids=audience_ids or [],
        entity_type="tv_show",
        signals=signals,
        limit=limit
    )
    
    entities = insights.get('results', {}).get('entities', [])
    postman_url = insights.get('postman_url')

    if not entities:
        return "No TV show results found."
    
    # Build formatted string result
    result = [f"""Top {limit} TV shows liked by the audience\n"""]
    result.append("")

    # Process each TV show entity
    for i, entity in enumerate(entities, 1):
        
        # Add TV show header for multiple TV shows
        if len(entities) > 1:
            result.append(f"--- TV SHOW Rank {i} ---")
        
        # Basic TV show information
        result.append(f"TV SHOW Name: {entity['name']}")
        # result.append(f"   Entity ID: {entity['entity_id']}")
        
        # Affinity and popularity - handle different possible locations
        affinity = 0
        if 'query' in entity and 'affinity' in entity['query']:
            affinity = entity['query']['affinity']
        elif 'affinity_score' in entity:
            affinity = entity['affinity_score']
        elif 'affinity' in entity:
            affinity = entity['affinity']
        
        result.append(f"   Affinity: {affinity:.3f}")
        
        # # Audience Growth
        # growth = 0
        # if 'query' in entity and 'measurements' in entity['query']:
        #     growth = entity['query']['measurements'].get('audience_growth', 0)
        
        # if growth > 0:
        #     growth_indicator = "(up)"
        # elif growth < 0:
        #     growth_indicator = "(down)"
        # else:
        #     growth_indicator = "(flat)"
        
        # result.append(f"   Audience Growth: {growth:.2f} {growth_indicator}")
        
        props = entity.get('properties', {})
        
        # # Release information
        # if 'release_year' in props:
        #     result.append(f"   Started: {props['release_year']}")
        
        # # Episode duration
        # if 'duration' in props:
        #     result.append(f"   Episode Length: {props['duration']} minutes")
        
        # Rating
        if 'content_rating' in props:
            result.append(f"   Content Rating: {props['content_rating']}\n")
        
        # Description
        if 'description' in props:
            description = props['description']
            result.append(f"   Description: {description}\n")
        
        # # Production details
        # if 'production_companies' in props and props['production_companies']:
        #     companies = props['production_companies'][:3]
        #     result.append(f"   Production: {', '.join(companies)}")
        
        # # Filming location
        # if 'filming_location' in props:
        #     result.append(f"   Filmed in: {props['filming_location']}")
        
        # # Image
        # if 'image' in props and 'url' in props['image']:
        #     result.append(f"   Image: {props['image']['url']}")
        
        # Add spacing between TV shows
        if i < len(entities):
            result.append("")
    
    # Postman URL (only show once at the end)
    # if postman_url:
    #     result.append("")
    #     result.append(f"Postman URL: {postman_url}")
    #     result.append("   " + "="*60)
    
    return "\n".join(result)



def get_tag_insights(signals: Optional[QlooSignals], audience_ids: Optional[List[str]] = None, tag_filter: Optional[str] = None,limit=10) -> str:
    """
    Get tag insights for any signals and/or audience ID list passed.
    
    :param signals: Optional QlooSignals object containing signals for the query
    :param audience_ids: Optional list of audience IDs to filter results
    :param tag_filter: Optional specific tag filter (e.g., "urn:tag:lifestyle:qloo")
    :return: Formatted string containing tag insights
    """
    insights = client.get_tag_insights(
        audience_ids=audience_ids or [],
        signals=signals,
        limit=limit,
        tag_filter=tag_filter
    )
    tags = insights.get('results', {}).get('tags', [])
    postman_url = insights.get('postman_url')
    #logging.info(f"Postman URL: {postman_url}")

    if not tags:
        return "No tag results found."
    
    # Build formatted string result
    tag_type = tag_filter or "all types"
    result = [f"""Top {limit} tags describing audience based on the signals provided.
              Tag insights reveal the themes, characteristics, and content attributes that resonate most strongly with your target audience.
              They're like a "DNA profile" of your audience's cultural preferences.\n"""]
    result.append("")

    # Process each tag
    for i, tag in enumerate(tags, 1):
        
        # Add tag header for multiple tags
        if len(tags) > 1:
            result.append(f"--- TAG Rank {i} ---")
        
        # Basic tag information
        result.append(f"TAG Name: {tag['name']}\n")
        # result.append(f"   Tag ID: {tag['tag_id']}")
        
        # Safe access to affinity score
        affinity = tag.get('query', {}).get('affinity', 0)
        result.append(f"   Affinity: {affinity:.3f}\n")
        
        # Tag type information
        result.append(f"   Type: {tag.get('subtype', 'Unknown')}\n")
        
        # What entities this tag applies to
        entity_types = tag.get('types', [])
        if entity_types:
            # Clean up the URN format for readability
            clean_types = []
            for entity_type in entity_types:
                if 'urn:entity:' in entity_type:
                    clean_type = entity_type.replace('urn:entity:', '')
                    clean_types.append(clean_type)
                else:
                    clean_types.append(entity_type)
            result.append(f"   Applies to: {', '.join(clean_types)}\n")
        
        # Add spacing between tags
        if i < len(tags):
            result.append("")
    
    # Postman URL (only show once at the end)
    # if postman_url:
    #     result.append("")
    #     result.append(f"Postman URL: {postman_url}")
    #     result.append("   " + "="*60)
    
    return "\n".join(result)