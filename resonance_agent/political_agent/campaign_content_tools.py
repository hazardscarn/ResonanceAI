# campaign_content_generation/campaign_content_tools.py
import logging
import json
import uuid
import io
from typing import Dict, List, Optional, Any
from datetime import datetime
from google.adk.tools import ToolContext
from google.genai import types
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel
from vertexai.generative_models import GenerativeModel, GenerationConfig
from google.cloud import storage

from .config import Modelconfig, SecretConfig

# Initialize logging
logger = logging.getLogger(__name__)
step_logger = logging.getLogger("AGENT_STEPS")

# Initialize Vertex AI
project_id = SecretConfig.get_google_cloud_project()
location = SecretConfig.get_google_cloud_location()
vertexai.init(project=project_id, location=location)

# GCS bucket for campaign images
CAMPAIGN_IMAGES_BUCKET = f"{project_id}-campaign-content"

async def restore_candidate_info_from_artifacts(
    tool_context: ToolContext
) -> Dict[str, Any]:
    """
    Restore candidate information from artifacts when state is lost.
    
    Returns:
        Dict with restored candidate information
    """
    
    try:
        step_logger.info("🔄 Attempting to restore candidate info from artifacts...")
        
        # Try to get artifact filename from state
        artifact_filename = tool_context.state.get("temp:candidate_analysis_artifact")
        
        if not artifact_filename:
            # Look for any analysis artifacts in the session
            try:
                artifacts = await tool_context.list_artifacts()
                analysis_artifacts = [a for a in artifacts if 'candidate_analysis' in a.lower()]
                if analysis_artifacts:
                    artifact_filename = analysis_artifacts[-1]  # Get most recent
                    step_logger.info(f"🔍 Found analysis artifact: {artifact_filename}")
            except:
                pass
        
        if not artifact_filename:
            return {
                "success": False,
                "message": "No candidate analysis artifacts found"
            }
        
        # Try to load the artifact and extract candidate info
        try:
            artifact = await tool_context.load_artifact(filename=artifact_filename)
            if artifact and artifact.inline_data:
                # Parse filename to extract candidate info (as fallback)
                # Format: candidate_analysis_{candidate}_{location}.pkl
                filename_parts = artifact_filename.replace('.pkl', '').split('_')
                if len(filename_parts) >= 3:
                    candidate_name_part = filename_parts[2]  # candidate name
                    candidate_name = candidate_name_part.replace('_', ' ').title()
                    
                    step_logger.info(f"🎯 Restored candidate name from filename: {candidate_name}")
                    
                    return {
                        "success": True,
                        "candidate_name": candidate_name,
                        "candidate_base": "progressive",  # Default assumption
                        "source": "artifact_filename"
                    }
        except Exception as e:
            step_logger.warning(f"Failed to load artifact {artifact_filename}: {e}")
        
        return {
            "success": False,
            "message": "Could not restore candidate info from artifacts"
        }
        
    except Exception as e:
        step_logger.error(f"Error restoring candidate info: {e}")
        return {
            "success": False,
            "message": f"Error during restoration: {str(e)}"
        }


async def collect_campaign_inputs(
    tool_context: ToolContext,
    campaign_goal: str,
    call_to_action: str,
    emotional_tone: str,
    platform_type: str,
    key_message: Optional[str] = None,
    candidate_name: Optional[str] = None,
    candidate_base: Optional[str] = None
) -> Dict[str, Any]:
    """
    Collect campaign manager inputs and prepare for content generation.
    
    Args:
        campaign_goal: The goal of this campaign (e.g., fundraising, awareness, volunteer recruitment)
        call_to_action: What action you want people to take (e.g., donate, RSVP, share)
        emotional_tone: The emotional tone (e.g., hopeful, urgent, inspiring, humorous)
        platform_type: Platform for the content (social media, direct mail, sms, email)
        key_message: Optional key message to communicate
        candidate_name: Optional candidate name (if state is lost)
        candidate_base: Optional candidate base (if state is lost)
    
    Returns:
        Dict with campaign inputs saved to context
    """
    
    try:
        step_logger.info("📝 Collecting campaign manager inputs...")
        
        # Validate required inputs
        required_fields = {
            "campaign_goal": campaign_goal,
            "call_to_action": call_to_action, 
            "emotional_tone": emotional_tone,
            "platform_type": platform_type
        }
        
        missing_fields = [field for field, value in required_fields.items() if not value or value.strip() == ""]
        if missing_fields:
            return {
                "status": "error",
                "message": f"Missing required fields: {', '.join(missing_fields)}",
                "required_fields": list(required_fields.keys())
            }
        
        # Try to get candidate info from multiple sources
        metadata = tool_context.state.get("temp:analysis_metadata", {})
        
        # First try from state
        retrieved_candidate_name = metadata.get('candidate_name')
        retrieved_candidate_base = metadata.get('candidate_base')
        
        # If not in state, try from function parameters  
        if not retrieved_candidate_name and candidate_name:
            retrieved_candidate_name = candidate_name
            step_logger.info("📝 Using candidate name from parameters")
            
        if not retrieved_candidate_base and candidate_base:
            retrieved_candidate_base = candidate_base
            step_logger.info("📝 Using candidate base from parameters")
            
        # If still not found, try to restore from artifacts
        if not retrieved_candidate_name:
            restored_info = await restore_candidate_info_from_artifacts(tool_context)
            if restored_info['success']:
                retrieved_candidate_name = restored_info['candidate_name']
                retrieved_candidate_base = restored_info['candidate_base']
                step_logger.info("📝 Restored candidate info from artifacts")
        
        if not retrieved_candidate_name:
            return {
                "status": "error",
                "message": "No candidate information found. Please provide candidate_name and candidate_base parameters, or run create_candidate_analysis first.",
                "suggestion": "Try: collect_campaign_inputs(..., candidate_name='Your Candidate', candidate_base='progressive')"
            }
        
        # Save campaign inputs to state with robust persistence
        campaign_inputs = {
            "campaign_goal": campaign_goal.strip(),
            "call_to_action": call_to_action.strip(),
            "emotional_tone": emotional_tone.strip(),
            "platform_type": platform_type.strip(),
            "key_message": key_message.strip() if key_message else "",
            "candidate_name": retrieved_candidate_name,
            "candidate_base": retrieved_candidate_base,
            "collected_timestamp": datetime.now().isoformat()
        }
        
        # Save to state with multiple keys for persistence
        tool_context.state['campaign_inputs'] = campaign_inputs
        
        # Also save candidate info separately for future use
        tool_context.state['candidate_name'] = retrieved_candidate_name
        tool_context.state['candidate_base'] = retrieved_candidate_base
        
        # Update analysis metadata if it was restored
        if not metadata.get('candidate_name'):
            tool_context.state["temp:analysis_metadata"] = {
                "candidate_name": retrieved_candidate_name,
                "candidate_base": retrieved_candidate_base,
                "location": metadata.get('location', 'Unknown'),
                "age": metadata.get('age'),
                "gender": metadata.get('gender'),
                "tag_names": metadata.get('tag_names', [])
            }
        
        step_logger.info(f"✅ Campaign inputs collected for {retrieved_candidate_name}")
        step_logger.info(f"   🎯 Goal: {campaign_goal}")
        step_logger.info(f"   📱 Platform: {platform_type}")
        step_logger.info(f"   🎭 Tone: {emotional_tone}")
        
        return {
            "status": "success",
            "message": f"Campaign inputs collected for {retrieved_candidate_name}",
            "campaign_inputs": campaign_inputs,
            "candidate_info_source": "state" if metadata.get('candidate_name') else "parameters/artifacts",
            "next_steps": [
                "Inputs ready for content generation",
                "Use generate_campaign_content to create personalized messages"
            ]
        }
        
    except Exception as e:
        logger.error(f"Error collecting campaign inputs: {e}")
        return {
            "status": "error",
            "message": f"Failed to collect campaign inputs: {str(e)}"
        }


async def generate_campaign_content(
    tool_context: ToolContext
) -> Dict[str, Any]:
    """
    Generate personalized campaign content using cultural insights and campaign inputs.
    
    Returns:
        Dict with generated campaign content variants and image concept
    """
    
    try:
        step_logger.info("🎨 Generating personalized campaign content...")
        
        # Get campaign inputs with fallback
        campaign_inputs = tool_context.state.get('campaign_inputs')
        
        if not campaign_inputs:
            # Try to restore candidate info if inputs are missing
            candidate_name = tool_context.state.get('candidate_name')
            candidate_base = tool_context.state.get('candidate_base')
            
            if not candidate_name:
                restored_info = await restore_candidate_info_from_artifacts(tool_context)
                if restored_info['success']:
                    candidate_name = restored_info['candidate_name']
                    candidate_base = restored_info['candidate_base']
            
            return {
                "status": "error",
                "message": "No campaign inputs found. Please run collect_campaign_inputs first.",
                "available_candidate_info": {
                    "candidate_name": candidate_name,
                    "candidate_base": candidate_base
                } if candidate_name else None,
                "suggestion": f"Try: collect_campaign_inputs(campaign_goal='awareness', call_to_action='share', emotional_tone='hopeful', platform_type='social media'{f", candidate_name='{candidate_name}'" if candidate_name else ''})"
            }
        
        # Get cultural insights from state
        all_insights = tool_context.state.get('all_content_insights', {})
        if not all_insights:
            return {
                "status": "error", 
                "message": "No cultural insights found. Please run generate_content_insights first."
            }
        
        # Get identified locations info
        identified_locations = tool_context.state.get('identified_locations', {})
        location_tag = identified_locations.get('tag', 'Target Audience')
        
        step_logger.info(f"🎯 Creating content for: {location_tag}")
        step_logger.info(f"📊 Using insights: {list(all_insights.keys())}")
        
        # Prepare cultural preferences for prompt
        cultural_prefs = {
            "movies": all_insights.get("movies", "No movie insights available"),
            "brands": all_insights.get("brands", "No brand insights available"), 
            "artists": all_insights.get("artists", "No artist insights available"),
            "tv_shows": all_insights.get("tv_shows", "No TV show insights available"),
            "tags": all_insights.get("tags", "No tag insights available"),
            "places": all_insights.get("place_insights", "No place insights available")
        }
        
        # Build the content generation prompt
        content_prompt = f"""You are an AI campaign strategist. Create a political campaign ad tailored to a target audience using their cultural preferences and campaign goals.

### Cultural Preferences of Target Audience
Movies they like: {cultural_prefs['movies']}

Brands they follow: {cultural_prefs['brands']}

Artists/Musicians they enjoy: {cultural_prefs['artists']}

TV Shows they watch: {cultural_prefs['tv_shows']}

Interest Tags: {cultural_prefs['tags']}

Places they like: {cultural_prefs['places']}

### Campaign Context
Candidate: {campaign_inputs['candidate_name']} ({campaign_inputs['candidate_base']})
Key message to communicate: {campaign_inputs['key_message'] or 'None specified'}

### Campaign Manager Inputs  
Goal of this post: {campaign_inputs['campaign_goal']}
Call to action: {campaign_inputs['call_to_action']}
Emotional tone: {campaign_inputs['emotional_tone']}
Platform type: {campaign_inputs['platform_type']}

### Task:
Using the above cultural preferences and campaign details:

1. Write **3 variants of ad copy** (short, medium, long) that resonate with this audience's cultural touchpoints.
2. Suggest an **image concept** (describe style, elements, cultural references).
3. Ensure tone matches {campaign_inputs['emotional_tone']} and include appropriate legal disclaimer.
4. Optimize language and references to feel natural for this cultural group (no pandering).
5. Make it platform-ready for {campaign_inputs['platform_type']}.

Format your response as JSON with this structure:
{{
    "short_copy": "Brief version (1-2 sentences)",
    "medium_copy": "Medium version (3-4 sentences)", 
    "long_copy": "Extended version (5+ sentences)",
    "image_concept": "Detailed description of image style and elements",
    "cultural_connections": ["list", "of", "cultural", "references", "used"],
    "platform_optimization": "How this is optimized for {campaign_inputs['platform_type']}",
    "legal_disclaimer": "Required legal text"
}}
"""
        
        # Generate content using Gemini
        model = GenerativeModel(
            Modelconfig.flash_model,
            generation_config=GenerationConfig(
                temperature=0.7,  # Higher for creativity
                max_output_tokens=4000,
                response_mime_type="application/json"
            )
        )
        
        step_logger.info("🧠 Generating content with cultural intelligence...")
        response = model.generate_content(content_prompt)
        
        if not response.text:
            return {
                "status": "error",
                "message": "Empty response from content generation model"
            }
        
        # Parse the generated content
        generated_content = json.loads(response.text.strip())
        
        # Save generated content to state
        campaign_content = {
            "generated_content": generated_content,
            "cultural_preferences_used": cultural_prefs,
            "campaign_inputs_used": campaign_inputs,
            "location_tag": location_tag,
            "generation_timestamp": datetime.now().isoformat()
        }
        
        tool_context.state['campaign_content'] = campaign_content
        
        step_logger.info("✅ Campaign content generated successfully")
        step_logger.info(f"   📝 3 copy variants created")
        step_logger.info(f"   🎨 Image concept developed")
        step_logger.info(f"   🎯 Optimized for {campaign_inputs['platform_type']}")
        
        return {
            "status": "success",
            "message": f"Campaign content generated for {campaign_inputs['candidate_name']}",
            "generated_content": generated_content,
            "cultural_connections": generated_content.get("cultural_connections", []),
            "platform_optimization": generated_content.get("platform_optimization", ""),
            "content_summary": {
                "target_audience": location_tag,
                "platform": campaign_inputs['platform_type'],
                "tone": campaign_inputs['emotional_tone'],
                "goal": campaign_inputs['campaign_goal']
            },
            "next_steps": [
                "Content ready for review",
                "Use generate_campaign_image to create visual content",
                "Review and approve before deployment"
            ]
        }
        
    except Exception as e:
        logger.error(f"Error generating campaign content: {e}")
        return {
            "status": "error",
            "message": f"Failed to generate campaign content: {str(e)}"
        }


async def generate_campaign_image(
    tool_context: ToolContext,
    image_style: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate campaign image using Imagen based on the content and cultural insights.
    
    Args:
        image_style: Optional override for image style (defaults to platform-appropriate)
    
    Returns:
        Dict with generated image info and download URL
    """
    
    try:
        step_logger.info("🎨 Generating campaign image with Imagen...")
        
        # Get campaign content
        campaign_content = tool_context.state.get('campaign_content')
        if not campaign_content:
            return {
                "status": "error",
                "message": "No campaign content found. Please run generate_campaign_content first."
            }
        
        generated_content = campaign_content['generated_content']
        campaign_inputs = campaign_content['campaign_inputs_used']
        
        # Get image concept from generated content
        base_image_concept = generated_content.get('image_concept', '')
        
        # Determine platform-specific image requirements
        platform_specs = get_platform_image_specs(campaign_inputs['platform_type'])
        
        # Build enhanced image prompt
        style_guidance = image_style or platform_specs['style']
        
        enhanced_prompt = f"""
{base_image_concept}

Style requirements: {style_guidance}
Aspect ratio: {platform_specs['aspect_ratio']}
Campaign context: Political campaign for {campaign_inputs['candidate_name']}
Tone: {campaign_inputs['emotional_tone']}
Platform: {campaign_inputs['platform_type']}

Additional requirements:
- Professional political campaign aesthetic
- Clear, engaging composition  
- Appropriate for {campaign_inputs['platform_type']} platform
- {campaign_inputs['emotional_tone']} emotional tone
- High quality, campaign-ready image
- No text overlay (will be added separately)
- DO NOT CREATE IMAGES OF A SINGLE PERSON
- IMAGE CREATED SHOULD NOT BE SINGLING OUT ANY RACE
- DO NOT CREATE IMAGES OF THE CANDIDATE. IMAGES GENERATED SHOULD BE OF A THEME THAN A SINGLE PERSON

"""
        
        step_logger.info(f"🎯 Generating {platform_specs['aspect_ratio']} image for {campaign_inputs['platform_type']}")
        
        # Initialize Imagen model
        image_model = ImageGenerationModel.from_pretrained(Modelconfig.imagen4_fast)
        
        # Generate image with retry logic
        max_retries = 2
        for attempt in range(max_retries):
            try:
                step_logger.info(f"🎨 Image generation attempt {attempt + 1}/{max_retries}")
                
                response = image_model.generate_images(
                    prompt=enhanced_prompt,
                    number_of_images=1,
                    aspect_ratio=platform_specs['aspect_ratio'],
                    safety_filter_level="allow_most",
                    person_generation="allow_adult",
                    add_watermark=False
                )
                
                if response.images and len(response.images) > 0:
                    image = response.images[0]
                    
                    # Save image to GCS
                    image_filename = f"campaign_{campaign_inputs['candidate_name'].lower().replace(' ', '_')}_{uuid.uuid4().hex[:8]}.png"
                    
                    # Convert PIL image to bytes
                    img_byte_arr = io.BytesIO()
                    image._pil_image.save(img_byte_arr, format='PNG')
                    image_bytes = img_byte_arr.getvalue()
                    
                    # Upload to GCS
                    gcs_url = await upload_image_to_gcs(image_bytes, image_filename)
                    
                    # Save image info to state
                    image_info = {
                        "image_filename": image_filename,
                        "gcs_url": gcs_url,
                        "image_concept": base_image_concept,
                        "enhanced_prompt": enhanced_prompt,
                        "platform_specs": platform_specs,
                        "style_used": style_guidance,
                        "generation_timestamp": datetime.now().isoformat(),
                        "candidate": campaign_inputs['candidate_name'],
                        "platform": campaign_inputs['platform_type']
                    }
                    
                    tool_context.state['campaign_image'] = image_info
                    
                    step_logger.info(f"✅ Campaign image generated successfully")
                    step_logger.info(f"   📁 Saved as: {image_filename}")
                    step_logger.info(f"   🔗 GCS URL: {gcs_url}")

                    clean_image_url = None
                    if gcs_url:
                        clean_image_url = gcs_url
                        if clean_image_url.startswith("gs://"):
                            clean_image_url = clean_image_url.replace("gs://", "https://storage.googleapis.com/")
                        while '.com//' in clean_image_url:
                            clean_image_url = clean_image_url.replace('.com//', '.com/')
                    
                    return {
                        "status": "success",
                        "message": f"Campaign image generated for {campaign_inputs['candidate_name']}",
                        "image_info": image_info,
                        "download_url": clean_image_url,
                        "platform_optimized": campaign_inputs['platform_type'],
                        "aspect_ratio": platform_specs['aspect_ratio'],
                        "style_applied": style_guidance,
                        "next_steps": [
                            f"Image ready at: {clean_image_url}",
                            "Review image quality and campaign alignment",
                            "Download for campaign deployment"
                        ]
                    }
                else:
                    logger.warning(f"No image generated on attempt {attempt + 1}")
                    
            except Exception as e:
                logger.warning(f"Image generation attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise e
        
        return {
            "status": "error",
            "message": "Failed to generate image after multiple attempts"
        }
        
    except Exception as e:
        logger.error(f"Error generating campaign image: {e}")
        return {
            "status": "error",
            "message": f"Failed to generate campaign image: {str(e)}"
        }


async def create_campaign_package(
    tool_context: ToolContext
) -> Dict[str, Any]:
    """
    Create a complete campaign package with content and image ready for deployment as HTML.
    
    Returns:
        Dict with complete campaign package as downloadable HTML artifact
    """
    
    try:
        step_logger.info("📦 Creating complete campaign package...")
        
        # Get campaign content and image
        campaign_content = tool_context.state.get('campaign_content')
        campaign_image = tool_context.state.get('campaign_image')
        campaign_inputs = tool_context.state.get('campaign_inputs')
        
        if not campaign_content:
            return {
                "status": "error",
                "message": "No campaign content found. Please generate content first."
            }
        
        generated_content = campaign_content['generated_content']
        cultural_prefs = campaign_content['cultural_preferences_used']
        location_tag = campaign_content['location_tag']
        
        # Build comprehensive campaign package as HTML
        package_date = datetime.now().strftime("%B %d, %Y at %I:%M %p")

        clean_image_url = None
        if campaign_image:
            clean_image_url = campaign_image['gcs_url']
            if clean_image_url.startswith("gs://"):
                clean_image_url = clean_image_url.replace("gs://", "https://storage.googleapis.com/")
            while '.com//' in clean_image_url:
                clean_image_url = clean_image_url.replace('.com//', '.com/')
        
        # HTML structure with embedded CSS
        html_package = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Campaign Package - {campaign_inputs['candidate_name']}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f9f9f9;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }}
        .header .subtitle {{
            font-size: 1.2em;
            opacity: 0.9;
            margin-top: 10px;
        }}
        .meta-info {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .meta-item {{
            text-align: center;
        }}
        .meta-label {{
            font-weight: bold;
            color: #666;
            font-size: 0.9em;
            text-transform: uppercase;
        }}
        .meta-value {{
            font-size: 1.1em;
            color: #333;
            margin-top: 5px;
        }}
        .section {{
            background: white;
            margin-bottom: 30px;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .section-header {{
            background: #4a5568;
            color: white;
            padding: 15px 20px;
            font-size: 1.3em;
            font-weight: bold;
        }}
        .section-content {{
            padding: 20px;
        }}
        .copy-variant {{
            background: #f7fafc;
            border-left: 4px solid #4299e1;
            padding: 15px;
            margin: 15px 0;
            border-radius: 0 5px 5px 0;
        }}
        .copy-variant h4 {{
            margin: 0 0 10px 0;
            color: #2d3748;
        }}
        .cta-box {{
            background: linear-gradient(135deg, #48bb78, #38a169);
            color: white;
            padding: 20px;
            text-align: center;
            font-size: 1.4em;
            font-weight: bold;
            border-radius: 8px;
            margin: 20px 0;
        }}
        .image-section {{
            text-align: center;
        }}
        .image-download {{
            display: inline-block;
            background: #4299e1;
            color: white;
            padding: 12px 24px;
            text-decoration: none;
            border-radius: 6px;
            margin: 10px;
            font-weight: bold;
        }}
        .image-download:hover {{
            background: #3182ce;
        }}
        .cultural-list {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 15px;
        }}
        .cultural-item {{
            background: #edf2f7;
            padding: 15px;
            border-radius: 6px;
            border-left: 3px solid #667eea;
        }}
        .checklist {{
            list-style: none;
            padding: 0;
        }}
        .checklist li {{
            background: #f7fafc;
            margin: 8px 0;
            padding: 12px;
            border-radius: 4px;
            border-left: 3px solid #48bb78;
        }}
        .checklist li:before {{
            content: "☐ ";
            font-weight: bold;
            color: #48bb78;
        }}
        .footer {{
            text-align: center;
            color: #666;
            font-style: italic;
            margin-top: 40px;
            padding: 20px;
            border-top: 2px solid #e2e8f0;
        }}
        .emoji {{
            font-size: 1.2em;
            margin-right: 8px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{campaign_inputs['candidate_name']} Campaign Package</h1>
        <div class="subtitle">{location_tag}</div>
    </div>

    <div class="meta-info">
        <div class="meta-item">
            <div class="meta-label">Generated</div>
            <div class="meta-value">{package_date}</div>
        </div>
        <div class="meta-item">
            <div class="meta-label">Platform</div>
            <div class="meta-value">{campaign_inputs['platform_type']}</div>
        </div>
        <div class="meta-item">
            <div class="meta-label">Goal</div>
            <div class="meta-value">{campaign_inputs['campaign_goal']}</div>
        </div>
        <div class="meta-item">
            <div class="meta-label">Tone</div>
            <div class="meta-value">{campaign_inputs['emotional_tone']}</div>
        </div>
    </div>

    <div class="section">
        <div class="section-header">
            <span class="emoji">📋</span>Campaign Copy Variants
        </div>
        <div class="section-content">
            <div class="copy-variant">
                <h4>Short Copy (Quick Impact)</h4>
                <p>{generated_content.get('short_copy', 'Not available')}</p>
            </div>
            
            <div class="copy-variant">
                <h4>Medium Copy (Balanced Detail)</h4>
                <p>{generated_content.get('medium_copy', 'Not available')}</p>
            </div>
            
            <div class="copy-variant">
                <h4>Long Copy (Full Message)</h4>
                <p>{generated_content.get('long_copy', 'Not available')}</p>
            </div>
        </div>
    </div>

    <div class="section">
        <div class="section-header">
            <span class="emoji">🎯</span>Call to Action
        </div>
        <div class="section-content">
            <div class="cta-box">
                {campaign_inputs['call_to_action']}
            </div>
        </div>
    </div>

    <div class="section">
        <div class="section-header">
            <span class="emoji">🎨</span>Image Concept
        </div>
        <div class="section-content">
            <p>{generated_content.get('image_concept', 'Not available')}</p>
        </div>
    </div>
"""

        # Add image section if available
        if campaign_image:
            # Use the correct URL format as mentioned by user
            # download_url = campaign_image['gcs_url'].replace("gs://", "https://storage.googleapis.com")
            # download_url = download_url.replace('.com//', '.com/')
            # Step 1: Convert gs:// if present
            # if download_url.startswith("gs://"):
            #     download_url = download_url.replace("gs://", "https://storage.googleapis.com/")

            # # Step 2: Keep removing double slashes until none remain
            # while '.com//' in download_url:
            #     download_url = download_url.replace('.com//', '.com/')
            download_url = clean_image_url


            html_package += f"""
    <div class="section">
        <div class="section-header">
            <span class="emoji">📸</span>Generated Image
        </div>
        <div class="section-content image-section">
            <p><strong>Filename:</strong> {campaign_image['image_filename']}</p>
            <p><strong>Aspect Ratio:</strong> {campaign_image['platform_specs']['aspect_ratio']}</p>
            <p><strong>Style:</strong> {campaign_image['style_used']}</p>
            
            <a href="{download_url}" class="image-download" target="_blank">
                📥 Download High-Quality Image
            </a>
        </div>
    </div>
"""
        
        # Add cultural connections
        cultural_connections = generated_content.get('cultural_connections', [])
        if cultural_connections:
            html_package += f"""
    <div class="section">
        <div class="section-header">
            <span class="emoji">🎭</span>Cultural Connections Used
        </div>
        <div class="section-content">
            <div class="cultural-list">
"""
            for connection in cultural_connections:
                html_package += f'                <div class="cultural-item">{connection}</div>\n'
            
            html_package += """            </div>
        </div>
    </div>
"""
        
        # Add platform optimization
        platform_optimization = generated_content.get('platform_optimization', '')
        if platform_optimization:
            html_package += f"""
    <div class="section">
        <div class="section-header">
            <span class="emoji">📱</span>Platform Optimization
        </div>
        <div class="section-content">
            <p>{platform_optimization}</p>
        </div>
    </div>
"""
        
        # Add cultural insights summary
        html_package += f"""
    <div class="section">
        <div class="section-header">
            <span class="emoji">📊</span>Target Audience Cultural Profile
        </div>
        <div class="section-content">
            <div class="cultural-list">
                <div class="cultural-item">
                    <h4>🎬 Movie Preferences</h4>
                    <p>{cultural_prefs.get('movies', 'Not available')[:300]}...</p>
                </div>
                
                <div class="cultural-item">
                    <h4>🏢 Brand Preferences</h4>
                    <p>{cultural_prefs.get('brands', 'Not available')[:300]}...</p>
                </div>
                
                <div class="cultural-item">
                    <h4>🎵 Music/Artist Preferences</h4>
                    <p>{cultural_prefs.get('artists', 'Not available')[:300]}...</p>
                </div>
                <div class="cultural-item">
                    <h4>🏢 Places</h4>
                    <p>{cultural_prefs.get('places', 'Not available')[:300]}...</p>
                </div>
                
                <div class="cultural-item">
                    <h4>🏷️ Interest Tags</h4>
                    <p>{cultural_prefs.get('tags', 'Not available')[:300]}...</p>
                </div>
            </div>
        </div>
    </div>
"""

        # Add locations section showing all analyzed coordinates
        identified_locations = tool_context.state.get("identified_locations", {})
        coordinates = identified_locations.get("coordinates", [])
        
        if coordinates:
            html_package += f"""
    <div class="section">
        <div class="section-header">
            <span class="emoji">📍</span>Analyzed Locations & Geographic Coverage
        </div>
        <div class="section-content">
            <div class="meta-info">
                <div class="meta-item">
                    <div class="meta-label">Total Locations</div>
                    <div class="meta-value">{len(coordinates)}</div>
                </div>
                <div class="meta-item">
                    <div class="meta-label">Analysis Method</div>
                    <div class="meta-value">Polygon Area</div>
                </div>
                <div class="meta-item">
                    <div class="meta-label">Filter Criteria</div>
                    <div class="meta-value">{identified_locations.get('tag', 'Custom Filter')}</div>
                </div>
            </div>
            
            <h4>📋 Filter Description:</h4>
            <p><em>{identified_locations.get('description', 'No description available')}</em></p>
            
            <h4>🗺️ Geographic Coordinates Analyzed:</h4>
            <div style="background: #f7fafc; padding: 15px; border-radius: 6px; font-family: monospace; max-height: 300px; overflow-y: auto;">
"""
            
            # Add coordinate list
            for i, coord in enumerate(coordinates, 1):
                lat, lon = coord[0], coord[1]
                html_package += f"                <div style='margin: 3px 0;'>{i:3d}. Latitude: {lat:>10.6f}, Longitude: {lon:>11.6f}</div>\n"
            
            # Calculate geographic bounds
            if len(coordinates) > 1:
                lats = [coord[0] for coord in coordinates]
                lons = [coord[1] for coord in coordinates]
                
                min_lat, max_lat = min(lats), max(lats)
                min_lon, max_lon = min(lons), max(lons)
                center_lat = (min_lat + max_lat) / 2
                center_lon = (min_lon + max_lon) / 2
                
                html_package += f"""
            </div>
            
            <h4>📏 Geographic Bounds:</h4>
            <div class="cultural-list">
                <div class="cultural-item">
                    <h4>🗺️ Coverage Area</h4>
                    <p><strong>North:</strong> {max_lat:.6f}<br>
                    <strong>South:</strong> {min_lat:.6f}<br>
                    <strong>East:</strong> {max_lon:.6f}<br>
                    <strong>West:</strong> {min_lon:.6f}</p>
                </div>
                
                <div class="cultural-item">
                    <h4>🎯 Center Point</h4>
                    <p><strong>Latitude:</strong> {center_lat:.6f}<br>
                    <strong>Longitude:</strong> {center_lon:.6f}</p>
                </div>
                
                <div class="cultural-item">
                    <h4>📐 Analysis Polygon</h4>
                    <p>Cultural insights generated using WKT polygon covering all {len(coordinates)} filtered locations</p>
                </div>
            </div>
"""
            else:
                html_package += """
            </div>
            
            <div class="cultural-item">
                <h4>📍 Single Location Analysis</h4>
                <p>Analysis based on individual coordinate point</p>
            </div>
"""
            
            html_package += """
        </div>
    </div>
"""

        html_package += f"""
    <div class="section">
        <div class="section-header">
            <span class="emoji">⚖️</span>Legal Requirements
        </div>
        <div class="section-content">
            <p>{generated_content.get('legal_disclaimer', 'Standard campaign disclaimers apply')}</p>
        </div>
    </div>

    <div class="section">
        <div class="section-header">
            <span class="emoji">📝</span>Deployment Checklist
        </div>
        <div class="section-content">
            <ul class="checklist">
                <li>Review all copy variants for accuracy</li>
                <li>Approve image content and quality</li>
                <li>Verify legal disclaimers are included</li>
                <li>Test on target platform format</li>
                <li>Schedule deployment timing</li>
                <li>Monitor engagement metrics</li>
            </ul>
        </div>
    </div>

    <div class="footer">
        Generated by Campaign Content Intelligence Platform<br>
        Target Audience: {location_tag}<br>
        Cultural Data: Qloo API Integration
    </div>

</body>
</html>"""
        
        # Create artifact as HTML
        package_bytes = html_package.encode('utf-8')
        artifact_part = types.Part(
            inline_data=types.Blob(
                mime_type="text/html",  # Changed to HTML
                data=package_bytes
            )
        )
        
        # Create filename with .html extension
        safe_candidate = campaign_inputs['candidate_name'].lower().replace(' ', '_').replace('.', '')
        safe_platform = campaign_inputs['platform_type'].lower().replace(' ', '_')
        package_filename = f"campaign_package_{safe_candidate}_{safe_platform}.html"
        
        # Save artifact
        version = await tool_context.save_artifact(filename=package_filename, artifact=artifact_part)
        
        step_logger.info(f"✅ Campaign package created: {package_filename}")
        
        return {
            "status": "success",
            "message": f"Complete campaign package created for {campaign_inputs['candidate_name']}",
            "package_filename": package_filename,
            "package_version": version,
            "package_format": "HTML",
            "package_summary": {
                "candidate": campaign_inputs['candidate_name'],
                "platform": campaign_inputs['platform_type'],
                "target_audience": location_tag,
                "content_variants": 3,
                "image_included": bool(campaign_image),
                "cultural_connections": len(cultural_connections),
                "file_size_kb": round(len(package_bytes) / 1024, 1)
            },
            "download_info": {
                "format": "HTML (.html)",
                "features": [
                    "📱 Responsive design",
                    "🎨 Professional styling", 
                    "📥 Direct image download links",
                    "📋 Interactive checklist",
                    "📊 Visual cultural profile"
                ],
                "contains": [
                    "📝 3 Copy Variants (Short/Medium/Long)",
                    "🎨 Image Concept & Download Link", 
                    "🎯 Cultural Connections Used",
                    "📱 Platform Optimization Notes",
                    "📊 Target Audience Profile",
                    "⚖️ Legal Requirements",
                    "📝 Deployment Checklist"
                ]
            },
            "image_download": clean_image_url if clean_image_url else None,
            "next_steps": [
                f"📥 Download HTML package: '{package_filename}'",
                "🌐 Open in browser to view formatted campaign package",
                "🔍 Review content and image quality",
                "✅ Complete deployment checklist",
                "🚀 Deploy on target platform"
            ]
        }
        
    except Exception as e:
        logger.error(f"Error creating campaign package: {e}")
        return {
            "status": "error",
            "message": f"Failed to create campaign package: {str(e)}"
        }


async def debug_campaign_state(
    tool_context: ToolContext
) -> Dict[str, Any]:
    """
    Debug function to check the current state of campaign data.
    
    Returns:
        Dict with current state information
    """
    
    try:
        step_logger.info("🔍 Debugging campaign state...")
        
        state_info = {
            "campaign_inputs": tool_context.state.get('campaign_inputs'),
            "candidate_name": tool_context.state.get('candidate_name'),
            "candidate_base": tool_context.state.get('candidate_base'),
            "analysis_metadata": tool_context.state.get("temp:analysis_metadata"),
            "analysis_artifact": tool_context.state.get("temp:candidate_analysis_artifact"),
            "identified_locations": tool_context.state.get("identified_locations"),
            "all_content_insights": tool_context.state.get('all_content_insights'),
            "campaign_content": tool_context.state.get('campaign_content'),
            "campaign_image": tool_context.state.get('campaign_image')
        }
        
        # Try to restore candidate info if missing
        if not state_info["candidate_name"]:
            restoration_result = await restore_candidate_info_from_artifacts(tool_context)
            state_info["restoration_attempt"] = restoration_result
        
        # Count available artifacts
        try:
            artifacts = await tool_context.list_artifacts()
            state_info["available_artifacts"] = artifacts
        except:
            state_info["available_artifacts"] = "Could not list artifacts"
        
        return {
            "status": "success",
            "message": "Campaign state debug completed",
            "state_info": state_info,
            "recommendations": [
                "Check if candidate_name is available",
                "Verify cultural insights are present", 
                "Ensure campaign inputs are collected",
                "Use restoration if needed"
            ]
        }
        
    except Exception as e:
        logger.error(f"Error debugging campaign state: {e}")
        return {
            "status": "error",
            "message": f"Failed to debug campaign state: {str(e)}"
        }


def get_platform_image_specs(platform_type: str) -> Dict[str, str]:
    """Get platform-specific image specifications."""
    
    platform_specs = {
        "social media": {
            "aspect_ratio": "1:1",
            "style": "eye-catching, social media optimized, vibrant colors"
        },
        "instagram": {
            "aspect_ratio": "1:1", 
            "style": "Instagram-style, visually appealing, modern design"
        },
        "facebook": {
            "aspect_ratio": "16:9",
            "style": "Facebook-optimized, clear messaging, professional"
        },
        "twitter": {
            "aspect_ratio": "16:9",
            "style": "Twitter-style, concise visual impact, engaging"
        },
        "direct mail": {
            "aspect_ratio": "4:3",
            "style": "print-ready, high contrast, clear details"
        },
        "email": {
            "aspect_ratio": "16:9", 
            "style": "email-friendly, clear composition, professional"
        },
        "sms": {
            "aspect_ratio": "1:1",
            "style": "mobile-optimized, simple, clear messaging"
        }
    }
    
    return platform_specs.get(platform_type.lower(), {
        "aspect_ratio": "16:9",
        "style": "professional campaign style, clear composition"
    })


async def upload_image_to_gcs(image_bytes: bytes, filename: str) -> str:
    """Upload image to Google Cloud Storage and return public URL."""
    
    try:
        # Initialize GCS client
        client = storage.Client(project=project_id)
        bucket = client.bucket(CAMPAIGN_IMAGES_BUCKET)
        
        # Create blob and upload
        blob = bucket.blob(filename)
        blob.upload_from_string(image_bytes, content_type='image/png')
        
        # Make blob publicly readable
        blob.make_public()
        
        return blob.public_url
        
    except Exception as e:
        logger.error(f"Failed to upload image to GCS: {e}")
        # Return local reference as fallback
        return f"https://storage.googleapis.com//{CAMPAIGN_IMAGES_BUCKET}/{filename}"