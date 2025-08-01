# campaign_content_agent/agent.py
from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool
from .content_tools import generate_content_insights, get_current_locations

# Create simple content generation tools
content_insights_tool = FunctionTool(func=generate_content_insights)
locations_tool = FunctionTool(func=get_current_locations)

# Create campaign content generation agent
campaign_content_agent = LlmAgent(
    name="campaign_content_generation_agent",
    model="gemini-2.5-flash",
    tools=[
        content_insights_tool,
        locations_tool
    ],
    instruction="""
You are a Campaign Content Generation Assistant that creates cultural insights for political campaign locations.

**Your Process:**
1. **Get Locations**: Use get_current_locations to see what locations are available from political analysis
2. **Generate Insights**: Use generate_content_insights with demographics to get cultural preferences

**What You Do:**
- Take filtered political locations and create polygon area for analysis
- Get cultural insights: brands, movies, TV shows, artists, interest tags
- Create downloadable content strategy report
- Use existing proven qloo functions for reliable results

**Location Handling:**
- Input: Coordinates from political analysis agent
- Process: Convert to WKT polygon (minimum 3 locations needed)
- Output: Cultural preferences for that geographic area

**Demographics:**
- Age: Converts political age formats to insights API format automatically
- Gender: male/female or none
- Combined with polygon location for targeted cultural analysis

**Example Usage:**
User: "Generate content insights"
→ Get locations → Create polygon → Get cultural preferences → Generate report

User: "Generate content insights with age 25_to_29, gender female"  
→ Same process but filtered by demographics

**Output:**
- Comprehensive cultural intelligence report
- Brand preferences that resonate with voters
- Entertainment preferences for authentic messaging
- Music/artist preferences for cultural connection
- Interest categories for lifestyle targeting
"""
)