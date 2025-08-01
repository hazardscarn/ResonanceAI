import os
from dotenv import load_dotenv

# Load environment variables from .env file (for local development)
load_dotenv()

import streamlit as st
import pandas as pd
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import time
import uuid
from src.heatmap import get_complete_heatmap_analysis
from app_components.heatmap_visuals import create_heatmap, create_segment_map, process_data

# ============================================================================
# MODERN STYLING COMPONENT (Inspired by Blue FC App)
# ============================================================================

def style_component():
    style = """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

        /* Hide Streamlit elements */
        .stApp > header {background-color: transparent;}
        .stApp > header [data-testid="stHeader"] {display: none;}
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .stDeployButton {display:none;}

        /* Global styling */
        html, body, [class*="st"] {
            font-family: 'Inter', sans-serif;
            color: #1a1a1a;
        }

        .stApp {
            background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        }

        /* Navigation */
        .nav-header {
            background: white;
            padding: 1.5rem 2rem;
            margin: -1rem -1rem 2rem -1rem;
            box-shadow: 0 2px 12px rgba(0,0,0,0.08);
            border-radius: 0 0 16px 16px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .nav-logo {
            font-size: 1.1rem;
            font-weight: 600;
            color: #dc2626;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            flex: 1;
            max-width: 70%;
            line-height: 1.4;
        }

        .status-indicator {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-size: 0.9rem;
            font-weight: 500;
        }

        .status-connected {
            background: linear-gradient(135deg, #d1fae5, #10b981);
            color: #065f46;
            border: 1px solid #059669;
        }

        .status-loading {
            background: linear-gradient(135deg, #fef3c7, #f59e0b);
            color: #92400e;
            border: 1px solid #d97706;
        }

        /* Hero Section */
        .hero-section {
            background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%);
            color: white;
            padding: 3rem;
            border-radius: 20px;
            text-align: center;
            margin-bottom: 2rem;
            position: relative;
            overflow: hidden;
        }

        .hero-title {
            font-size: 2.8rem;
            font-weight: 800;
            margin-bottom: 1rem;
            text-shadow: 0 4px 8px rgba(0,0,0,0.3);
        }

        .hero-subtitle {
            font-size: 1.2rem;
            opacity: 0.9;
            max-width: 600px;
            margin: 0 auto;
            line-height: 1.6;
        }

        /* Cards */
        .content-card {
            background: white;
            border-radius: 16px;
            padding: 2rem;
            margin: 1rem 0;
            box-shadow: 0 4px 20px rgba(0,0,0,0.06);
            border: 1px solid #e2e8f0;
            transition: all 0.3s ease;
        }

        .content-card:hover {
            box-shadow: 0 8px 32px rgba(0,0,0,0.12);
            transform: translateY(-2px);
        }

        /* Map Container */
        .map-container {
            background: white;
            border-radius: 16px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 4px 20px rgba(0,0,0,0.06);
            border: 1px solid #e2e8f0;
        }

        /* Legend Card */
        .legend-card {
            background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
            border: 1px solid #cbd5e1;
        }

        /* Buttons */
        .stButton > button {
            background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%) !important;
            color: white !important;
            font-weight: 600 !important;
            padding: 0.75rem 2rem !important;
            border-radius: 12px !important;
            border: none !important;
            transition: all 0.3s ease !important;
            font-size: 1rem !important;
            box-shadow: 0 4px 12px rgba(220, 38, 38, 0.3) !important;
        }

        .stButton > button:hover {
            background: linear-gradient(135deg, #991b1b 0%, #7f1d1d 100%) !important;
            transform: translateY(-1px) !important;
            box-shadow: 0 6px 20px rgba(220, 38, 38, 0.4) !important;
        }

        /* Loading animation */
        .loading-animation {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid #f3f3f3;
            border-top: 3px solid #dc2626;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        /* Success callout */
        .success-callout {
            background: linear-gradient(135deg, #d1fae5, #a7f3d0);
            border: 1px solid #059669;
            border-radius: 12px;
            padding: 1rem;
            margin: 1rem 0;
            color: #065f46;
        }

        /* Info callout */
        .info-callout {
            background: linear-gradient(135deg, #dbeafe, #bfdbfe);
            border: 1px solid #3b82f6;
            border-radius: 12px;
            padding: 1rem;
            margin: 1rem 0;
            color: #1e40af;
        }

        /* Stats Grid */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin: 1rem 0;
        }

        .stat-card {
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
            border: 1px solid #e2e8f0;
        }

        .stat-number {
            font-size: 2rem;
            font-weight: 700;
            color: #dc2626;
            margin-bottom: 0.5rem;
        }

        .stat-label {
            color: #64748b;
            font-size: 0.9rem;
            font-weight: 500;
        }

        /* Responsive */
        @media (max-width: 768px) {
            .hero-title { font-size: 2rem; }
            .nav-header { 
                flex-direction: column; 
                text-align: center; 
                gap: 1rem; 
            }
            .nav-logo { 
                max-width: 100%; 
                text-align: center; 
            }
            .content-card {
                padding: 1.5rem;
            }
        }
        </style>
        """
    return style

# ============================================================================
# SESSION STATE MANAGEMENT
# ============================================================================

def initialize_session_state():
    """Initialize session state variables to prevent reruns"""
    if "analysis_id" not in st.session_state:
        st.session_state.analysis_id = None
    
    if "political_data" not in st.session_state:
        st.session_state.political_data = None
    
    if "entity_comparison_data" not in st.session_state:
        st.session_state.entity_comparison_data = None
    
    if "entity1_data" not in st.session_state:
        st.session_state.entity1_data = None
    
    if "params" not in st.session_state:
        st.session_state.params = None
    
    if "loading_state" not in st.session_state:
        st.session_state.loading_state = False
    
    if "map_interactions" not in st.session_state:
        st.session_state.map_interactions = 0

# ============================================================================
# DATA PROCESSING FUNCTIONS
# ============================================================================

@st.cache_data
def process_entity_data(entity1_df, entity2_df):
    """Process entity comparison data with caching"""
    entity1_df1 = entity1_df.copy()
    entity2_df1 = entity2_df.copy()
    
    # Rename columns
    entity1_df1 = entity1_df1.rename(columns={
        'affinity': 'entity1_affinity',
        'popularity': 'entity1_popularity'
    })
    entity2_df1 = entity2_df1.rename(columns={
        'affinity': 'entity2_affinity', 
        'popularity': 'entity2_popularity'
    })
    
    # Merge on location
    df_combined = pd.merge(
        entity1_df1,
        entity2_df1[['latitude', 'longitude', 'entity2_affinity', 'entity2_popularity']],
        on=['latitude', 'longitude'], 
        how='inner'
    )
    
    # Calculate net popularity (multiply by 100 to get percentage points)
    df_combined['entity1_net_popularity'] = (df_combined['entity1_popularity'] - df_combined['entity2_popularity']) * 100
    
    # Ensure numeric types
    df_combined['entity1_net_popularity'] = pd.to_numeric(df_combined['entity1_net_popularity'], errors='coerce')
    
    # Remove any rows with NaN values
    df_combined = df_combined.dropna(subset=['entity1_net_popularity'])
    
    return df_combined

@st.cache_data
def process_political_data(conservative_df, progressive_df):
    """Process political competition data with caching"""
    return process_data(conservative_df, progressive_df)

# ============================================================================
# MAP CREATION FUNCTIONS
# ============================================================================

def create_entity_legend_map(df, entity1_name, entity2_name, map_key):
    """Create entity comparison map with unique key"""
    center_lat, center_lon = df['latitude'].mean(), df['longitude'].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
    
    # Define color ranges based on entity1_net_popularity
    def get_color_and_label(net_pop):
        if net_pop >= 5:
            return '#000080', f'Strong {entity1_name} (5+)'
        elif net_pop >= 2:
            return '#4169E1', f'Moderate {entity1_name} (2-5)'
        elif net_pop >= 0.5:
            return '#ADD8E6', f'Lean {entity1_name} (0.5-2)'
        elif net_pop >= -0.5:
            return '#D3D3D3', 'Competitive (-0.5 to 0.5)'
        elif net_pop >= -2:
            return '#FFB6C1', f'Lean {entity2_name} (-0.5 to -2)'
        elif net_pop >= -5:
            return '#DC143C', f'Moderate {entity2_name} (-2 to -5)'
        else:
            return '#8B0000', f'Strong {entity2_name} (-5+)'
    
    for _, row in df.iterrows():
        color, label = get_color_and_label(row['entity1_net_popularity'])
        
        popup_text = f"""
        <b>{label}</b><br>
        Net Score: {row['entity1_net_popularity']:.2f}<br>
        {entity1_name}: {row['entity1_popularity']:.3f}<br>
        {entity2_name}: {row['entity2_popularity']:.3f}
        """
        
        folium.CircleMarker(
            [row['latitude'], row['longitude']],
            radius=8,
            color='black',
            fillColor=color,
            fillOpacity=0.8,
            popup=popup_text
        ).add_to(m)
    
    return m

def create_political_legend_map(df, map_key):
    """Create political competition map with unique key"""
    center_lat, center_lon = df['latitude'].mean(), df['longitude'].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
    
    # Define color ranges based on conservative_net_popularity
    def get_color_and_label(net_pop):
        if net_pop >= 5:
            return '#8B0000', 'Strong Conservative (5+)'
        elif net_pop >= 2:
            return '#DC143C', 'Moderate Conservative (2-5)'
        elif net_pop >= 0.5:
            return '#FFB6C1', 'Lean Conservative (0.5-2)'
        elif net_pop >= -0.5:
            return '#D3D3D3', 'Competitive (-0.5 to 0.5)'
        elif net_pop >= -2:
            return '#ADD8E6', 'Lean Progressive (-0.5 to -2)'
        elif net_pop >= -5:
            return '#4169E1', 'Moderate Progressive (-2 to -5)'
        else:
            return '#000080', 'Strong Progressive (-5+)'
    
    for _, row in df.iterrows():
        color, label = get_color_and_label(row['conservative_net_popularity'])
        
        popup_text = f"""
        <b>{label}</b><br>
        Net Score: {row['conservative_net_popularity']:.2f}<br>
        Conservative: {row['conservative_popularity']:.3f}<br>
        Progressive: {row['progressive_popularity']:.3f}
        """
        
        folium.CircleMarker(
            [row['latitude'], row['longitude']],
            radius=8,
            color='black',
            fillColor=color,
            fillOpacity=0.8,
            popup=popup_text
        ).add_to(m)
    
    return m

# ============================================================================
# UI COMPONENTS
# ============================================================================

def render_navigation():
    """Render modern navigation header"""
    data_loaded = st.session_state.political_data is not None
    
    st.markdown("""
    <div class="hero-section">
        <div class="hero-title">🗳️ Political Campaign Analyzer</div>
        <div class="hero-subtitle">
            Advanced Political Intelligence & Entity Comparison Dashboard
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="nav-header">
        <div class="nav-logo">
            <span style="font-weight: 800; font-size: 1.3rem;">📊 Political Intelligence Platform</span><br>
            <span style="font-size: 0.9rem; opacity: 0.8;">Powered by Qloo Cultural Intelligence API</span>
        </div>
        <div class="status-indicator {'status-connected' if data_loaded else 'status-loading'}">
            {'🟢' if data_loaded else '🔴'} Data {'Loaded' if data_loaded else 'Not Loaded'}
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_parameter_panel():
    """Render parameter input panel with modern styling"""
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown("### 🎯 Campaign Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📍 Location & Demographics")
        location = st.text_input(
            "Location",
            value=st.session_state.get("last_location", "North Carolina"),
            help="Enter city, state, or region",
            key="location_input"
        )
        
        age = st.selectbox(
            "Age Group",
            options=["", "24_and_younger", "25_to_29", "30_to_34", "35_and_younger", "36_to_55", "35_to_44", "45_to_54", "55_and_older"],
            index=0,
            help="Optional: Select age group",
            key="age_input"
        )
        
        gender = st.selectbox(
            "Gender",
            options=["", "male", "female"],
            index=0,
            help="Optional: Select gender",
            key="gender_input"
        )
    
    with col2:
        st.markdown("#### 👥 Entity Comparison")
        entity1 = st.text_input(
            "Entity 1",
            value=st.session_state.get("last_entity1", "Roy Cooper"),
            help="First entity for comparison",
            key="entity1_input"
        )
        
        entity2 = st.text_input(
            "Entity 2", 
            value=st.session_state.get("last_entity2", "Michael Whatley"),
            help="Second entity for comparison",
            key="entity2_input"
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Load button with loading state
        if st.session_state.loading_state:
            st.markdown('''
            <div style="text-align: center; padding: 10px;">
                <div class="loading-animation"></div>
                <p style="margin-top: 10px; color: #64748b; font-size: 0.9rem;">Loading Analysis...</p>
            </div>
            ''', unsafe_allow_html=True)
        else:
            load_button = st.button("🔄 Load Analysis", type="primary", use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    return location, age, gender, entity1, entity2, st.session_state.loading_state or load_button

def render_analysis_results():
    """Render analysis results with modern styling"""
    if not st.session_state.political_data is not None:
        return
    
    params = st.session_state.params
    
    # Current analysis info
    age_display = f" • {params['age']}" if params['age'] != "All ages" else ""
    gender_display = f" • {params['gender']}" if params['gender'] != "All genders" else ""
    
    st.markdown(f'<div class="success-callout">✅ <strong>Analysis Complete!</strong><br>Location: {params["location"]}{age_display}{gender_display}</div>', unsafe_allow_html=True)
    
    # Political landscape section
    st.markdown("## 🏛️ Political Landscape Analysis")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown('<div class="map-container">', unsafe_allow_html=True)
        st.markdown("### 🗺️ Political Strength Map")
        
        df = st.session_state.political_data
        if isinstance(df, pd.DataFrame) and len(df) > 0:
            map_obj = create_political_legend_map(df, f"political_map_{st.session_state.analysis_id}")
            st.markdown("**Color-coded by political strength levels**")
            
            # Use unique key to prevent reruns with better sizing
            st_folium(map_obj, width=None, height=900, key=f"political_legend_{st.session_state.analysis_id}")
        else:
            st.error("No political data available")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        render_political_stats()

def render_political_stats():
    """Render political statistics with modern styling"""
    if not isinstance(st.session_state.political_data, pd.DataFrame):
        return
    
    df = st.session_state.political_data
    
    # Calculate areas by category  
    strong_conservative = len(df[df['conservative_net_popularity'] >= 5])
    moderate_conservative = len(df[(df['conservative_net_popularity'] >= 2) & (df['conservative_net_popularity'] < 5)])
    lean_conservative = len(df[(df['conservative_net_popularity'] >= 0.5) & (df['conservative_net_popularity'] < 2)])
    competitive = len(df[(df['conservative_net_popularity'] >= -0.5) & (df['conservative_net_popularity'] < 0.5)])
    lean_progressive = len(df[(df['conservative_net_popularity'] >= -2) & (df['conservative_net_popularity'] < -0.5)])
    moderate_progressive = len(df[(df['conservative_net_popularity'] >= -5) & (df['conservative_net_popularity'] < -2)])
    strong_progressive = len(df[df['conservative_net_popularity'] < -5])
    
    total_areas = len(df)
    
    st.markdown('<div class="legend-card">', unsafe_allow_html=True)
    st.markdown("### 📊 Political Breakdown")
    
    # Total areas stat
    st.markdown(f'<div class="stat-card"><div class="stat-number">{total_areas}</div><div class="stat-label">Total Areas</div></div>', unsafe_allow_html=True)
    
    # Conservative stats
    st.markdown("**🔴 Conservative Strength:**")
    st.markdown(f"• Strong: **{strong_conservative}** areas")
    st.markdown(f"• Moderate: **{moderate_conservative}** areas")  
    st.markdown(f"• Lean: **{lean_conservative}** areas")
    
    # Competitive
    st.markdown("**⚖️ Competitive Areas:**")
    st.markdown(f"• Swing: **{competitive}** areas")
    
    # Progressive stats
    st.markdown("**🔵 Progressive Strength:**")
    st.markdown(f"• Lean: **{lean_progressive}** areas")
    st.markdown(f"• Moderate: **{moderate_progressive}** areas")
    st.markdown(f"• Strong: **{strong_progressive}** areas")
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_entity_comparison():
    """Render entity comparison section"""
    if not st.session_state.entity_comparison_data is not None:
        return
    
    params = st.session_state.params
    
    st.markdown("## 👥 Entity Comparison Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="map-container">', unsafe_allow_html=True)
        st.markdown(f"### 🔄 {params['entity1']} vs {params['entity2']}")
        
        df = st.session_state.entity_comparison_data
        if isinstance(df, pd.DataFrame) and len(df) > 0:
            map_obj = create_entity_legend_map(df, params['entity1'], params['entity2'], f"entity_map_{st.session_state.analysis_id}")
            st.markdown("**Blue = Entity 1 stronger, Red = Entity 2 stronger**")
            
            # Use unique key to prevent reruns with better sizing
            st_folium(map_obj, width=None, height=900, key=f"entity_comparison_{st.session_state.analysis_id}")
            
            # Quick stats
            avg_net = df['entity1_net_popularity'].mean()
            entity1_dominant = len(df[df['entity1_net_popularity'] > 0])
            entity2_dominant = len(df[df['entity1_net_popularity'] < 0])
            competitive = len(df[abs(df['entity1_net_popularity']) < 0.5])
            
            st.markdown(f"""
            **📈 Net Score:** {avg_net:.2f}  
            **🔵 {params['entity1']} Dominant:** {entity1_dominant}  
            **🔴 {params['entity2']} Dominant:** {entity2_dominant}  
            **⚖️ Competitive:** {competitive}
            """)
        else:
            st.error("No comparison data available")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="map-container">', unsafe_allow_html=True)
        st.markdown(f"### 🎯 {params['entity1']} Strategy Segments")
        
        df = st.session_state.entity1_data
        if isinstance(df, pd.DataFrame) and len(df) > 0:
            map_obj = create_segment_map(df)
            st.markdown("**Campaign strategy areas**")
            
            # Use unique key to prevent reruns with better sizing
            st_folium(map_obj, width=None, height=900, key=f"entity1_segments_{st.session_state.analysis_id}")
            
            # Strategy segments stats
            if 'segment' in df.columns:
                segment_counts = df['segment'].value_counts()
                st.markdown("**🎯 Strategy Segments:**")
                for segment, count in segment_counts.items():
                    st.markdown(f"• {segment}: **{count}** areas")
            
            # Performance stats
            avg_affinity = df['affinity'].mean() if 'affinity' in df.columns else 0
            avg_popularity = df['popularity'].mean() if 'popularity' in df.columns else 0
            st.markdown(f"**📊 Avg Affinity:** {avg_affinity:.3f}")
            st.markdown(f"**📊 Avg Popularity:** {avg_popularity:.3f}")
        else:
            st.error("No entity 1 data available")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
def render_all_legends():
    """Render all legends in a dedicated section"""
    if not st.session_state.params:
        return
    
    params = st.session_state.params
    
    st.markdown("---")
    st.markdown("## 🎨 Map Legends & Strategy Guide")
    
    col1, col2, col3 = st.columns(3)
    
    # Political Legend
    with col1:
        st.markdown('<div class="legend-card">', unsafe_allow_html=True)
        st.markdown("### 🏛️ Political Strength Legend")
        
        legend_html = """
        <div style='margin: 10px 0;'>
            <div style='display: flex; align-items: center; margin: 8px 0;'>
                <div style='width: 20px; height: 20px; background-color: #8B0000; border-radius: 50%; margin-right: 12px; border: 2px solid white; box-shadow: 0 2px 4px rgba(0,0,0,0.2);'></div>
                <span style='font-weight: 600; color: #1f2937;'>Strong Conservative (5+)</span>
            </div>
            <div style='display: flex; align-items: center; margin: 8px 0;'>
                <div style='width: 20px; height: 20px; background-color: #DC143C; border-radius: 50%; margin-right: 12px; border: 2px solid white; box-shadow: 0 2px 4px rgba(0,0,0,0.2);'></div>
                <span style='font-weight: 600; color: #1f2937;'>Moderate Conservative (2-5)</span>
            </div>
            <div style='display: flex; align-items: center; margin: 8px 0;'>
                <div style='width: 20px; height: 20px; background-color: #FFB6C1; border-radius: 50%; margin-right: 12px; border: 2px solid white; box-shadow: 0 2px 4px rgba(0,0,0,0.2);'></div>
                <span style='font-weight: 600; color: #1f2937;'>Lean Conservative (0.5-2)</span>
            </div>
            <div style='display: flex; align-items: center; margin: 8px 0;'>
                <div style='width: 20px; height: 20px; background-color: #D3D3D3; border-radius: 50%; margin-right: 12px; border: 2px solid white; box-shadow: 0 2px 4px rgba(0,0,0,0.2);'></div>
                <span style='font-weight: 600; color: #1f2937;'>Competitive (-0.5 to 0.5)</span>
            </div>
            <div style='display: flex; align-items: center; margin: 8px 0;'>
                <div style='width: 20px; height: 20px; background-color: #ADD8E6; border-radius: 50%; margin-right: 12px; border: 2px solid white; box-shadow: 0 2px 4px rgba(0,0,0,0.2);'></div>
                <span style='font-weight: 600; color: #1f2937;'>Lean Progressive (-0.5 to -2)</span>
            </div>
            <div style='display: flex; align-items: center; margin: 8px 0;'>
                <div style='width: 20px; height: 20px; background-color: #4169E1; border-radius: 50%; margin-right: 12px; border: 2px solid white; box-shadow: 0 2px 4px rgba(0,0,0,0.2);'></div>
                <span style='font-weight: 600; color: #1f2937;'>Moderate Progressive (-2 to -5)</span>
            </div>
            <div style='display: flex; align-items: center; margin: 8px 0;'>
                <div style='width: 20px; height: 20px; background-color: #000080; border-radius: 50%; margin-right: 12px; border: 2px solid white; box-shadow: 0 2px 4px rgba(0,0,0,0.2);'></div>
                <span style='font-weight: 600; color: #1f2937;'>Strong Progressive (-5+)</span>
            </div>
        </div>
        """
        st.markdown(legend_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Entity Comparison Legend
    with col2:
        st.markdown('<div class="legend-card">', unsafe_allow_html=True)
        st.markdown(f"### 👥 {params['entity1']} vs {params['entity2']} Legend")
        
        entity_legend_html = f"""
        <div style='margin: 10px 0;'>
            <div style='display: flex; align-items: center; margin: 8px 0;'>
                <div style='width: 20px; height: 20px; background-color: #000080; border-radius: 50%; margin-right: 12px; border: 2px solid white; box-shadow: 0 2px 4px rgba(0,0,0,0.2);'></div>
                <span style='font-weight: 600; color: #1f2937;'>Strong {params['entity1']} (5+)</span>
            </div>
            <div style='display: flex; align-items: center; margin: 8px 0;'>
                <div style='width: 20px; height: 20px; background-color: #4169E1; border-radius: 50%; margin-right: 12px; border: 2px solid white; box-shadow: 0 2px 4px rgba(0,0,0,0.2);'></div>
                <span style='font-weight: 600; color: #1f2937;'>Moderate {params['entity1']} (2-5)</span>
            </div>
            <div style='display: flex; align-items: center; margin: 8px 0;'>
                <div style='width: 20px; height: 20px; background-color: #ADD8E6; border-radius: 50%; margin-right: 12px; border: 2px solid white; box-shadow: 0 2px 4px rgba(0,0,0,0.2);'></div>
                <span style='font-weight: 600; color: #1f2937;'>Lean {params['entity1']} (0.5-2)</span>
            </div>
            <div style='display: flex; align-items: center; margin: 8px 0;'>
                <div style='width: 20px; height: 20px; background-color: #D3D3D3; border-radius: 50%; margin-right: 12px; border: 2px solid white; box-shadow: 0 2px 4px rgba(0,0,0,0.2);'></div>
                <span style='font-weight: 600; color: #1f2937;'>Competitive (-0.5 to 0.5)</span>
            </div>
            <div style='display: flex; align-items: center; margin: 8px 0;'>
                <div style='width: 20px; height: 20px; background-color: #FFB6C1; border-radius: 50%; margin-right: 12px; border: 2px solid white; box-shadow: 0 2px 4px rgba(0,0,0,0.2);'></div>
                <span style='font-weight: 600; color: #1f2937;'>Lean {params['entity2']} (-0.5 to -2)</span>
            </div>
            <div style='display: flex; align-items: center; margin: 8px 0;'>
                <div style='width: 20px; height: 20px; background-color: #DC143C; border-radius: 50%; margin-right: 12px; border: 2px solid white; box-shadow: 0 2px 4px rgba(0,0,0,0.2);'></div>
                <span style='font-weight: 600; color: #1f2937;'>Moderate {params['entity2']} (-2 to -5)</span>
            </div>
            <div style='display: flex; align-items: center; margin: 8px 0;'>
                <div style='width: 20px; height: 20px; background-color: #8B0000; border-radius: 50%; margin-right: 12px; border: 2px solid white; box-shadow: 0 2px 4px rgba(0,0,0,0.2);'></div>
                <span style='font-weight: 600; color: #1f2937;'>Strong {params['entity2']} (-5+)</span>
            </div>
        </div>
        """
        st.markdown(entity_legend_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Strategy Segments Legend
    with col3:
        st.markdown('<div class="legend-card">', unsafe_allow_html=True)
        st.markdown("### 🎯 Strategy Segments Legend")
        
        strategy_legend_html = """
        <div style='margin: 10px 0;'>
            <div style='display: flex; align-items: center; margin: 12px 0;'>
                <div style='width: 24px; height: 24px; background-color: #00FF00; border-radius: 50%; margin-right: 15px; border: 2px solid white; box-shadow: 0 2px 4px rgba(0,0,0,0.2);'></div>
                <div>
                    <span style='font-weight: 700; color: #1f2937; font-size: 1.0rem;'>Rally the Base</span><br>
                    <span style='color: #6b7280; font-size: 0.8rem;'>(HA-HP: High Affinity, High Popularity)</span>
                </div>
            </div>
            <div style='display: flex; align-items: center; margin: 12px 0;'>
                <div style='width: 24px; height: 24px; background-color: #0000FF; border-radius: 50%; margin-right: 15px; border: 2px solid white; box-shadow: 0 2px 4px rgba(0,0,0,0.2);'></div>
                <div>
                    <span style='font-weight: 700; color: #1f2937; font-size: 1.0rem;'>Hidden Goldmine</span><br>
                    <span style='color: #6b7280; font-size: 0.8rem;'>(HA-LP: High Affinity, Low Popularity)</span>
                </div>
            </div>
            <div style='display: flex; align-items: center; margin: 12px 0;'>
                <div style='width: 24px; height: 24px; background-color: #FFFF00; border-radius: 50%; margin-right: 15px; border: 2px solid white; box-shadow: 0 2px 4px rgba(0,0,0,0.2);'></div>
                <div>
                    <span style='font-weight: 700; color: #1f2937; font-size: 1.0rem;'>Bring Them Over</span><br>
                    <span style='color: #6b7280; font-size: 0.8rem;'>(LA-HP: Low Affinity, High Popularity)</span>
                </div>
            </div>
            <div style='display: flex; align-items: center; margin: 12px 0;'>
                <div style='width: 24px; height: 24px; background-color: #FF0000; border-radius: 50%; margin-right: 15px; border: 2px solid white; box-shadow: 0 2px 4px rgba(0,0,0,0.2);'></div>
                <div>
                    <span style='font-weight: 700; color: #1f2937; font-size: 1.0rem;'>Deep Conversion</span><br>
                    <span style='color: #6b7280; font-size: 0.8rem;'>(LA-LP: Low Affinity, Low Popularity)</span>
                </div>
            </div>
        </div>
        """
        st.markdown(strategy_legend_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    """Render usage guide when no data is loaded"""
    st.markdown('<div class="info-callout">', unsafe_allow_html=True)
    st.markdown("### 💡 Usage Guide")
    st.markdown("""
    **This dashboard provides:**
    
    1. **🏛️ Political Competition Map** - Shows relative strength of Conservative vs Progressive audiences
    2. **👥 Entity Comparison Map** - Shows which entity dominates in popularity across different areas  
    3. **🎯 Strategy Segments** - Shows campaign strategy areas for the first entity
    
    **Get Started:**
    1. Enter your target location (city, state, or region)
    2. Optionally filter by age group and gender
    3. Enter two entities to compare (politicians, brands, public figures)
    4. Click "Load Analysis" to start
    
    **Example Entities:** Joe Biden, Donald Trump, Tesla, Nike, Taylor Swift, Elon Musk
    """)
    st.markdown('</div>', unsafe_allow_html=True)

def render_usage_guide():
    """Render usage guide when no data is loaded"""
    st.markdown('<div class="info-callout">', unsafe_allow_html=True)
    st.markdown("### 💡 Usage Guide")
    st.markdown("""
    **This dashboard provides:**
    
    1. **🏛️ Political Competition Map** - Shows relative strength of Conservative vs Progressive audiences
    2. **👥 Entity Comparison Map** - Shows which entity dominates in popularity across different areas  
    3. **🎯 Strategy Segments** - Shows campaign strategy areas for the first entity
    
    **Get Started:**
    1. Enter your target location (city, state, or region)
    2. Optionally filter by age group and gender
    3. Enter two entities to compare (politicians, brands, public figures)
    4. Click "Load Analysis" to start
    
    **Example Entities:** Joe Biden, Donald Trump, Tesla, Nike, Taylor Swift, Elon Musk
    """)
    st.markdown('</div>', unsafe_allow_html=True)


def debug_entity_data_loading(entity1_df, entity2_df, entity1_name, entity2_name, age, gender):
    """Debug entity data loading to find demographic bias"""
    
    st.write("### 🔍 Entity Data Debug Info")
    
    # Basic stats
    st.write(f"**{entity1_name} Data:**")
    st.write(f"• Data points: {len(entity1_df)}")
    if len(entity1_df) > 0:
        st.write(f"• Popularity range: {entity1_df['popularity'].min():.3f} - {entity1_df['popularity'].max():.3f}")
        st.write(f"• Average popularity: {entity1_df['popularity'].mean():.3f}")
        st.write(f"• Affinity range: {entity1_df['affinity'].min():.3f} - {entity1_df['affinity'].max():.3f}")
    
    st.write(f"**{entity2_name} Data:**")
    st.write(f"• Data points: {len(entity2_df)}")
    if len(entity2_df) > 0:
        st.write(f"• Popularity range: {entity2_df['popularity'].min():.3f} - {entity2_df['popularity'].max():.3f}")
        st.write(f"• Average popularity: {entity2_df['popularity'].mean():.3f}")
        st.write(f"• Affinity range: {entity2_df['affinity'].min():.3f} - {entity2_df['affinity'].max():.3f}")
    
    # Check demographic impact
    demo_filters = []
    if age: demo_filters.append(f"Age: {age}")
    if gender: demo_filters.append(f"Gender: {gender}")
    
    if demo_filters:
        st.write(f"**Applied Filters:** {', '.join(demo_filters)}")
    else:
        st.write("**No demographic filters applied**")
    
    # Data ratio analysis
    if len(entity1_df) > 0 and len(entity2_df) > 0:
        ratio = len(entity1_df) / len(entity2_df)
        pop_ratio = entity1_df['popularity'].mean() / entity2_df['popularity'].mean()
        
        st.write(f"**Data Point Ratio:** {ratio:.2f}:1 ({entity1_name}:{entity2_name})")
        st.write(f"**Popularity Ratio:** {pop_ratio:.2f}:1")
        
        if ratio > 3 or ratio < 0.33:
            st.error(f"🚨 **MAJOR DATA IMBALANCE**: {ratio:.1f}:1 ratio detected!")
            st.write("This will cause biased comparison results.")
        
        if pop_ratio > 2 or pop_ratio < 0.5:
            st.warning(f"⚠️ **POPULARITY IMBALANCE**: {pop_ratio:.1f}:1 popularity ratio")
    
    # Show sample data
    if len(entity1_df) > 0:
        st.write(f"**{entity1_name} Sample Data:**")
        st.dataframe(entity1_df[['latitude', 'longitude', 'affinity', 'popularity']].head(3))
    
    if len(entity2_df) > 0:
        st.write(f"**{entity2_name} Sample Data:**")
        st.dataframe(entity2_df[['latitude', 'longitude', 'affinity', 'popularity']].head(3))


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application with improved UX and no rerunning issues"""
    
    # Page config
    st.set_page_config(
        page_title="Political Campaign Analyzer", 
        page_icon="🗳️", 
        layout="wide"
    )
    
    # Apply modern styling
    st.markdown(style_component(), unsafe_allow_html=True)
    
    # Initialize session state
    initialize_session_state()
    
    # Navigation
    render_navigation()
    
    # Parameter panel
    location, age, gender, entity1, entity2, should_load = render_parameter_panel()
    
    # Load data only when button is clicked
    if should_load and not st.session_state.loading_state:
        if not location.strip():
            st.error("❌ Please enter a location")
            return
        
        if not entity1.strip() or not entity2.strip():
            st.error("❌ Please enter both entities for comparison")
            return
        
        # Set loading state
        st.session_state.loading_state = True
        
        # Store parameters
        st.session_state.last_location = location
        st.session_state.last_entity1 = entity1
        st.session_state.last_entity2 = entity2
        
        # Generate unique analysis ID to prevent map rerun issues
        st.session_state.analysis_id = f"analysis_{uuid.uuid4().hex[:8]}"
        
        # Show loading state
        with st.spinner("🔄 Loading political intelligence data..."):
            try:
                # Prepare common parameters
                common_params = {"location_query": location.strip()}
                if age: 
                    common_params["age"] = age
                if gender: 
                    common_params["gender"] = gender
                
                # Load political audience data
                progressive_df = get_complete_heatmap_analysis(
                    audience_ids=['urn:audience:political_preferences:politically_progressive'],
                    **common_params
                )
                progressive_df = progressive_df[progressive_df['popularity'] > 0.5]
                
                conservative_df = get_complete_heatmap_analysis(
                    audience_ids=['urn:audience:political_preferences:politically_conservative'],
                    **common_params
                )
                conservative_df = conservative_df[conservative_df['popularity'] > 0.5]

                # Process political competition data
                political_competition_df = process_political_data(conservative_df, progressive_df)
                
                # Load entity data
                entity1_df = get_complete_heatmap_analysis(
                    entity_names=[entity1.strip()],
                    **common_params
                )
                entity2_df = get_complete_heatmap_analysis(
                    entity_names=[entity2.strip()],
                    **common_params
                )
                # debug_entity_data_loading(entity1_df, entity2_df, entity1, entity2, age, gender)
                # Process entity comparison data
                entity_comparison_df = process_entity_data(entity1_df, entity2_df)
                
                # Store all data in session state
                st.session_state.political_data = political_competition_df
                st.session_state.entity_comparison_data = entity_comparison_df
                st.session_state.entity1_data = entity1_df
                st.session_state.params = {
                    'location': location,
                    'entity1': entity1,
                    'entity2': entity2,
                    'age': age if age else "All ages",
                    'gender': gender if gender else "All genders"
                }
                
                # Clear loading state
                st.session_state.loading_state = False
                
                st.success("✅ Analysis completed successfully!")
                time.sleep(1)
                st.rerun()
                
            except Exception as e:
                st.session_state.loading_state = False
                st.error(f"❌ Error: {str(e)}")
    
    # Display results if data is loaded
    if st.session_state.political_data is not None:
        render_analysis_results()
        render_entity_comparison()
        render_all_legends()
    else:
        render_usage_guide()

if __name__ == "__main__":
    main()