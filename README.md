# ResonanceAI 🎯

**AI-Powered Political Campaign Intelligence Platform**

ResonanceAI is a comprehensive political campaign platform built with Google's Agent Development Kit (ADK) that combines political intelligence, cultural insights, and AI-powered content generation to help campaigns create authentic, resonant messaging.

## 🚀 What is ResonanceAI?

ResonanceAI is an ADK-based agent system that provides:

- **Political Analysis**: Geographic targeting, popularity tracking, and strategic location filtering
- **Cultural Intelligence**: Deep demographic and cultural insights via Qloo API
- **Campaign Content Generation**: AI-generated copy, images, and complete campaign packages
- **Multi-Platform Optimization**: Content optimized for social media, email, direct mail, and SMS

## 🏗️ Architecture

Built using Google's ADK (Agent Development Kit), ResonanceAI consists of specialized agents:

- **Political Agent**: Core campaign analysis and strategic planning
- **Content Agent**: Cultural intelligence and content generation
- **Merch Agent**: Additional campaign asset creation

## 📋 Prerequisites

- Python 3.8+
- Google Cloud Platform account
- Qloo API access
- Google AI Studio API key or Vertex AI setup

## 🔧 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/hazardscarn/ResonanceAI.git
cd ResonanceAI
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# macOS/Linux:
source .venv/bin/activate

# Windows CMD:
.venv\Scripts\activate.bat

# Windows PowerShell:
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the project root:

```bash
# Google AI/Vertex AI Configuration
GOOGLE_API_KEY=your_google_ai_studio_key
GOOGLE_GENAI_USE_VERTEXAI=FALSE
GOOGLE_CLOUD_PROJECT=your_project_id
GOOGLE_CLOUD_LOCATION=us-central1

# Qloo API
QLOO_API_KEY=your_qloo_api_key

# Supabase (if using)
REACT_APP_SUPABASE_URL=your_supabase_url
SUPABASE_SECRET_KEY=your_supabase_secret_key
```

For Vertex AI instead of API key:
```bash
GOOGLE_GENAI_USE_VERTEXAI=TRUE
GOOGLE_CLOUD_PROJECT=your_project_id
GOOGLE_CLOUD_LOCATION=us-central1
```

### 5. Authentication Setup

For Google Cloud (if using Vertex AI):
```bash
gcloud auth login
gcloud config set project your_project_id
gcloud auth application-default login
```

## 🚀 Usage

### Running with ADK Web Interface

```bash
# From the project root directory
adk web
```

Then navigate to `http://localhost:8080` to interact with the agent.

### Running with FastAPI

```bash
# Using the main.py server
python resonance_agent/main.py
```

### Running with Docker

```bash
# Build the container
docker build -t resonanceai .

# Run the container
docker run -p 8080:8080 --env-file .env resonanceai
```

## 📊 Data Requirements

The platform expects campaign data in CSV format with these columns:

```csv
latitude,longitude,affinity,popularity,strategy,segment,popularity_status,base_popularity_status,tag_gun_violence,tag_politics
```

**Key Columns:**
- `latitude`, `longitude`: Geographic coordinates
- `affinity`: Voter affinity score (0-1)  
- `popularity`: Candidate popularity (0-1)
- `strategy`: "Rally the Base", "Hidden Goldmine", "Bring Them Over", "Deep Conversion"
- `popularity_status`: "Trailing Opponent", "Leading Opponent", "Similar Popularity"
- `base_popularity_status`: "Less Popular than Party", "More Popular than Party", "Similar Popularity to Party"

## 🎯 Core Features

### Political Intelligence
- Geographic targeting and analysis
- Candidate vs opponent popularity tracking
- Strategic location filtering
- Issue-based sentiment analysis

### Cultural Intelligence  
- Demographic-based cultural insights
- Brand and entertainment preferences
- Music and interest analysis
- Authentic messaging recommendations

### Content Generation
- Platform-optimized campaign content
- AI-generated images with Vertex AI Imagen
- Multiple copy variants (short, medium, long)
- Complete campaign packages

## 🔨 Project Structure

```
ResonanceAI/
├── resonance_agent/
│   ├── political_agent/          # Main political analysis agent
│   │   ├── agent.py             # Root agent definition
│   │   ├── tools.py             # Political analysis tools
│   │   ├── content_tools.py     # Cultural intelligence tools
│   │   └── campaign_content_tools.py # Content generation tools
│   ├── src/                     # Core utilities
│   │   ├── qloo.py             # Qloo API integration
│   │   ├── heatmap.py          # Geographic analysis
│   │   └── secret_manager.py    # Credential management
│   └── main.py                  # FastAPI server
├── merchagent/                  # Additional agent for campaign assets
├── contentagent/               # Content-focused agent
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Container setup
└── .env.example               # Environment template
```

## 🌐 Deployment

### Deploy to Google Cloud Run

```bash
# Build and deploy
gcloud run deploy resonanceai \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### Deploy to Agent Engine

```bash
# Deploy using the official ADK method
python deploy_resonance_to_agent_engine.py
```

## 🛠️ Configuration

### Model Configuration

Edit `resonance_agent/political_agent/config.py`:

```python
class Modelconfig:
    flash_lite_model = "gemini-2.0-flash-lite-001"
    pro_model = "gemini-2.5-pro"
    flash_model = "gemini-2.5-flash"
    temperature = 0.1
    max_tokens = 8192
```

### Google Cloud Settings

```python
class Settings:
    GOOGLE_CLOUD_PROJECT = "your-project-id"
    GOOGLE_CLOUD_LOCATION = "us-central1"
    GCS_BUCKET_NAME = "your-bucket-name"
```

## 📚 API Integrations

- **Google ADK**: Agent framework and orchestration
- **Qloo API**: Cultural intelligence and demographic insights
- **Vertex AI**: LLM models and image generation  
- **Google Cloud Storage**: Asset storage and management
- **Supabase**: Optional database integration

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/hazardscarn/ResonanceAI/issues)
- **Documentation**: Check the ADK documentation at [google.github.io/adk-docs](https://google.github.io/adk-docs)

## 🙏 Acknowledgments

- **Google ADK Team** for the agent development framework
- **Qloo** for cultural intelligence API
- **Google Cloud** for AI and infrastructure services

---

**ResonanceAI** - Intelligent Political Campaign Analysis & Content Generation 🎯✨