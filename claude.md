# AdDownloader Codebase Overview

## COMPREHENSIVE ADDOWNLOADER CODEBASE OVERVIEW

### 1. PROJECT STRUCTURE

**Root Directory Layout:**
```
/home/user/AdDownloader/
├── AdDownloader/              # Main package directory
│   ├── __init__.py
│   ├── __main__.py
│   ├── api/                   # FastAPI service module
│   │   ├── __init__.py
│   │   └── service.py         # FastAPI application and endpoints
│   ├── adlib_api.py           # Meta Ad Library API client
│   ├── analysis.py            # Text and image analysis functions
│   ├── app.py                 # Dash analytics dashboard
│   ├── cli.py                 # Command-line interface
│   ├── helpers.py             # Utility functions and validators
│   ├── media_download.py      # Media content download functionality
│   └── start_app.py           # Dashboard launcher
├── data/                       # Sample data folder
│   ├── McD_be.xlsx            # McDonald's Belgium ads data example
│   └── us_parties.xlsx        # US political parties data example
├── tests/                      # Test suite
│   ├── test_AdDownloader.py   # Performance tests
│   ├── us_elections_analysis.py # Example analysis script
│   └── regression_estimates.r  # R statistical analysis
├── docs/                       # Documentation (Sphinx)
│   ├── source/
│   │   ├── conf.py
│   │   ├── index.rst
│   │   ├── adlib_api.rst
│   │   ├── media_download.rst
│   │   ├── analysis.rst
│   │   ├── cli.rst
│   │   ├── helpers.rst
│   │   ├── api_download_tasks.md
│   └── html/                   # Built HTML documentation
├── output/                     # Output directory for downloaded data
├── pyproject.toml              # Project configuration and dependencies
├── .readthedocs.yaml           # ReadTheDocs build configuration
├── README.md                   # Main documentation
├── LICENSE.txt                 # GNU GPLv3 license
├── CODE_OF_CONDUCT.md          # Community guidelines
└── example.py                  # Usage examples
```

---

### 2. MAIN COMPONENTS AND MODULES

| Module | Lines | Purpose |
|--------|-------|---------|
| **analysis.py** | 780 | Text/image analysis, sentiment analysis, topic modeling, image feature extraction, BLIP image captioning/QA |
| **app.py** | 757 | Interactive Dash dashboard for visualization and analytics |
| **adlib_api.py** | 305 | Client for Meta Ad Library API, data fetching and processing |
| **media_download.py** | 313 | Media content downloading (images/videos), Selenium web scraping |
| **helpers.py** | 479 | Input validators, logging, utility functions, data conversion |
| **cli.py** | 242 | Interactive command-line interface with user prompts |
| **api/service.py** | 190 | FastAPI REST API endpoints for programmatic access |
| **Other files** | 26 | Entry points and app starters |
| **TOTAL** | 3,092 | Complete project codebase |

---

### 3. PROGRAMMING LANGUAGES USED

- **Python 3.9, 3.10, 3.11** (primary language)
  - 15 Python files total
  - 3,092 lines of production code
- **RST (reStructuredText)** - Sphinx documentation
- **Markdown** - README and documentation files
- **R** - Statistical analysis (regression_estimates.r)
- **YAML** - Configuration files (.readthedocs.yaml)
- **TOML** - Project configuration (pyproject.toml)

---

### 4. MAIN ENTRY POINTS

**As CLI:**
```bash
python -m AdDownloader.cli          # Interactive CLI mode
python -m AdDownloader              # Via __main__.py entry point
AdDownloader                        # Direct command (installed package)
```

**As Python Package:**
```python
from AdDownloader import adlib_api
from AdDownloader.media_download import start_media_download
from AdDownloader.app import app
```

**As FastAPI Service:**
```bash
uvicorn AdDownloader.api.service:app --reload
# POST /download endpoint
```

**As Dashboard:**
```bash
python -m AdDownloader.app
# Access: http://127.0.0.1:8050/
```

---

### 5. KEY LIBRARIES AND DEPENDENCIES

**Core Framework Dependencies:**
- **fastapi==0.110.0** - REST API framework
- **uvicorn[standard]==0.29.0** - ASGI server
- **typer==0.9.0** - CLI framework
- **click==8.1.3** - CLI utilities
- **inquirer3==0.4.0** - Interactive prompts
- **rich==13.6.0** - Rich terminal formatting

**Data Processing:**
- **pandas==2.0.3** - Data analysis and Excel I/O
- **openpyxl==3.1.2** - Excel file handling
- **numpy==1.26.4** - Numerical operations

**Text Analysis:**
- **nltk==3.8.1** - Natural language processing (tokenization, sentiment)
- **gensim==4.3.2** - Topic modeling (LDA)
- **textblob==0.17.1** - Sentiment analysis
- **scikit-learn==1.4.1.post1** - Machine learning utilities

**Deep Learning & Vision:**
- **torch==2.2.0** - PyTorch deep learning framework
- **transformers==4.37.2** - BLIP models (image captioning, Q&A)
- **pillow==11.0.0** - Image processing
- **opencv-python==4.9.0.80** - Computer vision
- **scikit-image==0.22.0** - Image processing algorithms

**Web Scraping & HTTP:**
- **selenium==4.16.0** - Browser automation (Chrome WebDriver)
- **requests==2.31.0** - HTTP client library

**Visualization & Dashboards:**
- **dash==2.15.0** - Interactive dashboards
- **plotly** - Interactive charts (via dash)

**Testing:**
- **pytest>=6.2.4** - Testing framework

**Build & Development:**
- **setuptools>=43.0.0** - Package building
- **meson==1.3.1** - Build system
- **ninja==1.11.1.1** - Build backend

---

### 6. PURPOSE OF MAJOR DIRECTORIES

| Directory | Purpose |
|-----------|---------|
| **AdDownloader/** | Main package with all modules |
| **AdDownloader/api/** | FastAPI service for REST API access |
| **data/** | Sample input files (Excel sheets with page IDs) |
| **output/** | Generated output folder for downloaded ads and media |
| **docs/** | Sphinx documentation source and built HTML |
| **tests/** | Test suite including performance benchmarks |
| **.vscode/** | VS Code workspace settings |
| **dist/** | Distribution packages (.whl, .tar.gz) |

---

### 7. CONFIGURATION FILES

| File | Purpose |
|------|---------|
| **pyproject.toml** | Main project configuration, dependencies, build settings, entry points |
| **.readthedocs.yaml** | ReadTheDocs build configuration (Python 3.11, Sphinx, PDF/EPUB output) |
| **docs/source/conf.py** | Sphinx documentation configuration |
| **.vscode/settings.json** | VS Code editor settings |

**Key Configuration Details from pyproject.toml:**
- Version: 0.2.11
- Python: >= 3.9
- License: GNU GPLv3
- CLI entry point: `AdDownloader.cli:main`
- 28 dependencies specified
- Classifiers: Development Status Alpha, multiple Python versions supported

---

### 8. TESTS AND DOCUMENTATION

**Test Files:**
- `/home/user/AdDownloader/tests/test_AdDownloader.py` - Performance benchmarking tests (measures text/topic analysis execution time across different dataset sizes)
- `/home/user/AdDownloader/tests/us_elections_analysis.py` - Example analysis workflow for US election data
- `/home/user/AdDownloader/tests/regression_estimates.r` - Statistical analysis in R

**Documentation Files:**
- `/home/user/AdDownloader/README.md` - Comprehensive getting started guide (14KB)
- `/home/user/AdDownloader/docs/source/` - Full API documentation in RST format
- `/home/user/AdDownloader/CODE_OF_CONDUCT.md` - Community guidelines
- `/home/user/AdDownloader/example.py` - Complete usage examples for all features
- ReadTheDocs integration for online hosting

---

## PROJECT PURPOSE & ARCHITECTURE SUMMARY

### Core Purpose
**AdDownloader** is a comprehensive, open-source Python tool for downloading advertisement data and media content from Meta's Ad Library API for research purposes. It provides both programmatic access (Python package) and user-friendly CLI/Dashboard interfaces.

### Main Workflows

**1. Ad Data Download Workflow:**
- Initialize AdLibAPI client with Meta access token
- Configure search parameters (countries, dates, search terms, page IDs, ad type)
- Fetch ad metadata from Meta Ad Library API (v20.0)
- Save raw JSON responses
- Process and convert to Excel/CSV formats
- Organize in project-specific output folders

**2. Media Download Workflow:**
- Access `ad_snapshot_url` from downloaded ad data
- Use Selenium WebDriver to navigate ad pages
- Download media assets (images as PNG, videos as MP4)
- Extract frames from videos at configurable intervals
- Handle cookies and dynamic page loading

**3. Analysis & Visualization Workflow:**
- Load processed ad data from Excel
- **Text Analysis:** Tokenization, lemmatization, sentiment analysis (VADER & TextBlob)
- **Topic Modeling:** LDA using Gensim with coherence/perplexity metrics
- **Image Analysis:** Dominant color extraction, quality assessment (resolution, brightness, contrast, sharpness)
- **Deep Learning:** BLIP-based image captioning and visual question answering
- **Dashboard:** Interactive Dash application with charts and graphs

### Three Usage Modes

1. **CLI Mode** - Interactive command-line interface (inquirer3 prompts)
2. **Python API** - Direct package import and method calls
3. **REST API** - FastAPI service with POST /download endpoint for programmatic/remote access

### Key Features
- Support for multiple ad types (ALL, POLITICAL_AND_ISSUE_ADS)
- Multi-country support (250+ country codes)
- Flexible search (by keywords or page IDs)
- Automated media scraping using Selenium
- Advanced analytics: sentiment, topic modeling, image features
- AI-powered image captioning and Q&A (BLIP models)
- Interactive dashboard with live data visualization
- Comprehensive logging and error handling
- Input validation for dates, countries, files

### Output Structure
```
output/<project_name>/
├── json/                          # Raw API responses
├── ads_data/
│   └── <project_name>_processed_data.xlsx  # Cleaned data
├── ads_images/                    # Downloaded images (PNG)
├── ads_videos/                    # Downloaded videos (MP4)
├── ads_video_frames/              # Extracted video frames
└── logs/                          # Execution logs
```

---

## Technology Stack Summary

**Frontend/UI:** Dash (interactive dashboards), Rich (terminal formatting)
**Backend:** FastAPI (REST API), Typer (CLI), Custom Python modules
**Data:** Pandas, NumPy, Excel I/O
**AI/ML:** Transformers (BLIP), Scikit-learn, Gensim, NLTK, TextBlob, PyTorch
**Vision:** OpenCV, PIL, Scikit-image
**Web:** Requests, Selenium (browser automation)
**Documentation:** Sphinx, ReadTheDocs
**Testing:** Pytest
**Deployment:** ASGI (Uvicorn)

The project is well-structured, professionally maintained, comprehensively documented, and designed for both researchers and developers with Python expertise levels ranging from beginner (CLI) to advanced (custom API integration).

---

## AdDownloader Summary

**AdDownloader** is a Python-based tool for downloading and analyzing advertisement data from Meta's Ad Library API, with support for automated media scraping, text analysis, and interactive visualization.

### Core Architecture

The project has **3,092 lines** of code organized around 3 main usage modes:

1. **CLI Mode** (`cli.py`) - Interactive command-line interface for users
2. **Python API** - Direct package import for developers
3. **FastAPI Service** (`api/service.py`) - REST API endpoints for programmatic access

### Main Components

| Component | Purpose |
|-----------|---------|
| **adlib_api.py** | Fetches ad data from Meta Ad Library API with filtering options |
| **media_download.py** | Uses Selenium to scrape and download images/videos from ads |
| **analysis.py** | Text analysis (sentiment, topic modeling) and image analysis (BLIP captioning, color extraction) |
| **app.py** | Dash dashboard for interactive visualization and analytics |
| **cli.py** | Interactive CLI with user prompts and workflows |
| **helpers.py** | Validators, logging, and utility functions |

### Key Technologies

**Data Processing:** Pandas, NumPy, Excel I/O
**AI/ML:** Transformers (BLIP), Scikit-learn, Gensim, NLTK, TextBlob
**Vision:** OpenCV, PIL, Scikit-image
**Web:** Selenium (browser automation), Requests
**Framework:** FastAPI, Typer, Dash
**Testing:** Pytest, Sphinx documentation

### Typical Workflow

1. **Download** → Fetch ads from Meta API with custom filters (countries, dates, keywords)
2. **Scrape** → Extract media assets using Selenium browser automation
3. **Analyze** → Apply NLP and computer vision analysis (sentiment, topics, image captions)
4. **Visualize** → Explore results via interactive Dash dashboard

### Output Structure

Downloaded data is organized per project:
```
output/<project_name>/
├── json/                   # Raw API responses
├── ads_data/              # Processed Excel files
├── ads_images/            # Downloaded images
├── ads_videos/            # Downloaded videos
└── ads_video_frames/      # Extracted frames
```

The codebase is well-documented with Sphinx docs, comprehensive logging, error handling, and input validation throughout.
