# 🌿 Spice Market Scout

> **AI-Powered Market Intelligence & Forecasting for Sri Lankan Spices**

Spice Market Scout is a cutting-edge analytics platform designed to provide accurate price forecasts and real-time market intelligence for the Sri Lankan spice industry. Focusing on key commodities like **Cinnamon** and **Pepper**, the tool leverages advanced machine learning models and generative AI to empower stakeholders with data-driven insights.

---

## 🚀 Key Features

-   **📈 AI Price Forecasting**: Multi-step forecasting (up to 6 months) for various spice grades and regions using deep learning models.
-   **🔄 Regional Comparison**: Compare price trends and forecasts across major producing regions (e.g., Colombo, Galle, Matara, Ratnapura).
-   **📰 Market Intelligence Agent**: Integrated AI news agent that scrapes the web for the latest market news and uses **Google Gemini** to analyze sentiment (Bullish/Bearish) and generate summaries.
-   **🌶️ Multi-Commodity Support**: Specialized analysis for Cinnamon (Alba, C5, H1, etc.) and Pepper (Garde 1, Light, etc.).
-   **💻 Modern Interactive Dashboard**: A premium, dark-mode data visualization interface built with React and Tailwind CSS.

---

## 🛠️ Tech Stack

### Frontend
-   **Framework**: [React](https://react.dev/) (Vite)
-   **Styling**: [Tailwind CSS](https://tailwindcss.com/) (v4), PostCSS
-   **Visualization**: [Recharts](https://recharts.org/)
-   **Icons**: [Lucide React](https://lucide.dev/)

### Backend & AI
-   **API**: [FastAPI](https://fastapi.tiangolo.com/)
-   **Machine Learning**: TensorFlow/Keras, Scikit-learn, Pandas, NumPy
-   **Generative AI**: Google Gemini (via `google-generativeai`)
-   **Data Processing**: Pandas, Beautiful Soup (for scraping)

---

## 🏁 Getting Started

### Prerequisites
-   Python 3.9+
-   Node.js 16+
-   Google Gemini API Key (for News Agent)

### 1. Clone the Repository
```bash
git clone https://github.com/vimesh630/spice-market-scout.git
cd spice-market-scout
```

### 2. Backend Setup
Create a virtual environment and install dependencies:

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Configuration**:
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=your_google_gemini_api_key_here
```

### 3. Frontend Setup
Navigate to the frontend directory and install dependencies:

```bash
cd frontend
npm install
```

---

## 🏃‍♂️ Usage

### Running the API Server
Start the FastAPI backend:
```bash
# From the root directory
python src/api.py
# OR
uvicorn src.api:app --reload
```
The API will be available at `http://localhost:8000`. Documentation at `http://localhost:8000/docs`.

### Running the Dashboard
Start the React development server:
```bash
# From the frontend directory
npm run dev
```
Open `http://localhost:5173` in your browser to view the application.

---

## 📂 Project Structure

```
spice-market-scout/
├── data/                   # Raw and processed data storage
│   ├── raw/
│   └── processed/
├── frontend/               # React application
│   ├── src/
│   └── public/
├── models/                 # Trained predictive models
├── notebooks/              # Jupyter notebooks for EDA and experimentation
├── src/                    # Backend source code
│   ├── api.py              # FastAPI application entry point
│   ├── data_pipeline.py    # ETL pipeline logic
│   ├── forecasting_engine.py # ML model training and inference
│   ├── news_agent.py       # Gemini-powered news analysis
│   └── ingest_data.py      # Data ingestion scripts
├── config.py               # Global configuration
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---
## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
