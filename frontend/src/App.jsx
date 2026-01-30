import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { LayoutDashboard, TrendingUp, RefreshCw, AlertCircle, Calendar } from 'lucide-react';
import './App.css';
import IntelligenceCard from './components/IntelligenceCard';


// Configure Axios base URL
const API_URL = 'http://127.0.0.1:8000';


function App() {
  const [regions, setRegions] = useState([]);
  const [grades, setGrades] = useState([]);
  const [allGradesCombinations, setAllGradesCombinations] = useState({});

  const [selectedRegion, setSelectedRegion] = useState('');
  const [selectedGrade, setSelectedGrade] = useState('');
  const [forecastDate, setForecastDate] = useState(6); // months

  const [forecastData, setForecastData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [trainingStatus, setTrainingStatus] = useState(null);

  // Fetch Metadata on Load
  useEffect(() => {
    const fetchMetadata = async () => {
      try {
        const response = await axios.get(`${API_URL}/metadata`);
        setRegions(response.data.regions || []);
        setGrades(response.data.grades || []);

        // Save the map for dependent filtering
        if (response.data.grades_by_region) {
          setAllGradesCombinations(response.data.grades_by_region);
        }

        // Set defaults
        if (response.data.regions.length > 0) {
          const initialRegion = response.data.regions[0];
          setSelectedRegion(initialRegion);

          // Set initial grade consistent with region
          if (response.data.grades_by_region && response.data.grades_by_region[initialRegion]) {
            setSelectedGrade(response.data.grades_by_region[initialRegion][0]);
          } else if (response.data.grades.length > 0) {
            setSelectedGrade(response.data.grades[0]);
          }
        }
      } catch (err) {
        console.error("Failed to fetch metadata:", err);
        setError("Could not connect to backend. Is uvicorn running?");
      }
    };
    fetchMetadata();
  }, []);

  // Handle Region Change causing Grade Change
  const handleRegionChange = (e) => {
    const newRegion = e.target.value;
    setSelectedRegion(newRegion);

    // Filter grades for this new region
    if (allGradesCombinations[newRegion]) {
      const available = allGradesCombinations[newRegion];
      // If current grade is not in available, switch to first available
      if (!available.includes(selectedGrade)) {
        setSelectedGrade(available[0] || '');
      }
    }
  };

  // Determine current available grades
  const currentGrades = allGradesCombinations[selectedRegion] || grades;

  const handleForecast = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await axios.post(`${API_URL}/predict`, {
        region: selectedRegion,
        grade: selectedGrade,
        months: forecastDate
      });

      const { dates: fDates, prices: fPrices } = response.data.forecast;
      const { dates: hDates, prices: hPrices } = response.data.history || { dates: [], prices: [] };

      // Combine for Recharts
      // History data: { name: date, History: price, Forecast: null }
      const historyData = hDates.map((date, index) => ({
        name: date,
        History: hPrices[index],
        Forecast: null
      }));

      // Connect the last history point to the first forecast point visually
      if (hDates.length > 0 && fDates.length > 0) {
        historyData[historyData.length - 1].Forecast = historyData[historyData.length - 1].History;
      }

      // Forecast data: { name: date, History: null, Forecast: price }
      const forecastChartData = fDates.map((date, index) => ({
        name: date,
        History: null,
        Forecast: fPrices[index]
      }));

      const fullChartData = [...historyData, ...forecastChartData];

      setForecastData(fullChartData);
    } catch (err) {
      console.error("Forecast failed:", err);
      setError("Failed to generate forecast. " + (err.response?.data?.detail || err.message));
    } finally {
      setLoading(false);
    }
  };

  const handleRetrain = async () => {
    setTrainingStatus("Training started...");
    try {
      await axios.post(`${API_URL}/retrain`, { epochs: 10 });
      setTrainingStatus("Training started in background! Check API terminal.");
      setTimeout(() => setTrainingStatus(null), 5000);
    } catch (err) {
      setTrainingStatus("Retrain failed.");
    }
  };

  return (
    <div className="app-container">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="logo">
          <TrendingUp className="icon" /> <span>Spice Scout</span>
        </div>

        <div className="controls">
          <div className="control-group">
            <label>Region</label>
            <select value={selectedRegion} onChange={handleRegionChange}>
              {regions.map(r => <option key={r} value={r}>{r}</option>)}
            </select>
          </div>

          <div className="control-group">
            <label>Grade</label>
            <select value={selectedGrade} onChange={(e) => setSelectedGrade(e.target.value)}>
              {currentGrades.map(g => <option key={g} value={g}>{g}</option>)}
            </select>
          </div>

          <button className="primary-btn" onClick={handleForecast} disabled={loading || !selectedRegion || !selectedGrade}>
            {loading ? 'Forecasting...' : 'Generate Forecast'}
          </button>


          <div className="divider"></div>

          <button className="secondary-btn" onClick={handleRetrain}>
            <RefreshCw size={16} style={{ marginRight: '8px' }} /> Retrain Model
          </button>
          {trainingStatus && <div className="status-msg">{trainingStatus}</div>}
        </div>
      </aside>

      {/* Main Content */}
      <main className="main-content">
        <header>
          <h1>Market Dashboard</h1>
          <div className="date-picker">
            <Calendar size={18} /> <span>{new Date().toLocaleDateString()}</span>
          </div>
        </header>

        {error && <div className="error-banner"><AlertCircle size={18} /> {error}</div>}

        <IntelligenceCard />

        <div className="dashboard-grid">

          {/* Metric Cards */}
          <div className="card metric">
            <h3>Selected Grade</h3>
            <p className="value">{selectedGrade || '-'}</p>
          </div>
          <div className="card metric">
            <h3>Latest Price</h3>
            {/* Use the last available price (forecast or history) */}
            <p className="value">
              {forecastData ?
                `LKR ${(forecastData.findLast(d => d.Forecast !== null)?.Forecast || forecastData.findLast(d => d.History !== null)?.History)}`
                : '-'}
            </p>
            <span className="subtext">Estimated current</span>
          </div>
          <div className="card metric">
            <h3>Next 6 Months</h3>
            <p className="value" style={{ color: '#4caf50' }}>
              {/* Calc trend based on forecast start vs end */}
              {forecastData ?
                (() => {
                  const forecastPoints = forecastData.filter(d => d.Forecast !== null);
                  if (forecastPoints.length < 2) return '-';
                  const start = forecastPoints[0].Forecast;
                  const end = forecastPoints[forecastPoints.length - 1].Forecast;
                  return (end - start) > 0 ? 'Trending Up' : 'Trending Down';
                })()
                : '-'}
            </p>
          </div>
        </div>

        <div className="card chart-card">
          <h2>Price Forecast</h2>
          <div className="chart-container">
            {forecastData ? (
              <ResponsiveContainer width="100%" height={400}>
                <LineChart data={forecastData}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#eee" />
                  <XAxis dataKey="name" stroke="#888" tick={{ fontSize: 12 }} interval="preserveStartEnd" minTickGap={50} />
                  <YAxis stroke="#888" domain={['auto', 'auto']} label={{ value: 'Price (LKR/kg)', angle: -90, position: 'insideLeft', style: { textAnchor: 'middle', fill: '#888' } }} />
                  <Tooltip
                    formatter={(value) => [`LKR ${value}`, 'Price']}
                    contentStyle={{ backgroundColor: '#fff', borderRadius: '8px', border: 'none', boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }}
                  />

                  <Legend />
                  <Line type="monotone" dataKey="History" stroke="#3b82f6" strokeWidth={2} dot={false} activeDot={{ r: 6 }} name="Historical" />
                  <Line type="monotone" dataKey="Forecast" stroke="#FF4B4B" strokeWidth={3} strokeDasharray="5 5" dot={{ r: 4 }} activeDot={{ r: 6 }} name="Forecast" />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="empty-state">
                <LayoutDashboard size={48} color="#ccc" />
                <p>Select parameters and click Generate Forecast</p>
              </div>
            )}
          </div>
        </div>

      </main>

      <style>{`
        :root {
          --sidebar-bg: #1e1e2d;
          --main-bg: #f5f6fa;
          --card-bg: #ffffff;
          --primary: #FF4B4B;
          --text-main: #2b2b2b;
          --text-muted: #888;
        }
        
        * { box-sizing: border-box; }
        body { margin: 0; font-family: 'Inter', sans-serif; background: var(--main-bg); color: var(--text-main); }
        
        .app-container { display: flex; height: 100vh; }
        
        .sidebar { width: 260px; background: var(--sidebar-bg); color: white; padding: 24px; display: flex; flex-direction: column; }
        .logo { font-size: 1.5rem; font-weight: bold; display: flex; align-items: center; gap: 10px; margin-bottom: 40px; color: var(--primary); }
        
        .control-group { margin-bottom: 20px; }
        .control-group label { display: block; margin-bottom: 8px; font-size: 0.9rem; color: #a6a6b0; }
        .control-group select { width: 100%; padding: 10px; border-radius: 6px; border: 1px solid #333; background: #2b2b3c; color: white; font-size: 1rem; }
        
        .primary-btn { width: 100%; padding: 12px; background: var(--primary); border: none; border-radius: 6px; color: white; font-weight: 600; cursor: pointer; transition: 0.2s; }
        .primary-btn:hover { opacity: 0.9; }
        .primary-btn:disabled { opacity: 0.6; cursor: wait; }
        
        .secondary-btn { width: 100%; padding: 10px; background: transparent; border: 1px solid #444; border-radius: 6px; color: #ccc; cursor: pointer; display: flex; align-items: center; justify-content: center; margin-top: 10px; }
        .secondary-btn:hover { border-color: #666; color: white; }
        
        .divider { height: 1px; background: #333; margin: 20px 0; }
        .status-msg { font-size: 0.8rem; color: #4caf50; margin-top: 10px; text-align: center; }
        
        .main-content { flex: 1; padding: 32px; overflow-y: auto; }
        header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 32px; }
        header h1 { font-size: 1.8rem; margin: 0; }
        .date-picker { display: flex; align-items: center; gap: 8px; color: var(--text-muted); background: white; padding: 8px 16px; border-radius: 20px; font-size: 0.9rem; }
        
        .dashboard-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 24px; margin-bottom: 24px; }
        .card { background: var(--card-bg); border-radius: 12px; padding: 24px; box-shadow: 0 2px 10px rgba(0,0,0,0.02); }
        
        .metric h3 { margin: 0 0 10px 0; font-size: 0.9rem; color: var(--text-muted); font-weight: 500; }
        .metric .value { font-size: 1.8rem; font-weight: 700; margin: 0; }
        .metric .subtext { font-size: 0.8rem; color: #ccc; }
        
        .chart-card h2 { margin-top: 0; font-size: 1.2rem; }
        .empty-state { height: 400px; display: flex; flex-direction: column; align-items: center; justify-content: center; color: #ccc; }
        .error-banner { background: #ffebee; color: #c62828; padding: 12px; border-radius: 8px; margin-bottom: 24px; display: flex; align-items: center; gap: 10px; }
      `}</style>
    </div>
  );
}

export default App;
