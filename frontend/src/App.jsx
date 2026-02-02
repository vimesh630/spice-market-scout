import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { LayoutDashboard, RefreshCw, AlertCircle, Calendar, Sparkles, TrendingUp, TrendingDown } from 'lucide-react';
import './App.css';
import logo from './assets/logo.png';

// Configure Axios base URL
const API_URL = 'http://127.0.0.1:8000';

function App() {
  const [selectedCommodity, setSelectedCommodity] = useState('cinnamon');
  const [regions, setRegions] = useState([]);
  const [grades, setGrades] = useState([]);
  const [allGradesCombinations, setAllGradesCombinations] = useState({});
  const [regionsByGrade, setRegionsByGrade] = useState({});

  const [selectedRegion, setSelectedRegion] = useState('');
  const [selectedGrade, setSelectedGrade] = useState('');
  const [forecastDate, setForecastDate] = useState(6); // months

  const [forecastData, setForecastData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [trainingStatus, setTrainingStatus] = useState(null);

  const [marketIntel, setMarketIntel] = useState(null);
  const [latestPrice, setLatestPrice] = useState(null);

  // Fetch Metadata and News on Load or Commodity Change
  useEffect(() => {
    setForecastData(null);
    setLatestPrice(null);
    setSelectedRegion('');
    setSelectedGrade('');
    setRegions([]);
    setGrades([]);
    setRegionsByGrade({});
    setMarketIntel(null);

    const fetchMetadata = async () => {
      try {
        const response = await axios.get(`${API_URL}/metadata?commodity=${selectedCommodity}`);

        const fetchedRegions = response.data.regions || [];
        const fetchedGrades = response.data.grades || [];
        const rByG = response.data.regions_by_grade || {};

        setRegions(fetchedRegions);
        setGrades(fetchedGrades);
        setRegionsByGrade(rByG);

        if (response.data.grades_by_region) {
          setAllGradesCombinations(response.data.grades_by_region);
        }

        if (fetchedGrades.length > 0) {
          const initialGrade = fetchedGrades[0];
          setSelectedGrade(initialGrade);
          const validRegions = rByG[initialGrade] || fetchedRegions;
          if (validRegions.length > 0) {
            setSelectedRegion(validRegions[0]);
          }
        }
      } catch (err) {
        console.error("Failed to fetch metadata:", err);
        setError("Could not connect to backend.");
      }
    };

    const fetchNews = async () => {
      try {
        const newsRes = await axios.get(`${API_URL}/news?commodity=${selectedCommodity}`);
        setMarketIntel(newsRes.data);
      } catch (err) {
        console.error("Failed to fetch news:", err);
      }
    };

    fetchMetadata();
    fetchNews();
  }, [selectedCommodity]);

  const handleGradeChange = (e) => {
    const newGrade = e.target.value;
    setSelectedGrade(newGrade);
    const validRegions = regionsByGrade[newGrade] || regions;
    if (!validRegions.includes(selectedRegion)) {
      setSelectedRegion(validRegions[0] || '');
    }
  };

  const handleRegionChange = (e) => {
    setSelectedRegion(e.target.value);
  };

  const currentRegions = selectedGrade ? (regionsByGrade[selectedGrade] || regions) : regions;

  const handleForecast = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await axios.post(`${API_URL}/predict?commodity=${selectedCommodity}`, {
        region: selectedRegion,
        grade: selectedGrade,
        months: forecastDate
      });

      const { dates: fDates, prices: fPrices } = response.data.forecast;
      const { dates: hDates, prices: hPrices } = response.data.history || { dates: [], prices: [] };

      const historyData = hDates.map((date, index) => ({
        name: date,
        History: hPrices[index],
        Forecast: null
      }));

      if (hDates.length > 0 && fDates.length > 0) {
        historyData[historyData.length - 1].Forecast = historyData[historyData.length - 1].History;
      }

      const forecastChartData = fDates.map((date, index) => ({
        name: date,
        History: null,
        Forecast: fPrices[index]
      }));

      const combinedData = [...historyData, ...forecastChartData];
      setForecastData(combinedData);

      const lastForecast = fPrices[fPrices.length - 1];
      const lastHistory = hPrices[hPrices.length - 1];
      setLatestPrice(lastForecast || lastHistory);

    } catch (err) {
      console.error("Forecast failed:", err);
      setError("Failed to generate forecast. " + (err.response?.data?.detail || err.message));
    } finally {
      setLoading(false);
    }
  };

  const handleRetrain = async () => {
    setTrainingStatus(`Training ${selectedCommodity}...`);
    try {
      await axios.post(`${API_URL}/retrain`, { epochs: 10 });
      setTrainingStatus("Training started in background!");
      setTimeout(() => setTrainingStatus(null), 5000);
    } catch (err) {
      setTrainingStatus("Retrain failed.");
    }
  };

  // Theme Colors
  const getThemeColor = () => selectedCommodity === 'clove' ? '#059669' : '#d97706';

  return (
    <div className="app-root">
      {/* 1. FIXED TOP TICKER */}
      <MarketTicker
        commodity={selectedCommodity}
        grade={selectedGrade}
        price={latestPrice}
      />

      {/* 2. MAIN LAYOUT - Padding Top for Ticker */}
      <div className="flex pt-12 text-slate-900 h-screen" style={{ height: '100vh', overflow: 'hidden' }}>

        {/* SIDEBAR */}
        <aside className="sidebar p-6 bg-white shadow-xl z-40 border-r border-slate-100 flex flex-col w-64 h-full">
          <div className="brand-container">
            <img src={logo} alt="Verger Logo" className="h-10 w-auto object-contain" />
            <span className="brand-name">Verger<br />Naturals</span>
          </div>

          <div className="controls space-y-6">
            <div className="control-group">
              <label>Commodity</label>
              <div className="toggle-group">
                <button
                  className={`toggle-btn ${selectedCommodity === 'cinnamon' ? 'active' : ''}`}
                  onClick={() => setSelectedCommodity('cinnamon')}
                >
                  Cinnamon
                </button>
                <button
                  className={`toggle-btn ${selectedCommodity === 'clove' ? 'active' : ''}`}
                  onClick={() => setSelectedCommodity('clove')}
                >
                  Clove
                </button>
              </div>
            </div>

            <div className="control-group">
              <label>Grade</label>
              <select value={selectedGrade} onChange={handleGradeChange} className="glass-input">
                {grades.map(g => <option key={g} value={g}>{g}</option>)}
              </select>
            </div>

            <div className="control-group">
              <label>Region</label>
              <select value={selectedRegion} onChange={handleRegionChange} className="glass-input">
                {currentRegions.map(r => <option key={r} value={r}>{r}</option>)}
              </select>
            </div>

            <button className="primary-btn" onClick={handleForecast} disabled={loading || !selectedRegion || !selectedGrade}>
              {loading ? 'Processing...' : 'Generate Forecast'}
            </button>
          </div>

          <div style={{ marginTop: 'auto' }}>
            <div className="h-px bg-slate-200 my-6"></div>
            <button className="secondary-btn" onClick={handleRetrain}>
              <RefreshCw size={16} style={{ marginRight: '8px' }} /> Retrain Model
            </button>
            {trainingStatus && <div className="status-msg">{trainingStatus}</div>}
          </div>
        </aside>

        {/* MAIN CONTENT */}
        <main className="flex-1 p-8 overflow-y-auto bg-slate-50 relative">
          <header className="flex justify-between items-center mb-10">
            <h1 className="font-display text-4xl font-bold text-slate-900 tracking-tight">
              {selectedCommodity.charAt(0).toUpperCase() + selectedCommodity.slice(1)} Forecasting
            </h1>
            <div className="bg-white px-4 py-2 rounded-full border border-slate-200 shadow-sm flex items-center gap-3 text-slate-500 font-medium">
              <Calendar size={18} /> <span>{new Date().toLocaleDateString()}</span>
            </div>
          </header>

          {error && (
            <div className="bg-red-50 border border-red-200 text-red-600 p-4 rounded-lg flex items-center gap-3 mb-8">
              <AlertCircle size={20} /> {error}
            </div>
          )}

          <IntelligenceCard data={marketIntel} />

          <div className="grid grid-cols-3 gap-6 mb-8 dashboard-grid">
            <div className="card stripe-gold metric">
              <h3>Selected Grade</h3>
              <p className="value">{selectedGrade || '-'}</p>
            </div>
            <div className="card stripe-green metric">
              <h3>Latest Price</h3>
              <p className="value">
                {forecastData ?
                  `LKR ${Math.round(latestPrice).toLocaleString()}`
                  : '-'}
              </p>
              <span className="subtext">Estimated current</span>
            </div>
            <div className="card stripe-gold metric">
              <h3>Forecast Trend</h3>
              <p className="value">
                {forecastData ?
                  (() => {
                    const forecastPoints = forecastData.filter(d => d.Forecast !== null);
                    if (forecastPoints.length < 2) return '-';
                    const start = forecastPoints[0].Forecast;
                    const end = forecastPoints[forecastPoints.length - 1].Forecast;
                    const diff = end - start;
                    const percent = ((diff / start) * 100).toFixed(1);
                    return <span style={{ color: diff >= 0 ? '#059669' : '#ef4444', fontSize: '1.4rem', display: 'flex', alignItems: 'center', gap: '8px' }}>
                      {diff >= 0 ? <TrendingUp size={24} /> : <TrendingDown size={24} />} {Math.abs(percent)}%
                    </span>;
                  })()
                  : '-'}
              </p>
            </div>
          </div>

          <div className="card chart-card">
            <h2>Price Forecast</h2>
            <div className="h-96 w-full">
              {forecastData ? (
                <ResponsiveContainer width="100%" height={400}>
                  <AreaChart data={forecastData}>
                    <defs>
                      <linearGradient id="colorMain" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor={getThemeColor()} stopOpacity={0.4} />
                        <stop offset="95%" stopColor={getThemeColor()} stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />

                    <XAxis
                      dataKey="name"
                      stroke="#94a3b8"
                      tick={{ fontSize: 13, fontFamily: 'JetBrains Mono', fill: '#64748b' }}
                      interval="preserveStartEnd"
                      minTickGap={50}
                      tickLine={false}
                      dy={10}
                    />
                    <YAxis
                      stroke="#94a3b8"
                      domain={['auto', 'auto']}
                      tickFormatter={(val) => `LKR ${val}`}
                      tick={{ fontFamily: 'JetBrains Mono', fill: '#64748b', fontSize: 13 }}
                      tickLine={false}
                      dx={-10}
                    />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#fff', border: 'none', borderRadius: '8px', boxShadow: '0 4px 6px -1px rgba(0,0,0,0.1)', fontFamily: 'JetBrains Mono', color: '#0f172a' }}
                      labelStyle={{ color: '#64748b' }}
                      formatter={(value) => [`LKR ${Math.round(value).toLocaleString()}`]}
                    />
                    <Legend wrapperStyle={{ color: '#0f172a', fontFamily: 'Inter' }} />

                    <Area
                      type="monotone"
                      dataKey="History"
                      stroke="#64748b"
                      fill="transparent"
                      strokeWidth={2}
                      name="Historical"
                    />
                    <Area
                      type="monotone"
                      dataKey="Forecast"
                      stroke={getThemeColor()}
                      fillOpacity={1}
                      fill="url(#colorMain)"
                      strokeWidth={4}
                      strokeDasharray="5 5"
                      name="Forecast"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              ) : (
                <div className="empty-state">
                  <LayoutDashboard size={48} />
                  <p className="text-body mt-4">Select parameters and click Generate Forecast</p>
                </div>
              )}
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}

const MarketTicker = ({ commodity, grade, price }) => {
  // Dynamic items & mocks
  const tickerItems = [
    { label: `${commodity ? (commodity.charAt(0).toUpperCase() + commodity.slice(1)) : 'Spice'} ${grade || 'Index'}`, value: price ? `LKR ${Math.round(price).toLocaleString()}` : 'Loading...', change: '+0.0%', type: 'neutral' },
    { label: 'Cinnamon C5', value: 'LKR 3,250', change: '+2.4%', type: 'positive' },
    { label: 'Clove FAQ', value: 'LKR 1,840', change: '-0.8%', type: 'negative' },
    { label: 'Pepper Black', value: 'LKR 1,120', change: '+1.1%', type: 'positive' },
    { label: 'Cardamom LG', value: 'LKR 4,500', change: '+0.5%', type: 'positive' },
  ];

  return (
    <div className="fixed-ticker">
      <span className="live-badge">LIVE</span>
      <div className="ticker-wrapper">
        {tickerItems.map((item, i) => (
          <div key={i} className="flex items-center mr-16">
            <span className="text-emerald-100 font-medium mr-3">{item.label}</span>
            <span className="font-mono font-bold text-white text-lg mr-3">{item.value}</span>
            <span className={`font-mono font-bold ${item.type === 'negative' ? 'text-red-300' : 'text-emerald-300'}`}>{item.change}</span>
          </div>
        ))}
        {tickerItems.map((item, i) => (
          <div key={`dup-${i}`} className="flex items-center mr-16">
            <span className="text-emerald-100 font-medium mr-3">{item.label}</span>
            <span className="font-mono font-bold text-white text-lg mr-3">{item.value}</span>
            <span className={`font-mono font-bold ${item.type === 'negative' ? 'text-red-300' : 'text-emerald-300'}`}>{item.change}</span>
          </div>
        ))}
      </div>
    </div>
  );
};

const IntelligenceCard = ({ data }) => {
  if (!data) return (
    <div className="intelligence-card">
      <h3 className="flex items-center gap-2 mb-4 font-bold text-xl text-amber-600 font-display">
        <Sparkles size={20} /> Market Intelligence (Gemini AI)
      </h3>
      <div className="flex gap-10">
        <p className="text-slate-400 italic">Analyzing latest market news...</p>
      </div>
    </div>
  );

  return (
    <div className="intelligence-card">
      <h3 className="flex items-center gap-2 mb-4 font-bold text-xl text-amber-600 font-display">
        <Sparkles size={20} /> Market Intelligence (Gemini AI)
      </h3>
      <div className="flex gap-12">
        <div className="flex flex-col gap-1">
          <span className="text-xs font-bold uppercase text-slate-500 tracking-wider">Sentiment</span>
          <span className={`text-lg font-bold font-display ${data.sentiment === 'Bullish' ? 'text-emerald-600' : (data.sentiment === 'Bearish' ? 'text-red-500' : 'text-slate-700')}`}>
            {data.sentiment}
          </span>
        </div>
        <div className="flex flex-col gap-1">
          <span className="text-xs font-bold uppercase text-slate-500 tracking-wider">Confidence</span>
          <span className="text-lg font-bold font-mono text-slate-900">{Math.round((data.confidence || 0) * 100)}%</span>
        </div>
        <div className="flex flex-col gap-1 flex-1">
          <span className="text-xs font-bold uppercase text-slate-500 tracking-wider">Summary</span>
          <span className="text-slate-700 font-medium leading-relaxed">{data.summary || "No summary available."}</span>
        </div>
      </div>
    </div>
  );
};

export default App;
