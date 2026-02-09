import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { AreaChart, Area, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceArea } from 'recharts';
import { LayoutDashboard, RefreshCw, AlertCircle, Calendar, Sparkles, TrendingUp, TrendingDown, Menu, ChevronLeft, ChevronRight, Layers, Maximize2, ZoomOut, Sun, Moon } from 'lucide-react';
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

  // Single Mode State
  const [selectedRegion, setSelectedRegion] = useState('');

  // Comparison Mode State
  const [isComparisonMode, setIsComparisonMode] = useState(false);
  const [selectedRegionsMulti, setSelectedRegionsMulti] = useState([]);

  const [selectedGrade, setSelectedGrade] = useState('');
  const [forecastDate, setForecastDate] = useState(6); // months

  const [forecastData, setForecastData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [trainingStatus, setTrainingStatus] = useState(null);

  const [marketIntel, setMarketIntel] = useState(null);
  const [latestPrice, setLatestPrice] = useState(null);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);

  // Zoom State
  const [refAreaLeft, setRefAreaLeft] = useState('');
  const [refAreaRight, setRefAreaRight] = useState('');
  const [left, setLeft] = useState('dataMin');
  const [right, setRight] = useState('dataMax');

  // Theme State
  const [isDarkMode, setIsDarkMode] = useState(() => {
    const saved = localStorage.getItem('theme');
    if (saved) return saved === 'dark';
    return window.matchMedia('(prefers-color-scheme: dark)').matches;
  });

  // Apply theme class to document
  useEffect(() => {
    if (isDarkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
    localStorage.setItem('theme', isDarkMode ? 'dark' : 'light');
  }, [isDarkMode]);

  const toggleTheme = () => setIsDarkMode(!isDarkMode);

  // Fetch Metadata and News on Load or Commodity Change
  useEffect(() => {
    setForecastData(null);
    setLatestPrice(null);
    if (!isComparisonMode) setSelectedRegion('');
    setSelectedRegionsMulti([]);
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

          // Default selection
          if (validRegions.length > 0) {
            if (!isComparisonMode) setSelectedRegion(validRegions[0]);
            // For multi select we start empty or maybe select first 2
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
        setMarketIntel({
          sentiment: 'Unavailable',
          confidence: 0,
          summary: 'Could not retrieve market news. Please check your connection or try again later.'
        });
      }
    };
    fetchMetadata();
    fetchNews();
  }, [selectedCommodity]); // Re-run when commodity changes (Comparison mode logic handled internally)

  // Validate selections when Grade changes
  const handleGradeChange = (e) => {
    const newGrade = e.target.value;
    setSelectedGrade(newGrade);
    const validRegions = regionsByGrade[newGrade] || regions;

    if (isComparisonMode) {
      // Filter out invalid selected regions
      const newMulti = selectedRegionsMulti.filter(r => validRegions.includes(r));
      setSelectedRegionsMulti(newMulti);
    } else {
      if (!validRegions.includes(selectedRegion)) {
        setSelectedRegion(validRegions[0] || '');
      }
    }
  };

  const handleRegionChange = (e) => {
    setSelectedRegion(e.target.value);
  };

  const toggleRegionSelection = (region) => {
    if (selectedRegionsMulti.includes(region)) {
      setSelectedRegionsMulti(selectedRegionsMulti.filter(r => r !== region));
    } else {
      if (selectedRegionsMulti.length < 5) { // Limit to 5 for clarity
        setSelectedRegionsMulti([...selectedRegionsMulti, region]);
      }
    }
  };

  const currentRegions = selectedGrade ? (regionsByGrade[selectedGrade] || regions) : regions;

  const handleForecast = async () => {
    setLoading(true);
    setError(null);
    try {
      if (isComparisonMode) {
        if (selectedRegionsMulti.length < 2) {
          setError("Please select at least 2 regions to compare.");
          setLoading(false);
          return;
        }
        const response = await axios.post(`${API_URL}/compare`, {
          commodity: selectedCommodity,
          grade: selectedGrade,
          regions: selectedRegionsMulti,
          months: forecastDate
        });

        // Process Compare Data
        // We need to merge all series into a single structure for Recharts
        // [{name: 'date', 'Colombo': price1, 'Kandy': price2, ...}]
        const results = response.data.results;
        const mergedData = {};

        // Collect all unique dates
        const allDates = new Set();
        results.forEach(res => {
          res.forecast.dates.forEach(d => allDates.add(d));
          res.history.dates.forEach(d => allDates.add(d));
        });
        const sortedDates = Array.from(allDates).sort();

        sortedDates.forEach(date => {
          mergedData[date] = { name: date };
        });

        // Fill prices
        results.forEach(res => {
          const regName = res.region;

          let lastHistoryDate = null;
          let lastHistoryPrice = null;

          // History
          res.history.dates.forEach((d, idx) => {
            if (mergedData[d]) {
              mergedData[d][regName] = res.history.prices[idx];
              lastHistoryDate = d;
              lastHistoryPrice = res.history.prices[idx];
            }
          });

          // Forecast
          res.forecast.dates.forEach((d, idx) => {
            if (mergedData[d]) {
              mergedData[d][`${regName}_Forecast`] = res.forecast.prices[idx];
            }
          });

          // Connect the lines: Ensure the last history point is also the start of the forecast line
          // But since they are different keys, we need the last history point to have BOTH keys?
          // No, better approach: The forecast series should START at the last history point.
          // BUT, we have unique dates. 
          // If T is last history, T+1 is first forecast.
          // To draw a continuous line, we need a point at T in the Forecast series.
          if (lastHistoryDate && mergedData[lastHistoryDate]) {
            mergedData[lastHistoryDate][`${regName}_Forecast`] = lastHistoryPrice;
          }
        });

        setForecastData(Object.values(mergedData));
        setLatestPrice(null); // Comparison has multiple prices

      } else {
        // Single Mode
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

        // Connect lines
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
      }

    } catch (err) {
      console.error("Forecast failed:", err);
      setError("Failed to generate forecast. " + (err.response?.data?.detail || err.message));
    } finally {
      setLoading(false);
    }
  };

  // Zoom Handlers
  const zoom = () => {
    if (refAreaLeft === refAreaRight || refAreaRight === '') {
      setRefAreaLeft('');
      setRefAreaRight('');
      return;
    }

    // Ensure left is smaller than right (users can drag L->R or R->L)
    let leftVal = refAreaLeft;
    let rightVal = refAreaRight;
    if (leftVal > rightVal) [leftVal, rightVal] = [rightVal, leftVal];

    setRefAreaLeft('');
    setRefAreaRight('');
    setLeft(leftVal);
    setRight(rightVal);
  };

  const zoomOut = () => {
    setLeft('dataMin');
    setRight('dataMax');
    setRefAreaLeft('');
    setRefAreaRight('');
  };

  // Theme Colors
  const getThemeColor = () => selectedCommodity === 'clove' ? '#059669' : '#d97706';
  // Palette for comparison
  const COLORS = ['#d97706', '#059669', '#2563eb', '#db2777', '#7c3aed', '#0891b2'];

  return (
    <div className="app-root relative">
      <MarketTicker commodity={selectedCommodity} grade={selectedGrade} price={latestPrice} isComparison={isComparisonMode} />

      {/* 2. MAIN LAYOUT - Padding Top for Ticker */}
      <div className="flex pt-12 h-screen relative" style={{ height: '100vh', overflow: 'hidden', color: 'var(--text-primary)' }}>
        {/* SIDEBAR */}
        <aside className={`sidebar shadow-xl z-40 flex flex-col h-full transition-all duration-300 ease-in-out ${isSidebarOpen ? 'w-72 p-6' : 'w-0 p-0 overflow-hidden border-none'}`} style={{ backgroundColor: 'var(--bg-card)', borderRight: '1px solid var(--border-subtle)' }}>
          <div className="brand-container justify-center mb-6">
            <img src={logo} alt="Verger Logo" className="h-16 w-auto object-contain" />
          </div>
          <div className="controls space-y-6 flex-1 overflow-y-auto pr-2 custom-scrollbar">

            {/* Mode Toggler */}
            <div className="p-1 rounded-lg flex mb-4" style={{ backgroundColor: 'var(--bg-secondary)' }}>
              <button
                className={`flex-1 py-1.5 text-xs font-bold rounded-md transition-all`}
                style={{
                  backgroundColor: !isComparisonMode ? 'var(--bg-card)' : 'transparent',
                  color: !isComparisonMode ? 'var(--text-primary)' : 'var(--text-secondary)',
                  boxShadow: !isComparisonMode ? '0 1px 2px rgba(0,0,0,0.1)' : 'none'
                }}
                onClick={() => { setIsComparisonMode(false); setForecastData(null); }}
              >
                Single Forecast
              </button>
              <button
                className={`flex-1 py-1.5 text-xs font-bold rounded-md transition-all`}
                style={{
                  backgroundColor: isComparisonMode ? 'var(--bg-card)' : 'transparent',
                  color: isComparisonMode ? 'var(--text-primary)' : 'var(--text-secondary)',
                  boxShadow: isComparisonMode ? '0 1px 2px rgba(0,0,0,0.1)' : 'none'
                }}
                onClick={() => { setIsComparisonMode(true); setForecastData(null); }}
              >
                Compare Regions
              </button>
            </div>

            <div className="control-group">
              <label className="block text-xs font-bold uppercase tracking-wider mb-2" style={{ color: 'var(--text-muted)' }}>Commodity</label>
              <div className="toggle-group">
                <button
                  className={`toggle-btn ${selectedCommodity === 'cinnamon' ? 'active' : ''}`}
                  style={{ color: selectedCommodity === 'cinnamon' ? 'var(--accent-gold)' : 'var(--text-secondary)' }}
                  onClick={() => setSelectedCommodity('cinnamon')}
                >
                  Cinnamon
                </button>
                <button
                  className={`toggle-btn ${selectedCommodity === 'clove' ? 'active' : ''}`}
                  style={{ color: selectedCommodity === 'clove' ? 'var(--accent-gold)' : 'var(--text-secondary)' }}
                  onClick={() => setSelectedCommodity('clove')}
                >
                  Clove
                </button>
              </div>
            </div>

            <div className="control-group">
              <label className="block text-xs font-bold uppercase tracking-wider mb-2" style={{ color: 'var(--text-muted)' }}>Grade</label>
              <select value={selectedGrade} onChange={handleGradeChange} className="glass-input text-sm">
                {grades.map(g => <option key={g} value={g}>{g}</option>)}
              </select>
            </div>

            <div className="control-group">
              <div className="flex justify-between items-center mb-2">
                <label className="block text-xs font-bold uppercase tracking-wider" style={{ color: 'var(--text-muted)' }}>Region(s)</label>
                {isComparisonMode && <span className="text-xs" style={{ color: 'var(--text-muted)' }}>{selectedRegionsMulti.length}/5</span>}
              </div>

              {!isComparisonMode ? (
                <select value={selectedRegion} onChange={handleRegionChange} className="glass-input text-sm">
                  {currentRegions.map(r => <option key={r} value={r}>{r}</option>)}
                </select>
              ) : (
                <div className="rounded-lg p-2 max-h-40 overflow-y-auto custom-scrollbar" style={{ backgroundColor: 'var(--bg-secondary)', border: '1px solid var(--border-default)' }}>
                  {currentRegions.map(r => (
                    <label key={r} className="flex items-center gap-2 p-1.5 rounded cursor-pointer text-sm" style={{ color: 'var(--text-primary)' }}>
                      <input
                        type="checkbox"
                        checked={selectedRegionsMulti.includes(r)}
                        onChange={() => toggleRegionSelection(r)}
                        className="rounded border-slate-300 text-amber-600 focus:ring-amber-500"
                      />
                      <span>{r}</span>
                    </label>
                  ))}
                  {currentRegions.length === 0 && <p className="text-xs p-2" style={{ color: 'var(--text-muted)' }}>No regions available</p>}
                </div>
              )}
            </div>

            <div className="control-group">
              <label className="block text-xs font-bold uppercase tracking-wider mb-2" style={{ color: 'var(--text-muted)' }}>Forecast Horizon (Months)</label>
              <div className="flex items-center gap-3">
                <input
                  type="range"
                  min="1"
                  max="24"
                  value={forecastDate}
                  onChange={(e) => setForecastDate(parseInt(e.target.value))}
                  className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-amber-600"
                />
                <span className="font-mono font-bold w-8" style={{ color: 'var(--text-secondary)' }}>{forecastDate}</span>
              </div>
            </div>

            <button
              className="primary-btn w-full flex items-center justify-center gap-2 mt-4"
              onClick={handleForecast}
              disabled={loading || (!isComparisonMode && !selectedRegion) || (isComparisonMode && selectedRegionsMulti.length < 2)}
            >
              {loading ? <RefreshCw className="animate-spin" size={18} /> : (isComparisonMode ? <Layers size={18} /> : <TrendingUp size={18} />)}
              {loading ? 'Processing...' : (isComparisonMode ? 'Compare Regions' : 'Generate Forecast')}
            </button>
          </div>
        </aside>


        {/* TOGGLE BUTTON */}
        <button
          onClick={() => setIsSidebarOpen(!isSidebarOpen)}
          className={`absolute top-1/2 z-50 p-2 rounded-full shadow-md hover:text-orange-600 transition-all duration-300 ease-in-out ${isSidebarOpen ? 'left-[17rem]' : 'left-4'}`}
          style={{ transform: 'translateY(-50%)', backgroundColor: 'var(--bg-card)', border: '1px solid var(--border-default)', color: 'var(--text-secondary)' }}
        >
          {isSidebarOpen ? <ChevronLeft size={20} /> : <ChevronRight size={20} />}
        </button>

        {/* MAIN CONTENT */}
        <main className="flex-1 p-8 overflow-y-auto relative flex flex-col h-full" style={{ backgroundColor: 'var(--bg-primary)' }}>
          <header className="flex justify-between items-center mb-8 shrink-0">
            <div>
              <h1 className="font-display text-3xl font-bold tracking-tight flex items-center gap-3" style={{ color: 'var(--text-primary)' }}>
                {selectedCommodity.charAt(0).toUpperCase() + selectedCommodity.slice(1)} Market Scout
                {isComparisonMode && <span className="bg-amber-100 text-amber-700 text-xs px-2 py-1 rounded-full border border-amber-200 uppercase tracking-widest font-bold">Compare Mode</span>}
              </h1>
              <p className="text-sm mt-1" style={{ color: 'var(--text-secondary)' }}>AI-Powered Price Forecasting & Analysis</p>
            </div>

            <div className="flex items-center gap-3">
              {/* Theme Toggle Button */}
              <button
                onClick={toggleTheme}
                className="p-2 rounded-full transition-all duration-300 hover:scale-110"
                style={{
                  backgroundColor: 'var(--bg-card)',
                  border: '1px solid var(--border-default)',
                  color: 'var(--text-secondary)'
                }}
                title={isDarkMode ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
              >
                {isDarkMode ? <Sun size={18} /> : <Moon size={18} />}
              </button>

              <div className="px-4 py-2 rounded-full shadow-sm flex items-center gap-3 font-medium text-sm" style={{ backgroundColor: 'var(--bg-card)', border: '1px solid var(--border-default)', color: 'var(--text-secondary)' }}>
                <Calendar size={16} /> <span>{new Date().toLocaleDateString()}</span>
              </div>
            </div>
          </header>

          {error && (
            <div className="bg-red-50 border border-red-200 text-red-600 p-4 rounded-lg flex items-center gap-3 mb-6 shrink-0 shadow-sm">
              <AlertCircle size={20} /> {error}
            </div>
          )}

          {!isComparisonMode && <IntelligenceCard data={marketIntel} />}

          {/* DASHBOARD METRICS (Single Mode Only) or Summary (Compare Mode) */}
          {!isComparisonMode && (
            <div className="grid grid-cols-3 gap-6 mb-6 shrink-0 dashboard-grid">
              <div className="card stripe-gold metric">
                <h3 className="tex-label" style={{ color: 'var(--text-secondary)' }}>Selected Grade</h3>
                <p className="value text-2xl font-bold" style={{ color: 'var(--text-primary)' }}>{selectedGrade || '-'}</p>
              </div>
              <div className="card stripe-green metric">
                <h3 className="tex-label" style={{ color: 'var(--text-secondary)' }}>Latest Price</h3>
                <p className="value text-2xl font-bold" style={{ color: 'var(--text-primary)' }}>
                  {forecastData ?
                    `LKR ${Math.round(latestPrice).toLocaleString()}`
                    : '-'}
                </p>
                <span className="subtext text-xs" style={{ color: 'var(--text-muted)' }}>Estimated current</span>
              </div>
              <div className="card stripe-gold metric">
                <h3 className="tex-label" style={{ color: 'var(--text-secondary)' }}>Forecast Trend</h3>
                <div className="value">
                  {forecastData ?
                    (() => {
                      const forecastPoints = forecastData.filter(d => d.Forecast !== null);
                      if (forecastPoints.length < 2) return '-';
                      const start = forecastPoints[0].Forecast;
                      const end = forecastPoints[forecastPoints.length - 1].Forecast;
                      const diff = end - start;
                      const percent = ((diff / start) * 100).toFixed(1);
                      return <span style={{ color: diff >= 0 ? '#059669' : '#ef4444', fontSize: '1.4rem', display: 'flex', alignItems: 'center', gap: '8px', fontWeight: 'bold' }}>
                        {diff >= 0 ? <TrendingUp size={24} /> : <TrendingDown size={24} />} {Math.abs(percent)}%
                      </span>;
                    })()
                    : '-'}
                </div>
              </div>
            </div>
          )}

          {/* CHART SECTION */}
          <div className="card chart-card flex-1 flex flex-col min-h-[400px] mb-6 shadow-md border-0">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-lg font-bold flex items-center gap-2" style={{ color: 'var(--text-primary)' }}>
                {isComparisonMode ? <Layers size={20} className="text-amber-500" /> : <TrendingUp size={20} className="text-emerald-500" />}
                {isComparisonMode ? 'Regional Price Comparison' : 'Price Forecast'}
              </h2>
              {forecastData && (
                <div className="text-xs flex items-center gap-2" style={{ color: 'var(--text-muted)' }}>
                  <Maximize2 size={14} /> Drag to zoom
                  <button onClick={zoomOut} className="ml-2 flex items-center gap-1 px-2 py-0.5 rounded transition-colors" style={{ backgroundColor: 'var(--bg-secondary)', color: 'var(--text-secondary)' }}>
                    <ZoomOut size={12} /> Reset
                  </button>
                </div>
              )}
            </div>

            <div className="flex-1 w-full min-h-0 relative">
              {forecastData ? (
                <ResponsiveContainer width="100%" height="100%">
                  {isComparisonMode ? (
                    <LineChart
                      data={forecastData}
                      margin={{ top: 10, right: 30, left: 20, bottom: 10 }}
                      onMouseDown={(e) => e && setRefAreaLeft(e.activeLabel)}
                      onMouseMove={(e) => refAreaLeft && e && setRefAreaRight(e.activeLabel)}
                      onMouseUp={zoom}
                    >
                      <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                      <XAxis
                        allowDataOverflow
                        domain={[left, right]}
                        type="category"
                        dataKey="name"
                        stroke="#94a3b8"
                        tick={{ fontSize: 11, fontFamily: 'JetBrains Mono', fill: '#64748b' }}
                        minTickGap={30}
                        tickLine={false}
                        dy={10}
                      />
                      <YAxis
                        allowDataOverflow
                        domain={['auto', 'auto']}
                        stroke="#94a3b8"
                        tickFormatter={(val) => `LKR ${val}`}
                        tick={{ fontFamily: 'JetBrains Mono', fill: '#64748b', fontSize: 11 }}
                        tickLine={false}
                        dx={-10}
                      />
                      <Tooltip
                        contentStyle={{ backgroundColor: '#fff', border: '1px solid #e2e8f0', borderRadius: '8px', boxShadow: '0 10px 15px -3px rgba(0,0,0,0.1)', fontFamily: 'JetBrains Mono', fontSize: '12px' }}
                        itemStyle={{ padding: 0 }}
                        formatter={(value) => [`LKR ${Math.round(value).toLocaleString()}`]}
                      />
                      <Legend wrapperStyle={{ paddingTop: '20px', fontFamily: 'Inter', fontSize: '13px' }} />

                      {selectedRegionsMulti.map((region, idx) => (
                        <React.Fragment key={region}>
                          {/* History Line (Solid) */}
                          <Line
                            type="monotone"
                            dataKey={region}
                            stroke={COLORS[idx % COLORS.length]}
                            strokeWidth={2.5}
                            dot={false}
                            activeDot={{ r: 6 }}
                          />
                          {/* Forecast Line (Dashed) */}
                          <Line
                            type="monotone"
                            dataKey={`${region}_Forecast`}
                            stroke={COLORS[idx % COLORS.length]}
                            strokeWidth={2.5}
                            strokeDasharray="5 5"
                            dot={false}
                            activeDot={{ r: 6 }}
                          />
                        </React.Fragment>
                      ))}
                      {refAreaLeft && refAreaRight ? (
                        <ReferenceArea x1={refAreaLeft} x2={refAreaRight} strokeOpacity={0.3} fill="#8884d8" fillOpacity={0.3} />
                      ) : null}
                    </LineChart>
                  ) : (
                    <AreaChart
                      data={forecastData}
                      margin={{ top: 10, right: 30, left: 10, bottom: 10 }}
                      onMouseDown={(e) => e && setRefAreaLeft(e.activeLabel)}
                      onMouseMove={(e) => refAreaLeft && e && setRefAreaRight(e.activeLabel)}
                      onMouseUp={zoom}
                    >
                      <defs>
                        <linearGradient id="colorMain" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor={getThemeColor()} stopOpacity={0.3} />
                          <stop offset="95%" stopColor={getThemeColor()} stopOpacity={0} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                      <XAxis
                        allowDataOverflow
                        domain={[left, right]}
                        type="category"
                        dataKey="name"
                        stroke="#94a3b8"
                        tick={{ fontSize: 11, fontFamily: 'JetBrains Mono', fill: '#64748b' }}
                        minTickGap={30}
                        tickLine={false}
                        dy={10}
                      />
                      <YAxis
                        allowDataOverflow
                        domain={['auto', 'auto']}
                        stroke="#94a3b8"
                        tickFormatter={(val) => `LKR ${val}`}
                        tick={{ fontFamily: 'JetBrains Mono', fill: '#64748b', fontSize: 11 }}
                        tickLine={false}
                        dx={-10}
                      />
                      <Tooltip
                        contentStyle={{ backgroundColor: '#fff', border: '1px solid #e2e8f0', borderRadius: '8px', boxShadow: '0 10px 15px -3px rgba(0,0,0,0.1)', fontFamily: 'JetBrains Mono', color: '#0f172a' }}
                        labelStyle={{ color: '#64748b', marginBottom: '5px' }}
                        formatter={(value) => [`LKR ${Math.round(value).toLocaleString()}`]}
                      />
                      <Legend wrapperStyle={{ paddingTop: '20px', fontFamily: 'Inter', fontSize: '13px' }} />

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
                        strokeWidth={3}
                        activeDot={{ r: 6, strokeWidth: 0 }}
                        name="Forecast"
                      />
                      {refAreaLeft && refAreaRight ? (
                        <ReferenceArea x1={refAreaLeft} x2={refAreaRight} strokeOpacity={0.3} fill="#8884d8" fillOpacity={0.3} />
                      ) : null}
                    </AreaChart>
                  )}
                </ResponsiveContainer>
              ) : (
                <div className="empty-state flex flex-col items-center justify-center h-full text-slate-300">
                  <div className="bg-slate-100 p-6 rounded-full mb-4">
                    {isComparisonMode ? <Layers size={48} className="text-slate-400" /> : <LayoutDashboard size={48} className="text-slate-400" />}
                  </div>
                  <p className="text-lg font-medium text-slate-500">Ready to Forecast</p>
                  <p className="text-sm mt-2 max-w-xs text-center text-slate-400">
                    {isComparisonMode ? 'Select multiple regions to compare price trends.' : 'Select comparison parameters to generate a prediction.'}
                  </p>
                </div>
              )}
            </div>
          </div>


          {/* FORECAST DATA TABLE */}
          {!isComparisonMode && forecastData && (
            <div className="card table-card mt-6">
              <h2 className="text-lg font-bold mb-6" style={{ color: 'var(--text-primary)' }}>Detailed Forecast Data</h2>
              <div className="overflow-x-auto">
                <table className="w-full text-left text-sm" style={{ color: 'var(--text-secondary)' }}>
                  <thead className="uppercase font-bold text-xs tracking-wider" style={{ backgroundColor: 'var(--bg-secondary)', color: 'var(--text-secondary)' }}>
                    <tr>
                      <th className="px-6 py-3" style={{ borderBottom: '1px solid var(--border-subtle)' }}>Date</th>
                      <th className="px-6 py-3" style={{ borderBottom: '1px solid var(--border-subtle)' }}>Projected Price (LKR)</th>
                      <th className="px-6 py-3" style={{ borderBottom: '1px solid var(--border-subtle)' }}>Trend</th>
                    </tr>
                  </thead>
                  <tbody style={{ borderColor: 'var(--border-subtle)' }}>
                    {forecastData
                      .filter(d => d.Forecast !== null)
                      .map((row, idx, arr) => {
                        const prevPrice = idx > 0 ? arr[idx - 1].Forecast : (forecastData.findLast(d => d.History !== null)?.History || row.Forecast);
                        const diff = row.Forecast - prevPrice;
                        const trendColor = diff >= 0 ? '#10b981' : '#ef4444';

                        return (
                          <tr key={idx} className="transition-colors" style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                            <td className="px-6 py-4 font-mono" style={{ color: 'var(--text-secondary)' }}>{row.name}</td>
                            <td className="px-6 py-4 font-bold" style={{ color: 'var(--text-primary)' }}>LKR {Math.round(row.Forecast).toLocaleString()}</td>
                            <td className="px-6 py-4 font-bold" style={{ color: trendColor }}>
                              {diff >= 0 ? '▲' : '▼'} {Math.abs(diff).toFixed(2)}
                            </td>
                          </tr>
                        );
                      })}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </main>
      </div >
    </div >
  );
}


const MarketTicker = ({ commodity, grade, price, isComparison }) => {
  return (
    <div className="fixed-ticker">
      <div className="w-full overflow-hidden flex whitespace-nowrap">
        <div className="flex animate-marquee">
          <div className="live-badge flex items-center gap-1">● LIVE</div>
          {!isComparison ? (
            <div className="flex items-center mx-6 font-mono text-sm">
              <span className="text-emerald-200 mr-2">{commodity ? (commodity.charAt(0).toUpperCase() + commodity.slice(1)) : 'Spice'} {grade || 'Index'}:</span>
              <span className="font-bold mr-2">{price ? `LKR ${Math.round(price).toLocaleString()}` : '...'}</span>
            </div>
          ) : (
            <div className="flex items-center mx-6 font-mono text-sm text-emerald-200">
              Comparing Multiple Regions
            </div>
          )}

          <div className="flex items-center mx-6 font-mono text-sm"><span className="text-emerald-200 mr-2">Cinnamon C5:</span><span className="font-bold mr-2">LKR 3,250</span><span className="text-emerald-300">+2.4%</span></div>
          <div className="flex items-center mx-6 font-mono text-sm"><span className="text-emerald-200 mr-2">Clove FAQ:</span><span className="font-bold mr-2">LKR 1,840</span><span className="text-red-300">-0.8%</span></div>
          <div className="flex items-center mx-6 font-mono text-sm"><span className="text-emerald-200 mr-2">Pepper Black:</span><span className="font-bold mr-2">LKR 1,120</span><span className="text-emerald-300">+1.1%</span></div>
        </div>
      </div>
    </div>
  );
}



const IntelligenceCard = ({ data }) => {
  if (!data) return (
    <div className="intelligence-card opacity-50">
      <div className="flex items-center gap-2 mb-2 font-bold uppercase text-xs tracking-widest" style={{ color: 'var(--text-secondary)' }}>
        <Sparkles size={14} /> Market Intelligence
      </div>
      <div className="h-20 rounded animate-pulse" style={{ backgroundColor: 'var(--bg-secondary)' }}></div>
    </div>
  );

  return (
    <div className="intelligence-card relative">
      <div className="flex gap-8 relative z-10 flex-wrap">
        <div className="min-w-[150px] flex-shrink-0 pr-6" style={{ borderRight: '1px solid var(--border-subtle)' }}>
          <div className="flex items-center gap-2 mb-2 font-bold uppercase text-xs tracking-widest" style={{ color: 'var(--text-secondary)' }}>
            Sentiment
          </div>
          <div className={`text-3xl font-bold font-heading ${data.sentiment === 'Bullish' ? 'text-emerald-500' : ''}`} style={{ color: data.sentiment === 'Bullish' ? '#10b981' : 'var(--text-primary)' }}>
            {data.sentiment || data.Sentiment || "Neutral"}
          </div>
        </div>
        <div className="flex-1 min-w-[200px] pl-2">
          <div className="flex items-center gap-2 mb-2 font-bold uppercase text-xs tracking-widest" style={{ color: 'var(--text-secondary)' }}>
            <Sparkles size={14} className="text-amber-500" /> AI Executive Summary
          </div>
          <p className="leading-relaxed text-sm" style={{ color: 'var(--text-secondary)' }}>
            {data.summary || data.Summary || "No summary available."}
          </p>
        </div>
      </div>
    </div>
  );
}

export default App;
