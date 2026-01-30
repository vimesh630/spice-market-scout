import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { TrendingUp, TrendingDown, Minus, AlertCircle, ShieldCheck } from 'lucide-react';

const IntelligenceCard = () => {
    const [intelligence, setIntelligence] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        const fetchNews = async () => {
            try {
                // Fetch from our local API
                const response = await axios.get('http://127.0.0.1:8000/news');
                setIntelligence(response.data);
            } catch (err) {
                console.error("Failed to fetch intelligence:", err);
                setError("Market Intelligence unavailable.");
            } finally {
                setLoading(false);
            }
        };

        fetchNews();
    }, []);

    if (loading) {
        return (
            <div className="intelligence-card skeleton">
                <div className="skeleton-icon"></div>
                <div className="skeleton-text">Analyzing market sentiment...</div>
            </div>
        );
    }

    if (error) {
        return (
            <div className="intelligence-card error">
                <AlertCircle size={24} color="#ef5350" />
                <span>{error}</span>
            </div>
        );
    }

    if (!intelligence) return null;

    const { sentiment, confidence, summary } = intelligence;

    // Determine visuals based on sentiment
    const isBullish = sentiment.toLowerCase().includes('bullish');
    const isBearish = sentiment.toLowerCase().includes('bearish');

    let Icon = Minus;
    let color = '#ffa726'; // Neutral Orange

    if (isBullish) {
        Icon = TrendingUp;
        color = '#66bb6a'; // Green
    } else if (isBearish) {
        Icon = TrendingDown;
        color = '#ef5350'; // Red
    }

    return (
        <div className="intelligence-card">
            <div className="icon-wrapper" style={{ backgroundColor: `${color}20`, color: color }}>
                <Icon size={32} />
            </div>

            <div className="content">
                <div className="header">
                    <h3>Market Intelligence</h3>
                    <div className="badge tooltip-container">
                        <ShieldCheck size={14} style={{ marginRight: 4 }} />
                        {Math.round(confidence * 100)}% Confidence
                    </div>
                </div>

                <p className="summary">{summary}</p>

                <div className="footer">
                    <span className="source-label">Powered by Gemini 1.5 Flash</span>
                </div>
            </div>

            <style>{`
                .intelligence-card {
                    background: white;
                    border-radius: 16px;
                    padding: 24px;
                    display: flex;
                    gap: 20px;
                    align-items: flex-start;
                    box-shadow: 0 4px 20px rgba(0,0,0,0.05);
                    margin-bottom: 32px;
                    border: 1px solid rgba(0,0,0,0.05);
                    transition: transform 0.2s;
                }
                
                .intelligence-card:hover {
                    transform: translateY(-2px);
                    box-shadow: 0 8px 30px rgba(0,0,0,0.08);
                }

                .icon-wrapper {
                    padding: 16px;
                    border-radius: 12px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }

                .content {
                    flex: 1;
                }

                .header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 8px;
                }

                .header h3 {
                    margin: 0;
                    font-size: 1.1rem;
                    color: #1e1e2d;
                    font-weight: 700;
                }

                .badge {
                    background: #f5f6fa;
                    color: #666;
                    padding: 4px 10px;
                    border-radius: 20px;
                    font-size: 0.75rem;
                    font-weight: 600;
                    display: flex;
                    align-items: center;
                    border: 1px solid #eee;
                }

                .summary {
                    margin: 0 0 12px 0;
                    color: #555;
                    line-height: 1.5;
                    font-size: 0.95rem;
                }

                .footer {
                    display: flex;
                    justify-content: flex-end;
                }
                
                .source-label {
                    font-size: 0.7rem;
                    color: #aaa;
                    font-style: italic;
                }

                /* Skeleton / Loading */
                .skeleton {
                    align-items: center;
                    justify-content: center;
                    height: 120px;
                    color: #999;
                    font-style: italic;
                }

                /* Error state */
                .error {
                    color: #ef5350;
                    align-items: center;
                }
            `}</style>
        </div>
    );
};

export default IntelligenceCard;
