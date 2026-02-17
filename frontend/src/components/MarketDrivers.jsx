import React from 'react';
import { ArrowUp, ArrowDown, HelpCircle, Activity } from 'lucide-react';

const MarketDrivers = ({ data, scenario }) => {
    const drivers = data?.scenarios?.[scenario]?.explanations || [];

    // Flatten: each element can be a list of strings (per month) or a string
    // We show the latest month's drivers (last element)
    const latestDrivers = drivers.length > 0
        ? (Array.isArray(drivers[drivers.length - 1]) ? drivers[drivers.length - 1] : [drivers[drivers.length - 1]])
        : [];

    if (latestDrivers.length === 0) {
        return (
            <div className="xai-card empty" style={{
                backgroundColor: 'var(--bg-card)',
                border: '1px solid var(--border-subtle)',
                borderRadius: '12px',
                padding: '24px',
                height: '100%',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                color: 'var(--text-muted)'
            }}>
                <Activity className="w-8 h-8 mb-2" style={{ opacity: 0.5 }} />
                <p style={{ fontSize: '0.85rem' }}>No XAI drivers detected for this scenario.</p>
            </div>
        );
    }

    return (
        <div style={{
            backgroundColor: 'var(--bg-card)',
            border: '1px solid var(--border-subtle)',
            borderRadius: '12px',
            padding: '24px',
            height: '100%',
            display: 'flex',
            flexDirection: 'column'
        }}>
            {/* Header */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '16px' }}>
                <div style={{
                    padding: '8px',
                    backgroundColor: 'var(--bg-secondary)',
                    borderRadius: '8px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center'
                }}>
                    <Activity size={18} style={{ color: 'var(--accent-gold, #d97706)' }} />
                </div>
                <h3 style={{
                    fontWeight: 700,
                    fontSize: '0.95rem',
                    color: 'var(--text-primary)',
                    margin: 0
                }}>Key Market Drivers</h3>
            </div>

            {/* Subtitle: which month */}
            <p style={{
                fontSize: '0.75rem',
                color: 'var(--text-muted)',
                marginBottom: '12px',
                textTransform: 'uppercase',
                letterSpacing: '0.05em',
                fontWeight: 600
            }}>
                Latest forecast month • {scenario}
            </p>

            {/* Driver Cards */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', flex: 1 }}>
                {latestDrivers.map((text, index) => {
                    const isBullish = text.toLowerCase().includes('driven up') || text.toLowerCase().includes('raising');
                    const isBearish = text.toLowerCase().includes('driven down') || text.toLowerCase().includes('lowering') || text.toLowerCase().includes('dampened');
                    const isStable = text.toLowerCase().includes('stable market');

                    // Parse cause and impact
                    const parts = text.split('(');
                    const cause = parts[0].trim();
                    const impact = parts[1] ? parts[1].replace(')', '').trim() : '';

                    // Clean the cause text: remove "Price driven up/down by "
                    const cleanCause = cause
                        .replace(/^Price driven (up|down) by /i, '')
                        .replace(/^Price dampened by /i, '')
                        .trim();

                    // Determine colors
                    let bgColor, borderColor, iconBg, iconColor, impactColor;
                    if (isBullish) {
                        bgColor = 'rgba(16, 185, 129, 0.08)';
                        borderColor = 'rgba(16, 185, 129, 0.2)';
                        iconBg = 'rgba(16, 185, 129, 0.15)';
                        iconColor = '#059669';
                        impactColor = '#059669';
                    } else if (isBearish) {
                        bgColor = 'rgba(239, 68, 68, 0.08)';
                        borderColor = 'rgba(239, 68, 68, 0.2)';
                        iconBg = 'rgba(239, 68, 68, 0.15)';
                        iconColor = '#dc2626';
                        impactColor = '#dc2626';
                    } else {
                        bgColor = 'var(--bg-secondary)';
                        borderColor = 'var(--border-subtle)';
                        iconBg = 'var(--bg-secondary)';
                        iconColor = 'var(--text-muted)';
                        impactColor = 'var(--text-secondary)';
                    }

                    return (
                        <div key={index} style={{
                            padding: '12px 14px',
                            borderRadius: '10px',
                            border: `1px solid ${borderColor}`,
                            backgroundColor: bgColor,
                            display: 'flex',
                            alignItems: 'flex-start',
                            gap: '12px',
                            transition: 'transform 0.15s ease, box-shadow 0.15s ease',
                            cursor: 'default'
                        }}
                            onMouseEnter={e => {
                                e.currentTarget.style.transform = 'translateY(-1px)';
                                e.currentTarget.style.boxShadow = '0 4px 12px rgba(0,0,0,0.06)';
                            }}
                            onMouseLeave={e => {
                                e.currentTarget.style.transform = 'translateY(0)';
                                e.currentTarget.style.boxShadow = 'none';
                            }}
                        >
                            {/* Icon */}
                            <div style={{
                                marginTop: '2px',
                                padding: '4px',
                                borderRadius: '50%',
                                backgroundColor: iconBg,
                                color: iconColor,
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                flexShrink: 0
                            }}>
                                {isStable ? <HelpCircle size={14} /> : isBullish ? <ArrowUp size={14} /> : isBearish ? <ArrowDown size={14} /> : <HelpCircle size={14} />}
                            </div>

                            {/* Text */}
                            <div style={{ flex: 1, minWidth: 0 }}>
                                <p style={{
                                    fontSize: '0.85rem',
                                    fontWeight: 600,
                                    color: 'var(--text-primary)',
                                    margin: 0,
                                    lineHeight: 1.4
                                }}>
                                    {isStable ? text : cleanCause}
                                </p>
                                {impact && !isStable && (
                                    <p style={{
                                        fontSize: '0.75rem',
                                        fontWeight: 700,
                                        color: impactColor,
                                        margin: '4px 0 0 0',
                                        fontFamily: "'JetBrains Mono', monospace"
                                    }}>
                                        {impact}
                                    </p>
                                )}
                            </div>
                        </div>
                    );
                })}
            </div>

            {/* Month selector hint */}
            {drivers.length > 1 && (
                <p style={{
                    fontSize: '0.7rem',
                    color: 'var(--text-muted)',
                    marginTop: '12px',
                    textAlign: 'center',
                    fontStyle: 'italic'
                }}>
                    Showing drivers for month {drivers.length} of {drivers.length}
                </p>
            )}
        </div>
    );
};

export default MarketDrivers;
