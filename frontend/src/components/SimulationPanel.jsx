import React, { useState, useCallback } from 'react';
import { Droplets, Factory, DollarSign, RotateCcw, Zap } from 'lucide-react';

const SLIDERS = [
    { key: 'Rainfall', label: 'Rainfall', icon: Droplets, min: -0.5, max: 0.5, step: 0.05, color: '#3b82f6', emoji: '🌧️' },
    { key: 'Local_Production_Volume', label: 'Production', icon: Factory, min: -0.5, max: 0.5, step: 0.05, color: '#10b981', emoji: '🏭' },
    { key: 'Exchange_Rate', label: 'Exchange Rate', icon: DollarSign, min: -0.2, max: 0.2, step: 0.02, color: '#f59e0b', emoji: '💲' },
];

const SimulationPanel = ({ onOverridesChange, isSimulating }) => {
    const [values, setValues] = useState(
        Object.fromEntries(SLIDERS.map(s => [s.key, 0]))
    );
    const [isExpanded, setIsExpanded] = useState(true);

    const hasOverrides = Object.values(values).some(v => v !== 0);

    const handleChange = useCallback((key, raw) => {
        const val = parseFloat(raw);
        const next = { ...values, [key]: val };
        setValues(next);

        // Only send non-zero overrides
        const overrides = {};
        for (const [k, v] of Object.entries(next)) {
            if (v !== 0) overrides[k] = v;
        }
        onOverridesChange(Object.keys(overrides).length > 0 ? overrides : null);
    }, [values, onOverridesChange]);

    const handleReset = useCallback(() => {
        const reset = Object.fromEntries(SLIDERS.map(s => [s.key, 0]));
        setValues(reset);
        onOverridesChange(null);
    }, [onOverridesChange]);

    return (
        <div style={{
            backgroundColor: 'var(--bg-card)',
            border: hasOverrides ? '1px solid rgba(139, 92, 246, 0.3)' : '1px solid var(--border-subtle)',
            borderRadius: '12px',
            overflow: 'hidden',
            transition: 'border-color 0.3s ease',
            boxShadow: hasOverrides ? '0 0 20px rgba(139, 92, 246, 0.08)' : 'none'
        }}>
            {/* Header */}
            <button
                onClick={() => setIsExpanded(!isExpanded)}
                style={{
                    width: '100%',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    padding: '14px 20px',
                    border: 'none',
                    backgroundColor: 'transparent',
                    cursor: 'pointer',
                    color: 'var(--text-primary)'
                }}
            >
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                    <div style={{
                        padding: '6px',
                        borderRadius: '8px',
                        backgroundColor: hasOverrides ? 'rgba(139, 92, 246, 0.12)' : 'var(--bg-secondary)',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        transition: 'background-color 0.3s ease'
                    }}>
                        <Zap size={16} style={{ color: hasOverrides ? '#8b5cf6' : 'var(--text-muted)' }} />
                    </div>
                    <span style={{ fontWeight: 700, fontSize: '0.9rem' }}>
                        What-If Simulator
                    </span>
                    {hasOverrides && (
                        <span style={{
                            fontSize: '0.65rem',
                            padding: '2px 8px',
                            borderRadius: '9999px',
                            backgroundColor: 'rgba(139, 92, 246, 0.12)',
                            color: '#8b5cf6',
                            fontWeight: 700,
                            textTransform: 'uppercase',
                            letterSpacing: '0.05em'
                        }}>
                            Active
                        </span>
                    )}
                    {isSimulating && (
                        <span style={{
                            fontSize: '0.65rem',
                            padding: '2px 8px',
                            borderRadius: '9999px',
                            backgroundColor: 'rgba(245, 158, 11, 0.12)',
                            color: '#f59e0b',
                            fontWeight: 700,
                            animation: 'pulse 1.5s ease-in-out infinite'
                        }}>
                            Simulating...
                        </span>
                    )}
                </div>
                <span style={{ color: 'var(--text-muted)', fontSize: '1.2rem', transition: 'transform 0.2s', transform: isExpanded ? 'rotate(0)' : 'rotate(-90deg)' }}>
                    ▾
                </span>
            </button>

            {/* Sliders */}
            {isExpanded && (
                <div style={{ padding: '4px 20px 20px 20px' }}>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                        {SLIDERS.map(({ key, label, emoji, min, max, step, color }) => {
                            const val = values[key];
                            const pct = Math.round(val * 100);
                            const isActive = val !== 0;

                            return (
                                <div key={key}>
                                    <div style={{
                                        display: 'flex',
                                        justifyContent: 'space-between',
                                        alignItems: 'center',
                                        marginBottom: '6px'
                                    }}>
                                        <span style={{
                                            fontSize: '0.8rem',
                                            fontWeight: 600,
                                            color: isActive ? color : 'var(--text-secondary)',
                                            transition: 'color 0.2s ease'
                                        }}>
                                            {emoji} {label}
                                        </span>
                                        <span style={{
                                            fontFamily: "'JetBrains Mono', monospace",
                                            fontSize: '0.75rem',
                                            fontWeight: 700,
                                            padding: '2px 8px',
                                            borderRadius: '6px',
                                            backgroundColor: isActive ? `${color}18` : 'var(--bg-secondary)',
                                            color: isActive ? color : 'var(--text-muted)',
                                            transition: 'all 0.2s ease',
                                            minWidth: '52px',
                                            textAlign: 'center'
                                        }}>
                                            {pct > 0 ? '+' : ''}{pct}%
                                        </span>
                                    </div>
                                    <input
                                        type="range"
                                        min={min}
                                        max={max}
                                        step={step}
                                        value={val}
                                        onChange={e => handleChange(key, e.target.value)}
                                        style={{
                                            width: '100%',
                                            height: '6px',
                                            borderRadius: '3px',
                                            appearance: 'none',
                                            WebkitAppearance: 'none',
                                            background: `linear-gradient(to right, ${color}40, ${color}40 ${((val - min) / (max - min)) * 100}%, var(--bg-secondary) ${((val - min) / (max - min)) * 100}%, var(--bg-secondary))`,
                                            cursor: 'pointer',
                                            outline: 'none',
                                            accentColor: color
                                        }}
                                    />
                                    <div style={{
                                        display: 'flex',
                                        justifyContent: 'space-between',
                                        marginTop: '2px'
                                    }}>
                                        <span style={{ fontSize: '0.6rem', color: 'var(--text-muted)' }}>{Math.round(min * 100)}%</span>
                                        <span style={{ fontSize: '0.6rem', color: 'var(--text-muted)' }}>{Math.round(max * 100)}%</span>
                                    </div>
                                </div>
                            );
                        })}
                    </div>

                    {/* Reset Button */}
                    {hasOverrides && (
                        <button
                            onClick={handleReset}
                            style={{
                                marginTop: '16px',
                                width: '100%',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                gap: '6px',
                                padding: '8px',
                                border: '1px solid var(--border-default)',
                                borderRadius: '8px',
                                backgroundColor: 'transparent',
                                color: 'var(--text-secondary)',
                                fontSize: '0.78rem',
                                fontWeight: 600,
                                cursor: 'pointer',
                                transition: 'all 0.2s ease'
                            }}
                            onMouseEnter={e => { e.currentTarget.style.backgroundColor = 'var(--bg-secondary)'; }}
                            onMouseLeave={e => { e.currentTarget.style.backgroundColor = 'transparent'; }}
                        >
                            <RotateCcw size={13} />
                            Reset All
                        </button>
                    )}
                </div>
            )}
        </div>
    );
};

export default SimulationPanel;
