import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Menu, ArrowRight, Sparkles, AlertTriangle, History, Plus } from 'lucide-react';
import './LandingPage.css';

export default function LandingPage() {
  const navigate = useNavigate();
  const [prefersReducedMotion, setPrefersReducedMotion] = useState(false);

  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    setPrefersReducedMotion(mediaQuery.matches);

    const handleChange = (e) => setPrefersReducedMotion(e.matches);
    mediaQuery.addEventListener('change', handleChange);
    return () => mediaQuery.removeEventListener('change', handleChange);
  }, []);

  return (
    <div className="landing-container">
      {/* Background Atmosphere Glow */}
      <div className="landing-atmosphere" />

      {/* LEFT PANEL — Video background + liquid glass card */}
      <div className="landing-left">
        {/* Background Video / Static Poster */}
        {!prefersReducedMotion ? (
          <video
            autoPlay
            loop
            muted
            playsInline
            poster="/assets/hero.png"
            className="landing-video"
            src="/video.mp4"
          />
        ) : (
          <img
            src="/assets/hero.png"
            alt="Hydra Earth View"
            className="landing-video"
          />
        )}

        {/* Liquid Glass Overlay Card */}
        <div className="landing-left-glass">
          
          {/* TOP NAV */}
          <nav className="landing-nav">
            <div className="landing-brand">
              <span className="landing-brand-logo">🌊</span>
              <span className="landing-brand-name">Hydra</span>
            </div>

            <button 
              className="landing-menu-btn"
              onClick={() => navigate('/dashboard')}
            >
              <Menu style={{ width: '16px', height: '16px', color: '#38bdf8' }} />
              <span>Menu</span>
            </button>
          </nav>

          {/* HERO CENTER CONTENT */}
          <div className="landing-hero">
            <h1 className="landing-title">
              Predicting India's next <br />
              <span className="landing-title-serif">flood and drought</span>
            </h1>

            {/* CTA Button */}
            <button
              onClick={() => navigate('/dashboard')}
              className="landing-cta"
            >
              <span>Launch Dashboard</span>
              <div className="landing-cta-icon">
                <ArrowRight style={{ width: '16px', height: '16px' }} />
              </div>
            </button>

            {/* Feature Pills */}
            <div className="landing-pills">
              <span className="landing-pill">Flood Risk Map</span>
              <span className="landing-pill">Drought Monitor</span>
              <span className="landing-pill">AI Emerging Factors</span>
            </div>
          </div>

          {/* BOTTOM MISSION SECTION */}
          <div className="landing-mission">
            <div className="landing-mission-label">
              EARLY WARNING SYSTEM
            </div>
            <p className="landing-mission-text">
              <span>Fifteen days </span>
              <span className="landing-mission-serif">is the difference </span>
              <span>between a disaster and a warning.</span>
            </p>
          </div>

        </div>
      </div>

      {/* RIGHT PANEL — Cosmic Sky Blue atmosphere + feature widgets */}
      <div className="landing-right">
        
        {/* TOP BAR */}
        <div className="landing-right-top">
          <a
            href="https://github.com/TarunScript/project-hydra"
            target="_blank"
            rel="noopener noreferrer"
            className="landing-github-link"
          >
            <svg style={{ width: '16px', height: '16px', fill: '#38bdf8' }} viewBox="0 0 24 24">
              <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z" />
            </svg>
            <span>GitHub Repository</span>
            <ArrowRight style={{ width: '14px', height: '14px', color: 'rgba(255,255,255,0.6)' }} />
          </a>

          <div className="landing-sparkles-btn" title="AI-Assisted Engine">
            <Sparkles style={{ width: '16px', height: '16px' }} />
          </div>
        </div>

        {/* INFO CARD */}
        <div className="landing-info-card">
          <h3 className="landing-info-title">How it works</h3>
          <p className="landing-info-text">
            Dual ML risk models combined with real-time AI contextual research for high-resolution 15-day early warning.
          </p>
        </div>

        {/* BOTTOM FEATURE CONTAINER */}
        <div className="landing-right-bottom">
          {/* Two Side-by-Side Feature Cards */}
          <div className="landing-feature-grid">
            <div 
              onClick={() => navigate('/dashboard')}
              className="landing-feature-card"
            >
              <div>
                <AlertTriangle style={{ width: '20px', height: '20px', color: '#38bdf8', marginBottom: '8px' }} />
                <h4 className="landing-feature-title">Risk Scoring</h4>
              </div>
              <p className="landing-feature-text">
                ML models predicting 15-day flood & drought intensity index.
              </p>
            </div>

            <div 
              onClick={() => navigate('/dashboard')}
              className="landing-feature-card"
            >
              <div>
                <History style={{ width: '20px', height: '20px', color: '#38bdf8', marginBottom: '8px' }} />
                <h4 className="landing-feature-title">Historical Archive</h4>
              </div>
              <p className="landing-feature-text">
                Historical environmental baseline & past disaster trends.
              </p>
            </div>
          </div>

          {/* Bottom Preview Card */}
          <div 
            onClick={() => navigate('/dashboard')}
            className="landing-preview-card"
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
              <img
                src="/assets/hero.png"
                alt="Dashboard Preview"
                className="landing-preview-thumb"
              />
              <div>
                <h4 className="landing-preview-title">Live Grid-Cell Risk Detection</h4>
                <p className="landing-preview-text">
                  High-resolution grid monitoring across Indian river basins and vulnerable districts.
                </p>
              </div>
            </div>

            <button 
              onClick={(e) => {
                e.stopPropagation();
                navigate('/dashboard');
              }}
              className="landing-plus-btn"
              title="Launch Dashboard"
            >
              <Plus style={{ width: '20px', height: '20px' }} />
            </button>
          </div>
        </div>

      </div>
    </div>
  );
}
