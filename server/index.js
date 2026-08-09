/**
 * server/index.js — Lightweight Express backend for Project Hydra
 *
 * Serves the /api/emerging-factors endpoint.
 * Runs in MOCK_MODE by default (no API key needed).
 * Set PERPLEXITY_API_KEY in .env to switch to live Sonar search.
 */

import express from 'express';
import cors from 'cors';
import { getEmergingFactors } from './emerging-factors.js';

const app = express();
const PORT = process.env.PORT || 3001;

app.use(cors());
app.use(express.json());

// ── Health check ──
app.get('/api/health', (_req, res) => {
  res.json({ status: 'ok', mode: process.env.PERPLEXITY_API_KEY ? 'live' : 'mock' });
});

// ── Emerging Factors endpoint ──
app.get('/api/emerging-factors', async (req, res) => {
  const { lat, lon, location_name } = req.query;

  if (!lat || !lon) {
    return res.status(400).json({ error: 'lat and lon query params are required' });
  }

  try {
    const result = await getEmergingFactors({
      lat: parseFloat(lat),
      lon: parseFloat(lon),
      location_name: location_name || `${lat}, ${lon}`
    });
    res.json(result);
  } catch (err) {
    console.error('[emerging-factors] Error:', err.message);
    res.status(500).json({ error: 'Failed to fetch emerging factors', detail: err.message });
  }
});

// ── 404 handler ──
app.use((_req, res) => {
  res.status(404).json({ error: 'Endpoint not found', status: 404 });
});

// ── Global error handler ──
app.use((err, _req, res, _next) => {
  console.error('[server] Unhandled error:', err.message);
  res.status(500).json({ error: 'Internal server error', detail: err.message });
});

// ── Start with port-in-use detection ──
const server = app.listen(PORT, () => {
  console.log(`\n  🌊 Project Hydra API server running`);
  console.log(`  ➜  Local:   http://localhost:${PORT}`);
  console.log(`  ➜  Mode:    ${process.env.PERPLEXITY_API_KEY ? 'LIVE (Perplexity Sonar)' : 'MOCK (no API key set)'}\n`);
});

server.on('error', (err) => {
  if (err.code === 'EADDRINUSE') {
    console.error(`\n  ⚠ Port ${PORT} is already in use!`);
    console.error(`  → Kill it: lsof -ti:${PORT} | xargs kill -9`);
    console.error(`  → Or set a different port: PORT=3002 node index.js\n`);
  } else {
    console.error(`\n  ✗ Server failed to start:`, err.message);
  }
  process.exit(1);
});

// Graceful shutdown
process.on('SIGINT', () => {
  console.log('\n  👋 Server stopped gracefully.');
  server.close();
  process.exit(0);
});

process.on('uncaughtException', (err) => {
  console.error('[server] Uncaught exception:', err.message);
});

process.on('unhandledRejection', (reason) => {
  console.error('[server] Unhandled promise rejection:', reason);
});
