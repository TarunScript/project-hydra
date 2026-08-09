/**
 * server/emerging-factors.js — Emerging Factors service
 *
 * MOCK_MODE: Returns hardcoded, realistic mock findings (default).
 * LIVE_MODE: Calls Perplexity Sonar API with web search (when PERPLEXITY_API_KEY is set).
 *
 * Swap is automatic — set the env var and restart the server. No code changes needed.
 */

// ── Valid categories (strictly enforced) ──
const VALID_CATEGORIES = [
  'New/expanding data centers or AI infrastructure',
  'New/expanding semiconductor fabrication plants',
  'Lithium/rare-earth/critical-mineral extraction projects',
  'Green hydrogen production or direct air capture facilities',
  'Groundwater extraction trends or new industrial water permits',
  'Major upstream dam/reservoir/irrigation changes',
  'Large-scale land-use change (deforestation, urban expansion)'
];

// ── Excluded topics (hard filter) ──
const EXCLUDED_KEYWORDS = [
  'geoengineering', 'stratospheric aerosol', 'weather modification',
  'cloud seeding', 'satellite re-entry', 'climate manipulation',
  'solar radiation management'
];

// ── In-memory cache (district-level, 24h TTL) ──
const cache = new Map();
const CACHE_TTL_MS = 24 * 60 * 60 * 1000; // 24 hours

function getCacheKey(lat, lon) {
  // Round to ~0.5° for district-level grouping
  const rLat = (Math.round(lat * 2) / 2).toFixed(1);
  const rLon = (Math.round(lon * 2) / 2).toFixed(1);
  return `${rLat},${rLon}`;
}

function getCached(key) {
  const entry = cache.get(key);
  if (!entry) return null;
  if (Date.now() - entry.timestamp > CACHE_TTL_MS) {
    cache.delete(key);
    return null;
  }
  return entry.data;
}

function setCache(key, data) {
  cache.set(key, { data, timestamp: Date.now() });
}

// ── Filter findings: enforce categories, sources, and excluded topics ──
function filterFindings(findings) {
  if (!Array.isArray(findings)) return [];

  return findings.filter(f => {
    // Must have a source URL
    if (!f.source_url || typeof f.source_url !== 'string' || !f.source_url.startsWith('http')) {
      return false;
    }
    // Must be one of the 7 valid categories
    if (!VALID_CATEGORIES.includes(f.category)) {
      return false;
    }
    // Must not contain excluded keywords
    const text = `${f.summary} ${f.relevance}`.toLowerCase();
    if (EXCLUDED_KEYWORDS.some(kw => text.includes(kw))) {
      return false;
    }
    return true;
  });
}

// ── LLM Prompt Template (embedded exactly as specified) ──
function buildPrompt(location_name, lat, lon) {
  return `You are a research assistant supporting a flood/drought risk system for India.
Location: ${location_name}, ${lat}, ${lon}

Search for real, verifiable, recent (last 24 months) developments near this
location strictly within these categories only:
1. New/expanding data centers or AI infrastructure
2. New/expanding semiconductor fabrication plants
3. Lithium/rare-earth/critical-mineral extraction projects
4. Green hydrogen production or direct air capture facilities
5. Groundwater extraction trends or new industrial water permits
6. Major upstream dam/reservoir/irrigation changes
7. Large-scale land-use change (deforestation, urban expansion)

Rules:
- Every finding MUST include a source URL. No source, no finding — omit it instead of guessing.
- Do not include anything outside the 7 categories above, even if it seems relevant.
- Do not speculate about atmospheric/climate mechanisms not supported by your source.
- Never mention geoengineering, weather modification, cloud seeding, or satellite
  re-entry effects, even if a source raises them — these are out of scope.
- If nothing relevant is found within ~50km of the location, say so explicitly.

Return strict JSON matching this schema:
{
  "location": "string",
  "findings": [
    {
      "category": "one of the 7 categories above, verbatim",
      "summary": "one sentence, plain language",
      "relevance": "why this could matter for local flood/drought risk",
      "source_url": "string",
      "source_date": "string"
    }
  ],
  "no_findings": true or false
}`;
}

// ════════════════════════════════════════════════════════════════
// MOCK DATA — remove once PERPLEXITY_API_KEY is set
// ════════════════════════════════════════════════════════════════
const MOCK_RESPONSES = {
  // Assam / Guwahati area (~26°N, 92°E)
  '26.0,92.5': {
    location: 'Assam - Brahmaputra Basin',
    findings: [
      {
        category: 'Major upstream dam/reservoir/irrigation changes',
        summary: 'China completed a major hydropower dam on the Yarlung Tsangpo (upper Brahmaputra) in Tibet, with a reported capacity of 60 GW.',
        relevance: 'Upstream flow regulation on the Brahmaputra can alter seasonal flood patterns in Assam, potentially intensifying or shifting peak discharge timing.',
        source_url: 'https://www.reuters.com/business/energy/china-approves-high-dam-project-yarlung-tsangpo-river-tibet-2024-12-25/',
        source_date: '2024-12-25'
      },
      {
        category: 'Large-scale land-use change (deforestation, urban expansion)',
        summary: 'Guwahati metropolitan area expanded by approximately 12% between 2022-2024, converting wetlands and floodplain areas to residential zones.',
        relevance: 'Loss of natural flood-absorbing wetlands reduces the landscape\'s capacity to buffer monsoon surges, increasing urban flood risk.',
        source_url: 'https://www.downtoearth.org.in/environment/guwahati-urban-expansion-wetland-loss',
        source_date: '2024-08-14'
      },
      {
        category: 'Groundwater extraction trends or new industrial water permits',
        summary: 'Central Ground Water Board reported a 15% increase in industrial groundwater extraction permits in the Kamrup district since 2023.',
        relevance: 'Increased extraction can lower water tables during dry periods while reducing aquifer capacity to absorb excess monsoon recharge, affecting both drought and flood resilience.',
        source_url: 'https://cgwb.gov.in/reports/kamrup-groundwater-assessment-2024',
        source_date: '2024-06-01'
      }
    ],
    no_findings: false
  },

  // Marathwada / Latur area (~19°N, 76.5°E)
  '19.0,76.5': {
    location: 'Marathwada - Latur Region',
    findings: [
      {
        category: 'Green hydrogen production or direct air capture facilities',
        summary: 'Reliance Industries announced a green hydrogen hub near Jalna, Marathwada, with an initial 100 MW electrolyzer capacity requiring significant freshwater input.',
        relevance: 'Green hydrogen electrolysis requires ~9 litres of purified water per kg of H2 produced, potentially straining already water-scarce Marathwada\'s resources.',
        source_url: 'https://economictimes.indiatimes.com/industry/renewables/reliance-green-hydrogen-maharashtra-jalna/articleshow/98765432.cms',
        source_date: '2025-01-18'
      },
      {
        category: 'Groundwater extraction trends or new industrial water permits',
        summary: 'Maharashtra Groundwater Authority flagged 23 talukas in Marathwada as "over-exploited" in its 2024 assessment, up from 17 in 2022.',
        relevance: 'Accelerating groundwater depletion directly worsens drought vulnerability and reduces the region\'s resilience to consecutive dry monsoon years.',
        source_url: 'https://www.hindustantimes.com/cities/pune-news/marathwada-groundwater-overexploited-talukas-increase-2024-101720000000.html',
        source_date: '2024-07-03'
      }
    ],
    no_findings: false
  },

  // Default — no findings for unknown locations
  default: {
    location: 'Unknown Region',
    findings: [],
    no_findings: true
  }
};

function getMockResponse(lat, lon, location_name) {
  const key = getCacheKey(lat, lon);
  const mock = MOCK_RESPONSES[key] || MOCK_RESPONSES.default;
  return {
    ...mock,
    location: location_name || mock.location
  };
}
// ════════════════════════════════════════════════════════════════
// END MOCK DATA
// ════════════════════════════════════════════════════════════════

// ── Live API call (Perplexity Sonar) ──
async function callPerplexityAPI(prompt) {
  const apiKey = process.env.PERPLEXITY_API_KEY;
  if (!apiKey) throw new Error('PERPLEXITY_API_KEY not set');

  const res = await fetch('https://api.perplexity.ai/chat/completions', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${apiKey}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      model: 'sonar',
      messages: [
        { role: 'system', content: 'You are a factual research assistant. Return only valid JSON. No markdown, no commentary.' },
        { role: 'user', content: prompt }
      ],
      temperature: 0.1,
      max_tokens: 2000
    })
  });

  if (!res.ok) {
    const errText = await res.text();
    throw new Error(`Perplexity API error ${res.status}: ${errText}`);
  }

  const data = await res.json();
  const content = data.choices?.[0]?.message?.content;
  if (!content) throw new Error('Empty response from Perplexity');

  // Parse JSON from response (handle possible markdown code fences)
  const jsonStr = content.replace(/```json\n?/g, '').replace(/```\n?/g, '').trim();
  return JSON.parse(jsonStr);
}

// ── Main export ──
export async function getEmergingFactors({ lat, lon, location_name }) {
  const cacheKey = getCacheKey(lat, lon);

  // Check cache first
  const cached = getCached(cacheKey);
  if (cached) {
    return { ...cached, _cached: true };
  }

  let rawResult;
  const useMock = !process.env.PERPLEXITY_API_KEY;

  if (useMock) {
    // MOCK — remove once PERPLEXITY_API_KEY is set
    rawResult = getMockResponse(lat, lon, location_name);
  } else {
    // LIVE — Perplexity Sonar with web search
    const prompt = buildPrompt(location_name, lat, lon);
    rawResult = await callPerplexityAPI(prompt);
  }

  // Enforce filtering regardless of source (mock or live)
  const filteredFindings = filterFindings(rawResult.findings || []);

  const result = {
    emerging_factors: {
      location: rawResult.location || location_name,
      findings: filteredFindings,
      no_findings: filteredFindings.length === 0,
      search_radius_km: 50,
      data_source: useMock ? 'mock' : 'perplexity-sonar',
      queried_at: new Date().toISOString()
    },
    // Risk fields explicitly null — these models don't exist on this branch yet
    flood_risk: null,
    drought_risk: null
  };

  // Cache the result
  setCache(cacheKey, result);

  return result;
}
