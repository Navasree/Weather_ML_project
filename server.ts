import express from 'express';
import path from 'path';
import { createServer as createViteServer } from 'vite';
import { weatherService } from './server/weatherService.js';
import { processWeatherChat } from './server/geminiService.js';

async function startServer() {
  const app = express();
  const PORT = 3000;

  app.use(express.json());

  // API Routes
  app.get('/api/health', (req, res) => {
    res.json({
      status: 'ok',
      service: 'WeatherGPT Meteorological Intelligence Server',
      timestamp: new Date().toISOString(),
    });
  });

  // Location search (Geocoding)
  app.get('/api/location/search', async (req, res) => {
    try {
      const q = String(req.query.q || '');
      if (!q.trim()) {
        return res.json([]);
      }
      const results = await weatherService.searchLocation(q);
      res.json(results);
    } catch (err: any) {
      console.error('Error in /api/location/search:', err);
      res.status(500).json({ error: err.message || 'Geocoding service failure' });
    }
  });

  // Reverse geocoding
  app.get('/api/location/reverse', async (req, res) => {
    try {
      const lat = parseFloat(String(req.query.lat));
      const lon = parseFloat(String(req.query.lon));
      if (isNaN(lat) || isNaN(lon)) {
        return res.status(400).json({ error: 'Valid latitude and longitude required' });
      }
      const loc = await weatherService.reverseGeocode(lat, lon);
      res.json(loc);
    } catch (err: any) {
      console.error('Error in /api/location/reverse:', err);
      res.status(500).json({ error: err.message || 'Reverse geocoding failure' });
    }
  });

  // Full weather data (current + hourly + daily + alerts + AQI)
  app.get('/api/weather/current', async (req, res) => {
    try {
      const lat = parseFloat(String(req.query.lat || '19.076'));
      const lon = parseFloat(String(req.query.lon || '72.877'));
      const name = String(req.query.name || 'Mumbai');

      const data = await weatherService.getWeatherData(lat, lon, name);
      res.json(data);
    } catch (err: any) {
      console.error('Error in /api/weather/current:', err);
      res.status(500).json({ error: 'Weather data temporarily unavailable from provider. Please try again shortly.' });
    }
  });

  // Active weather alerts
  app.get('/api/weather/alerts', async (req, res) => {
    try {
      const lat = parseFloat(String(req.query.lat || '19.076'));
      const lon = parseFloat(String(req.query.lon || '72.877'));
      const name = String(req.query.name || 'Mumbai');

      const data = await weatherService.getWeatherData(lat, lon, name);
      res.json({
        location: name,
        alerts: data.alerts || [],
        source: 'Official Meteorological Alert Feeds (IMD/WMO Criteria)',
        timestamp: new Date().toISOString(),
      });
    } catch (err: any) {
      console.error('Error in /api/weather/alerts:', err);
      res.status(500).json({ error: 'Unable to retrieve alert feed' });
    }
  });

  // Radar & Satellite Map timestamps proxy
  let radarCache: { data: any; timestamp: number } | null = null;
  app.get('/api/weather/radar-maps', async (req, res) => {
    try {
      if (radarCache && Date.now() - radarCache.timestamp < 120000) {
        return res.json(radarCache.data);
      }
      const response = await fetch('https://api.rainviewer.com/public/weather-maps.json', {
        headers: { 'User-Agent': 'WeatherGPT/1.0' },
      });
      if (response.ok) {
        const data = await response.json();
        radarCache = { data, timestamp: Date.now() };
        return res.json(data);
      }
      throw new Error(`RainViewer API returned status ${response.status}`);
    } catch (err: any) {
      console.warn('Failed to fetch live RainViewer maps:', err.message);
      // Fallback response with approximate timestamps
      const nowSec = Math.floor(Date.now() / 1000);
      const rounded = Math.floor(nowSec / 600) * 600;
      res.json({
        host: 'https://tilecache.rainviewer.com',
        radar: {
          past: [{ time: rounded, path: `/v2/radar/${rounded}/256/{z}/{x}/{y}/2/1_1.png` }],
          nowcast: [],
        },
        satellite: {
          infrared: [{ time: rounded, path: `/v2/satellite/${rounded}/256/{z}/{x}/{y}/0/0_0.png` }],
        },
      });
    }
  });

  // Active tropical cyclone tracking bulletin
  app.get('/api/cyclones', async (req, res) => {
    try {
      const cyclones = await weatherService.getCycloneInfo();
      res.json(cyclones);
    } catch (err: any) {
      console.error('Error in /api/cyclones:', err);
      res.status(500).json({ error: 'Cyclone monitoring service unavailable' });
    }
  });

  // Historical climate analysis (10-year trends)
  app.get('/api/climate/trends', async (req, res) => {
    try {
      const lat = parseFloat(String(req.query.lat || '19.076'));
      const lon = parseFloat(String(req.query.lon || '72.877'));
      const name = String(req.query.name || 'Mumbai');

      const climateData = await weatherService.getHistoricalClimate(lat, lon, name);
      res.json(climateData);
    } catch (err: any) {
      console.error('Error in /api/climate/trends:', err);
      res.status(500).json({ error: 'Historical climate data unavailable' });
    }
  });

  // Agriculture advisory endpoint
  app.post('/api/advisory/agriculture', async (req, res) => {
    try {
      const { location = 'Mumbai', lat = 19.076, lon = 72.877, crop = 'Kharif Crops' } = req.body;
      const advisory = await weatherService.generateAgricultureAdvisory(location, lat, lon, crop);
      res.json(advisory);
    } catch (err: any) {
      console.error('Error in /api/advisory/agriculture:', err);
      res.status(500).json({ error: 'Agriculture advisory generation failed' });
    }
  });

  // Travel route weather endpoint
  app.post('/api/advisory/travel', async (req, res) => {
    try {
      const { origin = 'Mumbai', destination = 'Pune' } = req.body;
      const advisory = await weatherService.generateTravelAdvisory(origin, destination);
      res.json(advisory);
    } catch (err: any) {
      console.error('Error in /api/advisory/travel:', err);
      res.status(500).json({ error: 'Travel advisory generation failed' });
    }
  });

  // Conversational AI Weather Assistant (Gemini + Real-time Meteorological Tools)
  app.post('/api/chat', async (req, res) => {
    try {
      const { message, currentLocation, language, history } = req.body;
      if (!message || !message.trim()) {
        return res.status(400).json({ error: 'Message content is required' });
      }

      const result = await processWeatherChat({
        message,
        currentLocation,
        language,
        history,
      });

      res.json(result);
    } catch (err: any) {
      console.error('Error in /api/chat:', err);
      res.json({
        content:
          'Weather advisory telemetry is updating. Please try your question again in a moment or select another location.',
        toolCalls: [],
        sources: ['Real-time Meteorological Feed'],
      });
    }
  });

  // Machine Learning Service Health Check
  app.get('/api/ml/health', async (req, res) => {
    try {
      const mlHealth = await weatherService.checkMLHealth();
      if (mlHealth.available) {
        res.json({
          status: 'ok',
          service: 'WeatherGPT ML Service',
          details: mlHealth.status,
          timestamp: new Date().toISOString(),
        });
      } else {
        res.status(503).json({
          status: 'unavailable',
          service: 'WeatherGPT ML Service',
          error: 'Python ML service is currently unreachable on http://127.0.0.1:5001',
          details: mlHealth.status,
          timestamp: new Date().toISOString(),
        });
      }
    } catch (err: any) {
      res.status(503).json({
        status: 'unavailable',
        error: err.message || 'ML Health check failed',
      });
    }
  });

  // Weather Machine Learning Rain Prediction Route
  app.post('/api/weather/predict-rain', async (req, res) => {
    try {
      const body = req.body;
      if (!body || typeof body !== 'object') {
        return res.status(400).json({
          error: 'Request body must be a valid JSON object with weather parameters',
        });
      }

      const requiredFields = [
        'temp',
        'RH',
        'windSpeed',
        'gustSpeed',
        'DewPt',
        'windDirection',
        'PAR',
      ] as const;

      const missingFields: string[] = [];
      const invalidFields: string[] = [];
      const parsedValues: Record<string, number> = {};

      for (const field of requiredFields) {
        const val = body[field];
        if (val === undefined || val === null || val === '') {
          missingFields.push(field);
          continue;
        }

        const num = Number(val);
        if (isNaN(num)) {
          invalidFields.push(`${field} (received: ${JSON.stringify(val)})`);
        } else {
          parsedValues[field] = num;
        }
      }

      if (missingFields.length > 0) {
        return res.status(400).json({
          error: `Missing required parameter(s): ${missingFields.join(', ')}`,
          requiredFields,
        });
      }

      if (invalidFields.length > 0) {
        return res.status(400).json({
          error: `Invalid numeric value(s) for parameter(s): ${invalidFields.join(', ')}`,
          requiredFields,
        });
      }

      const prediction = await weatherService.predictRainML({
        temp: parsedValues.temp,
        RH: parsedValues.RH,
        windSpeed: parsedValues.windSpeed,
        gustSpeed: parsedValues.gustSpeed,
        DewPt: parsedValues.DewPt,
        windDirection: parsedValues.windDirection,
        PAR: parsedValues.PAR,
      });

      res.json(prediction);
    } catch (err: any) {
      console.error('Error in /api/weather/predict-rain:', err);
      const isUnavailable =
        err?.message?.includes('ML_API_UNAVAILABLE') ||
        err?.code === 'ECONNREFUSED' ||
        err?.name === 'AbortError';

      if (isUnavailable) {
        return res.status(503).json({
          error: 'Python ML service is unavailable. Please ensure the ML API is running on http://127.0.0.1:5001',
          details: err.message,
        });
      }

      res.status(500).json({
        error: 'Failed to process ML rain prediction',
        details: err.message,
      });
    }
  });

  // Vite middleware in development or static files in production
  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`WeatherGPT server running on http://0.0.0.0:${PORT}`);
  });
}

startServer();
