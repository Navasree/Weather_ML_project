import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Brain,
  CloudRain,
  Sun,
  Loader2,
  AlertCircle,
  RefreshCw,
  Sparkles,
  Database,
  MapPin,
  SlidersHorizontal,
} from 'lucide-react';
import { CurrentWeather, MLRainPredictionResponse, MLRainPredictionRequest, WeatherLocation } from '../../types';

interface AiRainPredictionCardProps {
  currentWeather: CurrentWeather;
  locationName: string;
  onSelectIyerpadiLocation?: () => void;
}

// Authentic meteorological scenarios from ml/dataset/01_Iyerpadi_Weather_Data.csv
export interface IyerpadiScenario {
  id: string;
  label: string;
  description: string;
  expectedOutcome: 'Rain' | 'No Rain';
  params: MLRainPredictionRequest;
}

export const IYERPADI_SCENARIOS: IyerpadiScenario[] = [
  {
    id: 'iyerpadi-monsoon',
    label: 'Iyerpadi Monsoon Rain Event',
    description: 'High humidity, dense cloud cover & low PAR recorded at Iyerpadi station',
    expectedOutcome: 'Rain',
    params: {
      temp: 19.82,
      RH: 96.1,
      windSpeed: 0.76,
      gustSpeed: 2.77,
      DewPt: 19.2,
      windDirection: 64.6,
      PAR: 1.2,
    },
  },
  {
    id: 'iyerpadi-sunny',
    label: 'Iyerpadi Clear / Sunny Afternoon',
    description: 'High solar PAR radiation & moderate relative humidity recorded at Iyerpadi',
    expectedOutcome: 'No Rain',
    params: {
      temp: 22.97,
      RH: 68.9,
      windSpeed: 0.76,
      gustSpeed: 3.02,
      DewPt: 17.0,
      windDirection: 40.7,
      PAR: 1818.7,
    },
  },
  {
    id: 'iyerpadi-overcast',
    label: 'Iyerpadi Overcast Shower Onset',
    description: 'Near-saturation RH & atmospheric cooling recorded at Iyerpadi station',
    expectedOutcome: 'Rain',
    params: {
      temp: 20.13,
      RH: 99.0,
      windSpeed: 0.0,
      gustSpeed: 1.76,
      DewPt: 20.0,
      windDirection: 259.7,
      PAR: 13.7,
    },
  },
  {
    id: 'iyerpadi-dry-morning',
    label: 'Iyerpadi Dry Hill Morning',
    description: 'Calm winds, crisp dew point & dry hill atmosphere at Iyerpadi tea estate',
    expectedOutcome: 'No Rain',
    params: {
      temp: 19.58,
      RH: 67.1,
      windSpeed: 0.0,
      gustSpeed: 0.5,
      DewPt: 13.3,
      windDirection: 37.9,
      PAR: 1.2,
    },
  },
];

export const AiRainPredictionCard: React.FC<AiRainPredictionCardProps> = ({
  currentWeather,
  locationName,
  onSelectIyerpadiLocation,
}) => {
  // Mode selection: 'iyerpadi-mode' (default benchmark) or 'live-transfer'
  const [activeMode, setActiveMode] = useState<'iyerpadi-mode' | 'live-transfer'>('iyerpadi-mode');
  const [selectedScenarioId, setSelectedScenarioId] = useState<string>('iyerpadi-monsoon');
  const [prediction, setPrediction] = useState<MLRainPredictionResponse | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [showTelemetryDetails, setShowTelemetryDetails] = useState<boolean>(false);

  // Memoized signature to avoid repeated calls / infinite React useEffect loops
  const lastFetchedSignatureRef = useRef<string>('');

  const { temperature, relativeHumidity, windSpeed, windGusts, windDirection } = currentWeather;

  const fetchPrediction = useCallback(
    async (force = false) => {
      let payload: MLRainPredictionRequest;
      let signature = '';

      if (activeMode === 'iyerpadi-mode') {
        const scenario = IYERPADI_SCENARIOS.find((s) => s.id === selectedScenarioId) || IYERPADI_SCENARIOS[0];
        payload = scenario.params;
        signature = `iyerpadi_${scenario.id}`;
      } else {
        // Live Telemetry Mode
        const temp = temperature;
        const RH = relativeHumidity;
        const gustSpeed = windGusts ?? windSpeed;
        const DewPt = Number((temp - (100 - RH) / 5).toFixed(2));
        const PAR = 100.0; // Documented neutral baseline for Open-Meteo transfer

        payload = {
          temp,
          RH,
          windSpeed,
          gustSpeed,
          DewPt,
          windDirection,
          PAR,
        };
        signature = `live_${locationName}_${temp}_${RH}_${windSpeed}_${gustSpeed}_${windDirection}`;
      }

      if (!force && signature === lastFetchedSignatureRef.current) {
        return;
      }

      setIsLoading(true);
      setError(null);

      try {
        const res = await fetch('/api/weather/predict-rain', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(payload),
        });

        if (!res.ok) {
          const errorData = await res.json().catch(() => ({}));
          throw new Error(errorData.error || `Server responded with status ${res.status}`);
        }

        const data: MLRainPredictionResponse = await res.json();
        setPrediction(data);
        lastFetchedSignatureRef.current = signature;
      } catch (err: any) {
        console.warn('AI Rain Prediction unavailable:', err?.message);
        setError(
          err?.message?.includes('Python ML service is unavailable')
            ? 'ML prediction engine is currently offline. Please ensure the Python ML service is running.'
            : err?.message || 'Unable to compute ML prediction.'
        );
      } finally {
        setIsLoading(false);
      }
    },
    [activeMode, selectedScenarioId, temperature, relativeHumidity, windSpeed, windGusts, windDirection, locationName]
  );

  // Request prediction on mount and whenever mode/scenario/weather changes
  useEffect(() => {
    fetchPrediction();
  }, [fetchPrediction]);

  const activeScenario = IYERPADI_SCENARIOS.find((s) => s.id === selectedScenarioId) || IYERPADI_SCENARIOS[0];
  const activeParams = activeMode === 'iyerpadi-mode'
    ? activeScenario.params
    : {
        temp: temperature,
        RH: relativeHumidity,
        DewPt: Number((temperature - (100 - relativeHumidity) / 5).toFixed(1)),
        windSpeed,
        gustSpeed: windGusts ?? windSpeed,
        windDirection,
        PAR: 100.0,
      };

  return (
    <div className="bg-[#111111] p-5 sm:p-6 rounded-2xl border border-white/10 shadow-xl transition-all hover:border-white/20 relative overflow-hidden space-y-4">
      {/* Top Header with Explicit Location Label */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-white/10">
        <div>
          <div className="flex items-center gap-2">
            <div className="w-7 h-7 rounded-xl bg-cyan-500/10 border border-cyan-500/20 flex items-center justify-center text-cyan-400">
              <Brain className="w-3.5 h-3.5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 shadow-[0_0_8px_rgba(6,182,212,0.6)] animate-pulse" />
                <h3 className="font-mono text-sm text-white uppercase tracking-wider font-semibold">
                  AI Rain Prediction
                </h3>
              </div>
              <p className="text-xs font-mono text-cyan-400 font-medium tracking-wide mt-0.5 flex items-center gap-1">
                <MapPin className="w-3 h-3 text-cyan-400" />
                Location: Iyerpadi, Tamil Nadu
              </p>
            </div>
          </div>
        </div>

        {/* Mode Selector and Refresh */}
        <div className="flex items-center gap-2 self-start sm:self-auto">
          <span className="px-2.5 py-1 rounded-full text-[10px] font-mono font-medium border bg-cyan-500/10 text-cyan-300 border-cyan-500/30">
            Iyerpadi ML Mode
          </span>
          <button
            onClick={() => fetchPrediction(true)}
            disabled={isLoading}
            title="Re-run ML prediction"
            className="p-1.5 rounded-lg text-gray-400 hover:text-white hover:bg-white/5 border border-transparent hover:border-white/10 transition-all disabled:opacity-50"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${isLoading ? 'animate-spin text-cyan-400' : ''}`} />
          </button>
        </div>
      </div>

      {/* Dataset & Location Distinction Info Box */}
      <div className="bg-white/[0.02] border border-white/10 rounded-xl p-3.5 space-y-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-1.5 text-cyan-400 font-mono text-xs font-semibold">
            <Database className="w-3.5 h-3.5" />
            <span>ML Training Dataset: Iyerpadi Weather Data</span>
          </div>
          <span className="text-[10px] font-mono text-gray-500">
            Coimbatore Dist., TN
          </span>
        </div>

        <p className="text-[11px] text-gray-300 font-sans leading-relaxed">
          The Random Forest classifier was trained strictly on <strong>315,496 historical ground-station records from Iyerpadi, Tamil Nadu</strong>.
        </p>

        {/* Explicit Distinction Callout */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 pt-1 text-[10px] font-mono border-t border-white/5">
          <div className="flex items-center gap-1.5 text-gray-300 bg-white/[0.03] px-2 py-1 rounded-lg">
            <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />
            <span><strong>ML Training:</strong> Iyerpadi Dataset</span>
          </div>
          <div className="flex items-center gap-1.5 text-gray-300 bg-white/[0.03] px-2 py-1 rounded-lg">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
            <span><strong>Live WeatherGPT Data:</strong> Open-Meteo</span>
          </div>
        </div>
      </div>

      {/* Mode Navigation Tabs */}
      <div className="flex gap-2 p-1 bg-white/[0.03] rounded-xl border border-white/5 text-xs font-mono">
        <button
          onClick={() => setActiveMode('iyerpadi-mode')}
          className={`flex-1 py-1.5 px-3 rounded-lg text-center transition-all ${
            activeMode === 'iyerpadi-mode'
              ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30 font-medium shadow-sm'
              : 'text-gray-400 hover:text-white'
          }`}
        >
          🌲 Iyerpadi ML Mode (Benchmark)
        </button>
        <button
          onClick={() => setActiveMode('live-transfer')}
          className={`flex-1 py-1.5 px-3 rounded-lg text-center transition-all ${
            activeMode === 'live-transfer'
              ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30 font-medium shadow-sm'
              : 'text-gray-400 hover:text-white'
          }`}
        >
          📍 Live {locationName} Transfer
        </button>
      </div>

      {/* Scenario Selector when in Iyerpadi Mode */}
      {activeMode === 'iyerpadi-mode' ? (
        <div className="space-y-2">
          <div className="flex items-center justify-between text-[11px] font-mono text-gray-400">
            <span className="flex items-center gap-1">
              <SlidersHorizontal className="w-3 h-3 text-cyan-400" />
              Iyerpadi Weather Scenario (Dataset Samples):
            </span>
          </div>
          <div className="grid grid-cols-2 gap-2">
            {IYERPADI_SCENARIOS.map((sc) => (
              <button
                key={sc.id}
                onClick={() => setSelectedScenarioId(sc.id)}
                className={`p-2.5 rounded-xl border text-left transition-all text-xs ${
                  selectedScenarioId === sc.id
                    ? 'bg-cyan-950/40 border-cyan-500/40 text-white shadow-sm'
                    : 'bg-white/[0.02] border-white/10 text-gray-400 hover:border-white/20 hover:text-gray-300'
                }`}
              >
                <div className="flex items-center justify-between mb-1">
                  <span className="font-semibold truncate">{sc.label}</span>
                  <span
                    className={`text-[9px] font-mono px-1.5 py-0.5 rounded ${
                      sc.expectedOutcome === 'Rain'
                        ? 'bg-blue-500/20 text-blue-300'
                        : 'bg-emerald-500/20 text-emerald-300'
                    }`}
                  >
                    {sc.expectedOutcome}
                  </span>
                </div>
                <p className="text-[10px] text-gray-400 line-clamp-1">
                  Temp {sc.params.temp}°C • RH {sc.params.RH}% • PAR {sc.params.PAR}
                </p>
              </button>
            ))}
          </div>
        </div>
      ) : (
        <div className="bg-amber-500/10 border border-amber-500/20 rounded-xl p-3 text-xs text-amber-300 space-y-1">
          <p className="font-medium flex items-center gap-1.5">
            <AlertCircle className="w-3.5 h-3.5" />
            Cross-Regional Model Evaluation
          </p>
          <p className="text-[11px] text-amber-200/80 leading-relaxed font-sans">
            Evaluating live <strong>{locationName}</strong> weather through the Random Forest classifier trained on <strong>Iyerpadi, Tamil Nadu</strong>.
            Please note: {locationName} weather is NOT Iyerpadi weather.
          </p>
        </div>
      )}

      {/* Main Prediction Display Card */}
      <div>
        {isLoading && !prediction ? (
          <div className="flex flex-col items-center justify-center py-7 gap-3 bg-white/[0.02] rounded-xl border border-white/5">
            <Loader2 className="w-6 h-6 text-cyan-400 animate-spin" />
            <p className="font-mono text-xs text-gray-400">
              Querying Random Forest model for {activeMode === 'iyerpadi-mode' ? 'Iyerpadi, Tamil Nadu' : locationName}...
            </p>
          </div>
        ) : error ? (
          <div className="bg-amber-500/10 border border-amber-500/20 rounded-xl p-4 text-center space-y-2">
            <div className="flex items-center justify-center gap-1.5 text-amber-400 text-xs font-medium">
              <AlertCircle className="w-4 h-4" />
              <span>Prediction Service Offline</span>
            </div>
            <p className="text-[11px] text-gray-400 font-sans leading-relaxed">
              {error}
            </p>
            <button
              onClick={() => fetchPrediction(true)}
              className="mt-2 text-xs font-mono text-amber-300 hover:text-white underline underline-offset-4 transition-colors"
            >
              Retry Connection
            </button>
          </div>
        ) : prediction ? (
          <div className="space-y-4">
            {/* Primary Result Box */}
            <div className="flex items-center justify-between bg-white/[0.03] p-4 rounded-xl border border-white/10">
              <div className="flex items-center gap-3">
                <div
                  className={`w-12 h-12 rounded-xl flex items-center justify-center border ${
                    prediction.prediction === 'Rain'
                      ? 'bg-blue-500/15 border-blue-500/30 text-blue-400 shadow-[0_0_12px_rgba(59,130,246,0.3)]'
                      : 'bg-emerald-500/15 border-emerald-500/30 text-emerald-400 shadow-[0_0_12px_rgba(16,185,129,0.3)]'
                  }`}
                >
                  {prediction.prediction === 'Rain' ? (
                    <CloudRain className="w-6 h-6 animate-pulse" />
                  ) : (
                    <Sun className="w-6 h-6" />
                  )}
                </div>
                <div>
                  <span className="font-mono text-[9px] text-gray-400 uppercase tracking-widest block">
                    RANDOM FOREST CLASSIFICATION
                  </span>
                  <div className="flex items-baseline gap-2">
                    <span className="text-sm font-mono text-gray-400">Prediction:</span>
                    <span
                      className={`text-2xl font-bold tracking-tight font-serif ${
                        prediction.prediction === 'Rain' ? 'text-blue-300' : 'text-emerald-300'
                      }`}
                    >
                      {prediction.prediction}
                    </span>
                  </div>
                </div>
              </div>

              <div className="text-right">
                <span className="font-mono text-[9px] text-gray-400 uppercase tracking-widest block">
                  CONFIDENCE
                </span>
                <span className="text-xl font-light text-white font-mono">
                  {prediction.probability}%
                </span>
              </div>
            </div>

            {/* Probability Breakdown Dual Bar */}
            <div className="space-y-2.5 bg-white/[0.02] p-3.5 rounded-xl border border-white/5">
              <div className="flex flex-col sm:flex-row justify-between sm:items-center text-xs font-mono gap-1.5">
                <span className="flex items-center gap-1.5 text-blue-300 font-medium">
                  <span className="w-2 h-2 rounded-full bg-blue-400 inline-block shadow-[0_0_6px_rgba(96,165,250,0.6)]" />
                  Rain Probability: {prediction.rain_probability}%
                </span>
                <span className="flex items-center gap-1.5 text-emerald-300 font-medium">
                  <span className="w-2 h-2 rounded-full bg-emerald-400 inline-block shadow-[0_0_6px_rgba(52,211,153,0.6)]" />
                  No Rain Probability: {prediction.no_rain_probability}%
                </span>
              </div>

              {/* Progress bar split */}
              <div className="h-2.5 w-full bg-white/5 rounded-full overflow-hidden flex border border-white/10">
                <div
                  className="bg-gradient-to-r from-blue-600 to-cyan-400 transition-all duration-700 ease-out"
                  style={{ width: `${Math.max(prediction.rain_probability, 0)}%` }}
                  title={`Rain probability: ${prediction.rain_probability}%`}
                />
                <div
                  className="bg-gradient-to-r from-emerald-500 to-teal-400 transition-all duration-700 ease-out"
                  style={{ width: `${Math.max(prediction.no_rain_probability, 0)}%` }}
                  title={`No Rain probability: ${prediction.no_rain_probability}%`}
                />
              </div>

              {/* Model Credit Label */}
              <div className="flex items-center justify-between pt-1 text-[11px] font-mono text-gray-400">
                <span className="text-cyan-400 font-medium">
                  Powered by Random Forest ML
                </span>
                <span className="text-gray-400 text-[10px]">
                  Target: {activeMode === 'iyerpadi-mode' ? 'Iyerpadi, TN' : locationName}
                </span>
              </div>
            </div>

            {/* Features Used for Inference */}
            <div className="pt-1 border-t border-white/5">
              <button
                onClick={() => setShowTelemetryDetails(!showTelemetryDetails)}
                className="w-full flex items-center justify-between text-[10px] font-mono text-gray-400 hover:text-gray-300 transition-colors py-1"
              >
                <span className="flex items-center gap-1">
                  <Sparkles className="w-3 h-3 text-cyan-400" />
                  Features Passed to weather_model.pkl
                </span>
                <span>{showTelemetryDetails ? '▲ Hide Features' : '▼ View Features'}</span>
              </button>

              {showTelemetryDetails && (
                <div className="mt-2 grid grid-cols-2 sm:grid-cols-4 gap-2 text-[10px] font-mono bg-white/[0.02] p-2.5 rounded-xl border border-white/10 text-gray-300">
                  <div>
                    <span className="text-gray-500 block">temp:</span>
                    <span>{activeParams.temp}°C</span>
                  </div>
                  <div>
                    <span className="text-gray-500 block">RH:</span>
                    <span>{activeParams.RH}%</span>
                  </div>
                  <div>
                    <span className="text-gray-500 block">DewPt:</span>
                    <span>{activeParams.DewPt}°C</span>
                  </div>
                  <div>
                    <span className="text-gray-500 block">windSpeed:</span>
                    <span>{activeParams.windSpeed} km/h</span>
                  </div>
                  <div>
                    <span className="text-gray-500 block">gustSpeed:</span>
                    <span>{activeParams.gustSpeed} km/h</span>
                  </div>
                  <div>
                    <span className="text-gray-500 block">windDirection:</span>
                    <span>{activeParams.windDirection}°</span>
                  </div>
                  <div className="col-span-2">
                    <span className="text-gray-500 block">PAR (Radiation):</span>
                    <span className="text-cyan-400">{activeParams.PAR} W/m²</span>
                  </div>
                </div>
              )}
            </div>
          </div>
        ) : null}
      </div>

      {/* Switch Dashboard to Iyerpadi action if not already there */}
      {onSelectIyerpadiLocation && locationName !== 'Iyerpadi' && (
        <div className="pt-1 border-t border-white/5 text-center">
          <button
            onClick={onSelectIyerpadiLocation}
            className="text-[11px] font-mono text-gray-400 hover:text-cyan-300 transition-colors inline-flex items-center gap-1"
          >
            <span>👉 Switch main dashboard to live Iyerpadi weather</span>
          </button>
        </div>
      )}
    </div>
  );
};
