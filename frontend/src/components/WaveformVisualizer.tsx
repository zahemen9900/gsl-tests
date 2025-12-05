import React, { useRef, useEffect, useMemo, useCallback, useState } from 'react';
import * as echarts from 'echarts';
import styles from './WaveformVisualizer.module.css';

// Ring buffer for efficient data management
class RingBuffer<T> {
  private buffer: T[];
  private head: number = 0;
  private tail: number = 0;
  private count: number = 0;
  
  constructor(private capacity: number) {
    this.buffer = new Array(capacity);
  }
  
  push(item: T): void {
    this.buffer[this.tail] = item;
    this.tail = (this.tail + 1) % this.capacity;
    
    if (this.count < this.capacity) {
      this.count++;
    } else {
      this.head = (this.head + 1) % this.capacity;
    }
  }
  
  toArray(): T[] {
    const result: T[] = [];
    for (let i = 0; i < this.count; i++) {
      result.push(this.buffer[(this.head + i) % this.capacity]);
    }
    return result;
  }
  
  get length(): number {
    return this.count;
  }
  
  clear(): void {
    this.head = 0;
    this.tail = 0;
    this.count = 0;
  }
}

// Simple stride downsampling (no-op now that minimap is removed) kept for future use
function strideDownsample<T>(data: T[], maxPoints: number): T[] {
  if (data.length <= maxPoints) return data;
  const step = Math.ceil(data.length / maxPoints);
  const result: T[] = [];
  for (let i = 0; i < data.length; i += step) {
    result.push(data[i]);
  }
  if (result[result.length - 1] !== data[data.length - 1]) {
    result.push(data[data.length - 1]);
  }
  return result;
}

// Throttle hook
function useThrottle<T extends (...args: unknown[]) => void>(
  callback: T,
  delay: number
): T {
  const lastCall = useRef(0);
  const lastCallTimer = useRef<ReturnType<typeof setTimeout>>();
  
  return useCallback((...args: Parameters<T>) => {
    const now = Date.now();
    const remaining = delay - (now - lastCall.current);
    
    if (remaining <= 0) {
      if (lastCallTimer.current) {
        clearTimeout(lastCallTimer.current);
        lastCallTimer.current = undefined;
      }
      lastCall.current = now;
      callback(...args);
    } else if (!lastCallTimer.current) {
      lastCallTimer.current = setTimeout(() => {
        lastCall.current = Date.now();
        lastCallTimer.current = undefined;
        callback(...args);
      }, remaining);
    }
  }, [callback, delay]) as T;
}

// Motion energy calculation
function computeMotionEnergy(frames: number[][]): number[] {
  if (frames.length < 2) return [];
  
  const energies: number[] = [0]; // First frame has 0 motion
  for (let i = 1; i < frames.length; i++) {
    let energy = 0;
    const prev = frames[i - 1];
    const curr = frames[i];
    for (let j = 0; j < Math.min(prev.length, curr.length); j++) {
      energy += Math.pow(curr[j] - prev[j], 2);
    }
    energies.push(Math.sqrt(energy));
  }
  return energies;
}

interface FrameData {
  timestamp: number;
  frameIndex: number;
  motionEnergy: number;
  isUsedForPrediction: boolean;
}

interface WaveformVisualizerProps {
  frames: number[][];
  streamWindowSize?: number;
  maxBufferSize?: number;
  isRecording?: boolean;
  isPredicting?: boolean;
  predictionWindowStart?: number;
  predictionWindowEnd?: number;
  motionThreshold?: number;
}

interface Metrics {
  totalFrames: number;
  usedFrames: number;
  avgMotion: number;
  maxMotion: number;
  minMotion: number;
  currentMotion: number;
  fps: number;
}

const WaveformVisualizer: React.FC<WaveformVisualizerProps> = ({
  frames,
  streamWindowSize = 64,
  maxBufferSize = 500,
  isRecording = false,
  isPredicting = false,
  predictionWindowStart = 0,
  predictionWindowEnd = 0,
  motionThreshold = 0.001,
}) => {
  const mainChartRef = useRef<HTMLDivElement>(null);
  const mainChartInstance = useRef<echarts.ECharts | null>(null);
  const dataBufferRef = useRef(new RingBuffer<FrameData>(maxBufferSize));
  const lastFrameTimeRef = useRef(Date.now());
  const frameCountRef = useRef(0);
  const fpsRef = useRef(0);
  const isActiveRef = useRef(true);
  const [isExpanded, setIsExpanded] = useState(false);
  const startTimeRef = useRef<number | null>(null);
  
  // Calculate metrics
  const metrics = useMemo<Metrics>(() => {
    const data = dataBufferRef.current.toArray();
    if (data.length === 0) {
      return {
        totalFrames: 0,
        usedFrames: 0,
        avgMotion: 0,
        maxMotion: 0,
        minMotion: 0,
        currentMotion: 0,
        fps: 0,
      };
    }
    
    const motionValues = data.map(d => d.motionEnergy);
    const usedFrames = data.filter(d => d.isUsedForPrediction).length;
    
    return {
      totalFrames: data.length,
      usedFrames,
      avgMotion: motionValues.reduce((a, b) => a + b, 0) / motionValues.length,
      maxMotion: Math.max(...motionValues),
      minMotion: Math.min(...motionValues),
      currentMotion: motionValues[motionValues.length - 1] || 0,
      fps: fpsRef.current,
    };
  }, [frames]);
  
  // Initialize charts
  useEffect(() => {
    isActiveRef.current = true;
    if (mainChartRef.current && !mainChartInstance.current) {
      mainChartInstance.current = echarts.init(mainChartRef.current, undefined, {
        renderer: 'canvas',
      });
    }
    
    // Handle resize
    const handleResize = () => {
      mainChartInstance.current?.resize();
    };
    window.addEventListener('resize', handleResize);
    
    return () => {
      isActiveRef.current = false;
      window.removeEventListener('resize', handleResize);
      if (mainChartInstance.current) {
        mainChartInstance.current.dispose();
        mainChartInstance.current = null;
      }
    };
  }, []);
  
  // Update data buffer when frames change
  useEffect(() => {
    if (frames.length === 0) {
      dataBufferRef.current.clear();
      startTimeRef.current = null;
      return;
    }
    
    // Calculate FPS
    const now = Date.now();
    frameCountRef.current++;
    if (now - lastFrameTimeRef.current >= 1000) {
      fpsRef.current = frameCountRef.current;
      frameCountRef.current = 0;
      lastFrameTimeRef.current = now;
    }
    
    // Compute motion energy
    const motionEnergies = computeMotionEnergy(frames);
    
    // Update buffer with new frame
    const lastFrameIndex = frames.length - 1;
    const isUsedForPrediction = lastFrameIndex >= predictionWindowStart && 
                                 lastFrameIndex <= predictionWindowEnd;
    
    if (startTimeRef.current === null) {
      startTimeRef.current = now;
    }

    dataBufferRef.current.push({
      timestamp: now,
      frameIndex: lastFrameIndex,
      motionEnergy: motionEnergies[lastFrameIndex] || 0,
      isUsedForPrediction,
    });
  }, [frames, predictionWindowStart, predictionWindowEnd]);

  const renderCharts = useCallback(() => {
    if (!isActiveRef.current) return;
    if (!mainChartInstance.current) return;
    const data = dataBufferRef.current.toArray();
    if (data.length === 0) return;

    const firstTimestamp = data[0].timestamp;
    const origin = startTimeRef.current ?? firstTimestamp;
    const fullData: Array<[number, number, number, boolean]> = data.map((d) => [
      (d.timestamp - origin) / 1000,
      d.motionEnergy,
      d.frameIndex,
      d.isUsedForPrediction,
    ]);

    const predictionData = fullData.filter(([, , , used]) => used).map(([x, y]) => [x, y]);

    const minTime = fullData[0]?.[0] ?? 0;
    const maxTime = fullData[fullData.length - 1]?.[0] ?? 0;

    const mainOption: echarts.EChartsOption = {
      animation: false,
      backgroundColor: 'transparent',
      grid: {
        left: 50,
        right: 20,
        top: 20,
        bottom: 40,
      },
      xAxis: {
        type: 'value',
        min: minTime,
        max: maxTime,
        axisLine: { lineStyle: { color: 'rgba(255,255,255,0.2)' } },
        axisLabel: { 
          color: 'rgba(255,255,255,0.5)', 
          fontSize: 10,
          formatter: (v: number) => `${v.toFixed(1)}s`,
        },
        splitLine: { lineStyle: { color: 'rgba(255,255,255,0.05)' } },
        name: 'Time (s)',
        nameTextStyle: { color: 'rgba(255,255,255,0.5)', fontSize: 10 },
      },
      yAxis: {
        type: 'value',
        min: 0,
        axisLine: { lineStyle: { color: 'rgba(255,255,255,0.2)' } },
        axisLabel: { 
          color: 'rgba(255,255,255,0.5)', 
          fontSize: 10,
          formatter: (v: number) => v.toFixed(3)
        },
        splitLine: { lineStyle: { color: 'rgba(255,255,255,0.05)' } },
        name: 'Motion',
        nameTextStyle: { color: 'rgba(255,255,255,0.5)', fontSize: 10 },
      },
      tooltip: {
        trigger: 'axis',
        backgroundColor: 'rgba(20, 20, 30, 0.9)',
        borderColor: 'rgba(99, 102, 241, 0.5)',
        textStyle: { color: 'white', fontSize: 12 },
        formatter: (params: any) => {
          const items = Array.isArray(params) ? params : [params];
          const p = items[0];
          if (!p || typeof p.value === 'undefined') return '';
          const val = Array.isArray(p.value) ? p.value[1] : p.value;
          const time = Array.isArray(p.value) ? p.value[0] : p.dataIndex;
          return `Time: ${Number(time).toFixed(2)}s<br/>Motion: ${Number(val).toFixed(5)}`;
        },
      },
      series: [
        {
          name: 'Motion Energy',
          type: 'line',
          data: fullData.map(([x, y]) => [x, y]),
          smooth: true,
          symbol: 'none',
          lineStyle: {
            width: 2,
            color: new echarts.graphic.LinearGradient(0, 0, 1, 0, [
              { offset: 0, color: 'rgba(59, 130, 246, 0.8)' },
              { offset: 0.5, color: 'rgba(99, 102, 241, 0.8)' },
              { offset: 1, color: 'rgba(168, 85, 247, 0.8)' },
            ]),
          },
          areaStyle: {
            color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
              { offset: 0, color: 'rgba(99, 102, 241, 0.3)' },
              { offset: 1, color: 'rgba(99, 102, 241, 0.02)' },
            ]),
          },
        },
        {
          name: 'Prediction Window',
          type: 'scatter',
          data: predictionData,
          symbol: 'circle',
          symbolSize: 6,
          itemStyle: {
            color: '#4ade80',
            borderColor: '#22c55e',
            borderWidth: 1,
          },
        },
        {
          name: 'Threshold',
          type: 'line',
          markLine: {
            silent: true,
            symbol: 'none',
            lineStyle: { 
              color: 'rgba(239, 68, 68, 0.5)', 
              type: 'dashed',
              width: 1,
            },
            data: [{ yAxis: motionThreshold }],
            label: {
              formatter: 'Threshold',
              color: 'rgba(239, 68, 68, 0.7)',
              fontSize: 10,
            },
          },
        },
      ],
    };

    mainChartInstance.current.setOption(mainOption, true);
  }, [motionThreshold]);

  // Recompute flags when prediction window changes so existing points reflect latest window
  useEffect(() => {
    const items = dataBufferRef.current.toArray();
    if (items.length === 0) return;
    const rebuilt = items.map(item => ({
      ...item,
      isUsedForPrediction: item.frameIndex >= predictionWindowStart && item.frameIndex <= predictionWindowEnd,
    }));
    dataBufferRef.current.clear();
    rebuilt.forEach(entry => dataBufferRef.current.push(entry));
    renderCharts();
  }, [predictionWindowStart, predictionWindowEnd, renderCharts]);

  const updateCharts = useThrottle(() => {
    renderCharts();
  }, 66);
  
  // Trigger chart updates
  useEffect(() => {
    updateCharts();
  }, [frames, updateCharts]);

  // Force refresh on expand/collapse
  useEffect(() => {
    if (!mainChartInstance.current) return;
    requestAnimationFrame(() => {
      mainChartInstance.current?.resize();
      renderCharts();
    });
  }, [isExpanded, renderCharts]);
  
  const containerClassName = [
    styles.container,
    isExpanded ? styles.expanded : styles.collapsed,
  ].join(' ');

  return (
    <div className={containerClassName}>
      {/* Header */}
      <div className={styles.header}>
        <div className={styles.titleSection}>
          <h3 className={styles.title}>
            <span className={styles.titleIcon}>📊</span>
            Frame Capture Waveform
          </h3>
          <div className={styles.statusBadges}>
            {isRecording && (
              <span className={`${styles.badge} ${styles.recordingBadge}`}>
                <span className={styles.recordingDot} />
                Recording
              </span>
            )}
            {isPredicting && (
              <span className={`${styles.badge} ${styles.predictingBadge}`}>
                <span className={styles.predictingDot} />
                Processing
              </span>
            )}
          </div>
          {!isExpanded && (
            <span className={styles.previewLabel}>Preview</span>
          )}
        </div>
        
        <div className={styles.headerActions}>
          {/* Live Metrics */}
          <div className={styles.metricsPanel}>
            <div className={styles.metric}>
              <span className={styles.metricLabel}>Frames</span>
              <span className={styles.metricValue}>{metrics.totalFrames}</span>
            </div>
            <div className={styles.metricDivider} />
            <div className={styles.metric}>
              <span className={styles.metricLabel}>Used</span>
              <span className={`${styles.metricValue} ${styles.highlight}`}>{metrics.usedFrames}</span>
            </div>
            <div className={styles.metricDivider} />
            <div className={styles.metric}>
              <span className={styles.metricLabel}>FPS</span>
              <span className={styles.metricValue}>{metrics.fps}</span>
            </div>
            <div className={styles.metricDivider} />
            <div className={styles.metric}>
              <span className={styles.metricLabel}>Avg Motion</span>
              <span className={styles.metricValue}>{metrics.avgMotion.toFixed(4)}</span>
            </div>
            <div className={styles.metricDivider} />
            <div className={styles.metric}>
              <span className={styles.metricLabel}>Current</span>
              <span className={`${styles.metricValue} ${metrics.currentMotion > motionThreshold ? styles.motionActive : styles.motionIdle}`}>
                {metrics.currentMotion.toFixed(4)}
              </span>
            </div>
          </div>

          <button
            type="button"
            className={styles.expandButton}
            onClick={() => setIsExpanded(prev => !prev)}
            aria-label={isExpanded ? 'Collapse waveform panel' : 'Expand waveform panel'}
          >
            <svg
              width="16"
              height="16"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              {isExpanded ? (
                <>
                  <polyline points="9 9 9 5 5 5" />
                  <polyline points="15 9 19 9 19 5" />
                  <polyline points="15 15 19 15 19 19" />
                  <polyline points="9 15 9 19 5 19" />
                </>
              ) : (
                <>
                  <polyline points="15 5 19 5 19 9" />
                  <polyline points="9 5 5 5 5 9" />
                  <polyline points="19 15 19 19 15 19" />
                  <polyline points="5 15 5 19 9 19" />
                </>
              )}
            </svg>
          </button>
        </div>
      </div>
      
      {/* Main Chart */}
      <div className={styles.mainChartContainer}>
        <div ref={mainChartRef} className={styles.mainChart} />
        
        {/* Gradient overlays for visual polish */}
        <div className={styles.chartGlowLeft} />
        <div className={styles.chartGlowRight} />
      </div>
      
      {/* Legend */}
      <div className={styles.legend}>
        <div className={styles.legendItem}>
          <span className={styles.legendDot} style={{ background: 'linear-gradient(90deg, #3b82f6, #6366f1, #a855f7)' }} />
          <span>Motion Energy</span>
        </div>
        <div className={styles.legendItem}>
          <span className={styles.legendDot} style={{ background: '#4ade80' }} />
          <span>Prediction Window</span>
        </div>
        <div className={styles.legendItem}>
          <span className={styles.legendLine} />
          <span>Motion Threshold</span>
        </div>
      </div>
    </div>
  );
};

export default WaveformVisualizer;
