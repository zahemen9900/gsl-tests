# GhSL Sign2Text: Frontend Application
**Version:** 1.0  
**Status:** ✅ Production-Ready  
**Last Updated:** December 2024

---

## Executive Summary

This document describes the **React Frontend Application** that provides the user interface for the GhSL Sign2Text Translation System. The frontend handles real-time webcam capture, MediaPipe pose extraction, visualization, and seamless communication with the backend inference API.

### Key Features

- **React 18** with TypeScript
- **Vite** for fast development and builds
- **MediaPipe Holistic** for real-time pose detection
- **ECharts** for high-performance waveform visualization
- **Mode Selection** for Sign2Text vs Text2Sign (teaser)
- **Real-time Streaming Inference** with motion gating
- **Skeleton Overlay** visualization toggle
- **Expandable Preview Modal** for detailed analysis

---

## 1. Application Architecture

### 1.1 Component Hierarchy

```
App
├── LandingPage
├── ModeSelectionPage
│   ├── Sign2Text Card
│   └── Text2Sign Card (Teaser)
└── InferenceTest
    ├── Header (Back, Title, Counter)
    ├── Main Layout
    │   ├── Left Panel (Reference Video)
    │   │   ├── Sample Video Player
    │   │   └── Navigation Buttons
    │   └── Right Panel (Camera)
    │       ├── Webcam + Skeleton Overlay
    │       ├── WaveformVisualizer
    │       ├── Real-time Controls
    │       ├── Recording Controls
    │       └── Result Card
    ├── Modal (Expanded Preview)
    └── Tips Footer
```

### 1.2 File Structure

```
frontend/
├── src/
│   ├── App.tsx                    # Main router component
│   ├── main.tsx                   # Entry point
│   ├── components/
│   │   ├── LandingPage.tsx        # Welcome screen
│   │   ├── ModeSelectionPage.tsx  # Sign2Text/Text2Sign choice
│   │   ├── InferenceTest.tsx      # Main inference UI
│   │   ├── InferenceTest.module.css
│   │   ├── WaveformVisualizer.tsx # Frame capture chart
│   │   └── WaveformVisualizer.module.css
│   └── utils/
│       └── mediapipe.ts           # Feature extraction utilities
├── public/
├── index.html
├── package.json
├── tsconfig.json
├── tsconfig.app.json
├── tsconfig.node.json
├── vite.config.ts
└── eslint.config.js
```

---

## 2. Core Components

### 2.1 App.tsx - Router

```tsx
type Page = 'landing' | 'mode-selection' | 'sign2text';

function App() {
  const [currentPage, setCurrentPage] = useState<Page>('landing');

  return (
    <div className="App">
      {currentPage === 'landing' && (
        <LandingPage onStart={() => setCurrentPage('mode-selection')} />
      )}
      {currentPage === 'mode-selection' && (
        <ModeSelectionPage 
          onSelectSign2Text={() => setCurrentPage('sign2text')} 
          onBack={() => setCurrentPage('landing')}
        />
      )}
      {currentPage === 'sign2text' && (
        <InferenceTest onBack={() => setCurrentPage('mode-selection')} />
      )}
    </div>
  );
}
```

### 2.2 ModeSelectionPage - Translation Direction Choice

```tsx
interface ModeSelectionPageProps {
  onSelectSign2Text: () => void;
  onBack: () => void;
}

const ModeSelectionPage: React.FC<ModeSelectionPageProps> = ({
  onSelectSign2Text,
  onBack,
}) => {
  return (
    <div className={styles.container}>
      <h1>Choose Translation Mode</h1>
      
      {/* Sign2Text Card - Active */}
      <div className={styles.card} onClick={onSelectSign2Text}>
        <span className={styles.icon}>🤟</span>
        <h2>Sign2Text</h2>
        <p>Translate sign language gestures to text</p>
        <span className={styles.badge}>Available Now</span>
      </div>
      
      {/* Text2Sign Card - Teaser */}
      <div className={`${styles.card} ${styles.disabled}`}>
        <span className={styles.icon}>✍️</span>
        <h2>Text2Sign</h2>
        <p>Generate sign language videos from text</p>
        <span className={styles.badge}>Coming Soon</span>
      </div>
      
      <button onClick={onBack}>← Back</button>
    </div>
  );
};
```

### 2.3 InferenceTest - Main Interface

The primary component handling webcam capture, inference, and results display.

**Key State:**
```tsx
// Samples from backend
const [samples, setSamples] = useState<SampleItem[]>([]);
const [currentSampleIndex, setCurrentSampleIndex] = useState(0);

// Recording state machine
const [status, setStatus] = useState<'idle' | 'countdown' | 'recording' | 'processing' | 'result'>('idle');
const [countdown, setCountdown] = useState(3);

// Inference results
const [prediction, setPrediction] = useState<string>('');
const [recordedFrames, setRecordedFrames] = useState<number[][]>([]);

// Streaming mode
const [realTimeEnabled, setRealTimeEnabled] = useState(false);
const [streamingPrediction, setStreamingPrediction] = useState('');

// Visualization
const [showLandmarks, setShowLandmarks] = useState(true);
const [allCapturedFrames, setAllCapturedFrames] = useState<number[][]>([]);
const [predictionWindowStart, setPredictionWindowStart] = useState(0);
const [predictionWindowEnd, setPredictionWindowEnd] = useState(0);

// Modal
const [isPanelModalOpen, setIsPanelModalOpen] = useState(false);
```

**Key Refs:**
```tsx
const webcamRef = useRef<Webcam>(null);
const canvasRef = useRef<HTMLCanvasElement>(null);          // Skeleton overlay
const modalCanvasRef = useRef<HTMLCanvasElement>(null);     // Modal skeleton overlay
const holisticRef = useRef<Holistic | null>(null);          // MediaPipe instance
const streamBufferRef = useRef<number[][]>([]);             // Streaming frame buffer
```

---

## 3. MediaPipe Integration

### 3.1 Holistic Initialization

```tsx
// mediapipe.ts
let holisticInstancePromise: Promise<Holistic> | null = null;

export function getHolisticInstance(): Promise<Holistic> {
  if (!holisticInstancePromise) {
    holisticInstancePromise = new Promise((resolve, reject) => {
      const holistic = new Holistic({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`
      });

      holistic.setOptions({
        modelComplexity: 1,
        smoothLandmarks: true,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5
      });

      holistic.initialize()
        .then(() => resolve(holistic))
        .catch((err) => {
          holisticInstancePromise = null;
          reject(err);
        });
    });
  }

  return holisticInstancePromise;
}
```

### 3.2 Feature Extraction

```tsx
// Constants matching training pipeline
const FACE_DOWNSAMPLE = 5;
const NUM_POSE_LANDMARKS = 33;
const NUM_HAND_LANDMARKS = 21;
const NUM_FACE_LANDMARKS = Math.ceil(468 / FACE_DOWNSAMPLE); // 94
const PER_FRAME_FEATURES = NUM_POSE_LANDMARKS * 4 + 2 * NUM_HAND_LANDMARKS * 3 + NUM_FACE_LANDMARKS * 3; // 540

export function extractFeatureVector(results: Results): Float32Array | null {
  const feats = new Float32Array(PER_FRAME_FEATURES);
  let offset = 0;

  // Pose (33 * 4 = 132)
  if (results.poseLandmarks) {
    for (const lm of results.poseLandmarks) {
      feats[offset++] = lm.x;
      feats[offset++] = lm.y;
      feats[offset++] = lm.z;
      feats[offset++] = lm.visibility || 0;
    }
  } else {
    offset += NUM_POSE_LANDMARKS * 4;
  }

  // Left Hand (21 * 3 = 63)
  if (results.leftHandLandmarks) {
    for (const lm of results.leftHandLandmarks) {
      feats[offset++] = lm.x;
      feats[offset++] = lm.y;
      feats[offset++] = lm.z;
    }
  } else {
    offset += NUM_HAND_LANDMARKS * 3;
  }

  // Right Hand (21 * 3 = 63)
  if (results.rightHandLandmarks) {
    for (const lm of results.rightHandLandmarks) {
      feats[offset++] = lm.x;
      feats[offset++] = lm.y;
      feats[offset++] = lm.z;
    }
  } else {
    offset += NUM_HAND_LANDMARKS * 3;
  }

  // Face downsampled (94 * 3 = 282)
  if (results.faceLandmarks) {
    for (let i = 0; i < 468; i += FACE_DOWNSAMPLE) {
      const lm = results.faceLandmarks[i];
      feats[offset++] = lm.x;
      feats[offset++] = lm.y;
      feats[offset++] = lm.z;
    }
  } else {
    offset += NUM_FACE_LANDMARKS * 3;
  }

  return normalizeFrameLandmarks(feats);
}
```

### 3.3 Torso-Centric Normalization

```tsx
function normalizeFrameLandmarks(feats: Float32Array): Float32Array {
  // Reference points
  const LEFT_SHOULDER = 11, RIGHT_SHOULDER = 12;
  const LEFT_HIP = 23, RIGHT_HIP = 24;
  
  // Extract torso points
  const getPosePt = (idx: number) => {
    const base = idx * 4;
    return { x: feats[base], y: feats[base + 1], z: feats[base + 2] };
  };
  
  const ls = getPosePt(LEFT_SHOULDER);
  const rs = getPosePt(RIGHT_SHOULDER);
  const lh = getPosePt(LEFT_HIP);
  const rh = getPosePt(RIGHT_HIP);
  
  // Compute torso center
  const torsoCenter = {
    x: (ls.x + rs.x + lh.x + rh.x) / 4,
    y: (ls.y + rs.y + lh.y + rh.y) / 4,
    z: (ls.z + rs.z + lh.z + rh.z) / 4
  };
  
  // Compute scale from torso dimensions
  const shoulderSpan = dist(ls, rs);
  const hipSpan = dist(lh, rh);
  const torsoHeight = dist(shoulderCenter, hipCenter);
  const scale = median([shoulderSpan, hipSpan, torsoHeight]);
  
  // Apply normalization to all landmarks
  // ... (see full implementation in mediapipe.ts)
}
```

---

## 4. Frame Processing Loop

### 4.1 Video Frame Processing

```tsx
const processVideo = useCallback(() => {
  if (webcamRef.current && webcamRef.current.video && holisticRef.current) {
    const video = webcamRef.current.video;
    if (video.readyState === 4) {
      holisticRef.current.send({ image: video });
    }
  }
  requestRef.current = requestAnimationFrame(processVideo);
}, []);

useEffect(() => {
  requestRef.current = requestAnimationFrame(processVideo);
  return () => {
    if (requestRef.current) cancelAnimationFrame(requestRef.current);
  };
}, [processVideo]);
```

### 4.2 Results Handler

```tsx
const onResults = useCallback((results: Results) => {
  // 1. Draw skeleton overlay (with mirroring)
  if (canvasRef.current && webcamRef.current?.video) {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (ctx && showLandmarks) {
      ctx.save();
      ctx.translate(canvas.width, 0);
      ctx.scale(-1, 1);
      drawLandmarks(ctx, results, canvas.width, canvas.height);
      ctx.restore();
    }
  }

  // 2. Extract features
  const features = extractFeatureVector(results);
  if (!features) return;
  const featureArray = Array.from(features);

  // 3. Always capture for visualization
  setAllCapturedFrames(prev => [...prev, featureArray]);

  // 4. Capture for recording if active
  if (statusRef.current === 'recording') {
    setRecordedFrames(prev => [...prev, featureArray]);
  }

  // 5. Handle streaming mode
  if (realTimeEnabledRef.current) {
    streamBufferRef.current.push(featureArray);
    if (streamBufferRef.current.length > STREAM_BUFFER_LIMIT) {
      streamBufferRef.current.splice(0, streamBufferRef.current.length - STREAM_BUFFER_LIMIT);
    }
    void sendStreamingInference();
  }
}, [sendStreamingInference, showLandmarks]);
```

---

## 5. Recording Flow

### 5.1 Recording Sequence

```tsx
const startRecordingSequence = () => {
  setStatus('countdown');
  setCountdown(3);
  setPrediction('');
  
  let count = 3;
  const timer = setInterval(() => {
    count--;
    setCountdown(count);
    if (count === 0) {
      clearInterval(timer);
      startRecording();
    }
  }, 1000);
};

const startRecording = () => {
  setStatus('recording');
  setRecordedFrames([]);
  
  // Mark prediction window start
  setPredictionWindowStart(allCapturedFrames.length);
  
  recordingStartRef.current = Date.now();
  recordTimeoutRef.current = setTimeout(() => {
    stopRecording();
  }, recordDurationMs);
};

const stopRecording = () => {
  setStatus('processing');
  setPredictionWindowEnd(allCapturedFrames.length);
  
  setTimeout(() => {
    sendInference();
  }, 100);
};
```

### 5.2 Inference Request

```tsx
const sendInference = useCallback(async () => {
  if (recordedFrames.length === 0) {
    setStatus('result');
    setPrediction("No frames captured. Please try again.");
    return;
  }

  try {
    const res = await fetch('http://localhost:8000/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ pose_data: recordedFrames })
    });
    const data = await res.json();
    setPrediction(data.prediction);
  } catch (err) {
    console.error(err);
    setPrediction("Error during inference. Please try again.");
  } finally {
    setStatus('result');
  }
}, [recordedFrames]);
```

---

## 6. Streaming Inference

### 6.1 Streaming Parameters

```tsx
const STREAM_WINDOW = 64;          // Frames per inference
const STREAM_BUFFER_LIMIT = 160;   // Max buffer size
const STREAM_COOLDOWN_MS = 2000;   // Min time between requests
```

### 6.2 Streaming Inference Handler

```tsx
const sendStreamingInference = useCallback(async () => {
  // Guard checks
  if (!realTimeEnabledRef.current) return;
  if (streamInFlightRef.current) return;
  if (streamBufferRef.current.length < STREAM_WINDOW) return;
  
  const now = Date.now();
  if (now - streamLastSentRef.current < STREAM_COOLDOWN_MS) return;

  streamInFlightRef.current = true;
  streamLastSentRef.current = now;
  setStreamingStatus('processing');
  
  // Take last 64 frames
  const payload = streamBufferRef.current.slice(-STREAM_WINDOW);
  
  // Update prediction window markers
  const endIdx = allCapturedFramesRef.current - 1;
  const startIdx = Math.max(0, endIdx - payload.length + 1);
  setPredictionWindowStart(startIdx);
  setPredictionWindowEnd(endIdx);

  try {
    const res = await fetch('http://localhost:8000/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ pose_data: payload })
    });
    const data = await res.json();
    setStreamingPrediction(data.prediction);
    setStreamingHistory(prev => [...prev.slice(-4), data.prediction]);
    setStreamingStatus('idle');
  } catch (err) {
    console.error(err);
    setStreamingStatus('error');
  } finally {
    streamInFlightRef.current = false;
  }
}, []);
```

---

## 7. Skeleton Overlay

### 7.1 Drawing Landmarks

```tsx
export function drawLandmarks(
  ctx: CanvasRenderingContext2D,
  results: Results,
  width: number,
  height: number
): void {
  ctx.save();
  // Mirror to match mirrored webcam
  ctx.translate(width, 0);
  ctx.scale(-1, 1);

  // Draw pose skeleton
  if (results.poseLandmarks) {
    drawConnections(ctx, results.poseLandmarks, POSE_CONNECTIONS, width, height, COLORS.pose.line, 3);
    drawPoints(ctx, results.poseLandmarks, width, height, COLORS.pose.point, 6);
  }

  // Draw left hand (green)
  if (results.leftHandLandmarks) {
    drawConnections(ctx, results.leftHandLandmarks, HAND_CONNECTIONS, width, height, COLORS.leftHand.line, 2);
    drawPoints(ctx, results.leftHandLandmarks, width, height, COLORS.leftHand.point, 4);
  }

  // Draw right hand (orange)
  if (results.rightHandLandmarks) {
    drawConnections(ctx, results.rightHandLandmarks, HAND_CONNECTIONS, width, height, COLORS.rightHand.line, 2);
    drawPoints(ctx, results.rightHandLandmarks, width, height, COLORS.rightHand.point, 4);
  }

  // Draw key face landmarks
  if (results.faceLandmarks) {
    const keyFaceIndices = [33, 133, 362, 263, /* ... */];
    const keyLandmarks = keyFaceIndices.map(i => results.faceLandmarks![i]);
    drawPoints(ctx, keyLandmarks, width, height, COLORS.face.point, 2);
  }

  ctx.restore();
}
```

### 7.2 Color Scheme

```tsx
const COLORS = {
  pose: { line: 'rgba(99, 102, 241, 0.8)', point: 'rgba(139, 92, 246, 1)' },      // Purple
  leftHand: { line: 'rgba(34, 197, 94, 0.8)', point: 'rgba(74, 222, 128, 1)' },   // Green
  rightHand: { line: 'rgba(249, 115, 22, 0.8)', point: 'rgba(251, 146, 60, 1)' }, // Orange
  face: { line: 'rgba(236, 72, 153, 0.4)', point: 'rgba(244, 114, 182, 0.6)' },   // Pink
};
```

---

## 8. Waveform Visualizer

### 8.1 Component Overview

High-performance motion energy visualization using Apache ECharts.

**Features:**
- Ring buffer for efficient memory management
- Throttled updates (~15 FPS) for smooth rendering
- Motion threshold line indicator
- Prediction window markers (green dots)
- Expandable/collapsible view
- Real-time metrics display

### 8.2 Ring Buffer Implementation

```tsx
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
}
```

### 8.3 Motion Energy Calculation

```tsx
function computeMotionEnergy(frames: number[][]): number[] {
  if (frames.length < 2) return [];
  
  const energies: number[] = [0];
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
```

### 8.4 ECharts Configuration

```tsx
const mainOption: echarts.EChartsOption = {
  animation: false,
  backgroundColor: 'transparent',
  grid: { left: 50, right: 20, top: 20, bottom: 40 },
  xAxis: {
    type: 'value',
    min: minTime,
    max: maxTime,
    axisLabel: { formatter: (v: number) => `${v.toFixed(1)}s` },
    name: 'Time (s)',
  },
  yAxis: {
    type: 'value',
    min: 0,
    name: 'Motion',
  },
  tooltip: { trigger: 'axis' },
  series: [
    {
      name: 'Motion Energy',
      type: 'line',
      data: fullData.map(([x, y]) => [x, y]),
      smooth: true,
      symbol: 'none',
      lineStyle: {
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
      itemStyle: { color: '#4ade80' },
    },
  ],
};
```

---

## 9. Modal Preview

### 9.1 Expanded View

```tsx
{isPanelModalOpen && (
  <div className={styles.modalOverlay}>
    <div className={styles.modalContent}>
      <div className={styles.modalHeader}>
        <div className={styles.modalTitle}>Expanded Preview</div>
        <div className={styles.modalHeaderActions}>
          <button onClick={() => setShowLandmarks(prev => !prev)}>
            {showLandmarks ? 'Skeleton On' : 'Skeleton Off'}
          </button>
          <button onClick={() => setIsPanelModalOpen(false)}>✕</button>
        </div>
      </div>
      
      <div className={styles.modalBody}>
        {/* Video Section */}
        <div className={styles.modalVideoSection}>
          <video ref={modalVideoRef} autoPlay muted playsInline />
          <canvas ref={modalCanvasRef} className={styles.modalOverlayCanvas} />
        </div>
        
        {/* Right Panel */}
        <div className={styles.modalRight}>
          <WaveformVisualizer frames={allCapturedFrames} {...props} />
          <div className={styles.modalPrediction}>
            <p>{streamingPrediction || prediction || 'Waiting...'}</p>
          </div>
        </div>
      </div>
    </div>
  </div>
)}
```

### 9.2 Modal Video Source

```tsx
useEffect(() => {
  if (!isPanelModalOpen) return;
  const videoEl = modalVideoRef.current;
  const src = webcamRef.current?.video?.srcObject as MediaStream | null;
  if (videoEl && src) {
    videoEl.srcObject = src;
  }
}, [isPanelModalOpen]);
```

---

## 10. API Communication

### 10.1 Backend Endpoints Used

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/samples` | GET | Fetch sample videos for practice |
| `/predict` | POST | Submit pose data for inference |
| `/videos/{filename}` | GET | Stream reference video |

### 10.2 API Requests

```tsx
// Fetch samples on mount
useEffect(() => {
  fetch('http://localhost:8000/samples')
    .then(res => res.json())
    .then(data => setSamples(data))
    .catch(err => setError("Could not connect to server"));
}, []);

// Send inference
const res = await fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ pose_data: recordedFrames })
});
```

---

## 11. Development

### 11.1 Setup

```bash
cd frontend
npm install
npm run dev
```

### 11.2 Build

```bash
npm run build
npm run preview  # Preview production build
```

### 11.3 Dependencies

```json
{
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-webcam": "^7.2.0",
    "@mediapipe/holistic": "^0.5.1675471629",
    "echarts": "^5.4.3",
    "echarts-for-react": "^3.0.2"
  },
  "devDependencies": {
    "typescript": "^5.3.0",
    "vite": "^5.0.0",
    "@vitejs/plugin-react": "^4.2.0"
  }
}
```

### 11.4 Environment

Create `.env` for backend URL:
```
VITE_API_BASE_URL=http://localhost:8000
```

---

## 12. UX Flow

### 12.1 User Journey

1. **Landing Page** → Click "Get Started"
2. **Mode Selection** → Choose "Sign2Text"
3. **Inference Test Page**
   - View reference video on left panel
   - Practice sign in front of webcam
   - Toggle skeleton overlay for feedback
   - Click "Start Recording" when ready
   - Wait for 3-2-1 countdown
   - Perform sign during recording period
   - View prediction result
   - Compare with expected sentence
   - Navigate to next sample or retry

### 12.2 Recording States

```
idle → countdown → recording → processing → result
  ↑                                           │
  └───────────────────────────────────────────┘
                  (Try Again)
```

### 12.3 Tips for Users

- Position hands and face clearly visible
- Good lighting improves detection accuracy
- Recording duration adapts to reference video length
- Real-time mode provides continuous feedback

---

## 13. Future Enhancements

- [ ] Video playback overlay comparison
- [ ] Recording history with scores
- [ ] Tutorial mode with guided practice
- [ ] Gesture-specific feedback
- [ ] Performance analytics dashboard
- [ ] Multi-language support
- [ ] Mobile responsive design
- [ ] PWA offline support
