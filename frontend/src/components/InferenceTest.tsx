import React, { useState, useEffect, useRef, useCallback } from 'react';
import Webcam from 'react-webcam';
import type { Holistic, Results } from '@mediapipe/holistic';
import { extractFeatureVector, getHolisticInstance, drawLandmarks } from '../utils/mediapipe';
import WaveformVisualizer from './WaveformVisualizer';
import styles from './InferenceTest.module.css';

interface SampleItem {
    video_file: string;
    sentence: string;
    feature_path: string;
}

interface InferenceTestProps {
    onBack: () => void;
}

const DEFAULT_RECORDING_MS = 3000;
const STREAM_WINDOW = 64;
const STREAM_BUFFER_LIMIT = 160;
const STREAM_COOLDOWN_MS = 2000;

const InferenceTest: React.FC<InferenceTestProps> = ({ onBack }) => {
    const [samples, setSamples] = useState<SampleItem[]>([]);
    const [currentSampleIndex, setCurrentSampleIndex] = useState(0);
    const [status, setStatus] = useState<'idle' | 'countdown' | 'recording' | 'processing' | 'result'>('idle');
    const [countdown, setCountdown] = useState(3);
    const [prediction, setPrediction] = useState<string>('');
    const [recordedFrames, setRecordedFrames] = useState<number[][]>([]);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [pipelineError, setPipelineError] = useState<string | null>(null);
    const [recordingProgress, setRecordingProgress] = useState(0);
    const [recordDurationMs, setRecordDurationMs] = useState(DEFAULT_RECORDING_MS);
    const [realTimeEnabled, setRealTimeEnabled] = useState(false);
    const [streamingPrediction, setStreamingPrediction] = useState('');
    const [streamingStatus, setStreamingStatus] = useState<'idle' | 'processing' | 'error'>('idle');
    const [streamingHistory, setStreamingHistory] = useState<string[]>([]);
    const [showLandmarks, setShowLandmarks] = useState(true);
    const [allCapturedFrames, setAllCapturedFrames] = useState<number[][]>([]);
    const [predictionWindowStart, setPredictionWindowStart] = useState(0);
    const [predictionWindowEnd, setPredictionWindowEnd] = useState(0);
    const [isPanelModalOpen, setIsPanelModalOpen] = useState(false);
    const modalVideoRef = useRef<HTMLVideoElement>(null);
    const allCapturedFramesRef = useRef(0);
    
    const webcamRef = useRef<Webcam>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const holisticRef = useRef<Holistic | null>(null);
    const requestRef = useRef<number>();
    const recordingStartRef = useRef<number>(0);
    const recordTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
    const statusRef = useRef(status);
    const realTimeEnabledRef = useRef(realTimeEnabled);
    const streamBufferRef = useRef<number[][]>([]);
    const streamInFlightRef = useRef(false);
    const streamLastSentRef = useRef(0);
    const videoDurationRef = useRef<HTMLVideoElement | null>(null);

    const currentSample = samples[currentSampleIndex] || null;
    const recordingSeconds = Math.max(1, Math.round(recordDurationMs / 1000));

    useEffect(() => {
        statusRef.current = status;
    }, [status]);

    useEffect(() => {
        setRecordDurationMs(DEFAULT_RECORDING_MS);
    }, [currentSample?.video_file]);

    useEffect(() => {
        realTimeEnabledRef.current = realTimeEnabled;
        if (!realTimeEnabled) {
            streamBufferRef.current = [];
            setStreamingPrediction('');
            setStreamingHistory([]);
            setStreamingStatus('idle');
        }
    }, [realTimeEnabled]);

    useEffect(() => {
        allCapturedFramesRef.current = allCapturedFrames.length;
    }, [allCapturedFrames]);

    useEffect(() => {
        fetch('http://localhost:8000/samples')
            .then(res => {
                if (!res.ok) throw new Error('Failed to fetch samples');
                return res.json();
            })
            .then(data => {
                setSamples(data);
                setIsLoading(false);
            })
            .catch(err => {
                console.error("Failed to load samples", err);
                setError("Could not connect to server. Make sure the backend is running.");
                setIsLoading(false);
            });
    }, []);

    const sendStreamingInference = useCallback(async () => {
        if (!realTimeEnabledRef.current) return;
        if (streamInFlightRef.current) return;
        if (streamBufferRef.current.length < STREAM_WINDOW) return;
        const now = Date.now();
        if (now - streamLastSentRef.current < STREAM_COOLDOWN_MS) return;

        streamInFlightRef.current = true;
        streamLastSentRef.current = now;
        setStreamingStatus('processing');
        const payload = streamBufferRef.current.slice(-STREAM_WINDOW);
        const endIdx = allCapturedFramesRef.current - 1;
        const startIdx = Math.max(0, endIdx - payload.length + 1);
        setPredictionWindowStart(startIdx);
        setPredictionWindowEnd(Math.max(startIdx, endIdx));

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

    const onResults = useCallback((results: Results) => {
        // Draw landmarks overlay
        if (canvasRef.current && webcamRef.current?.video) {
            const video = webcamRef.current.video;
            const canvas = canvasRef.current;
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const ctx = canvas.getContext('2d');
            if (ctx) {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                if (showLandmarks) {
                    drawLandmarks(ctx, results, canvas.width, canvas.height);
                }
            }
        }

        const features = extractFeatureVector(results);
        if (!features) return;
        const featureArray = Array.from(features);

        // Always capture frames for visualization
        setAllCapturedFrames(prev => {
            const next = [...prev, featureArray];
            allCapturedFramesRef.current = next.length;
            return next;
        });

        if (statusRef.current === 'recording') {
            setRecordedFrames(prev => [...prev, featureArray]);
        }

        if (realTimeEnabledRef.current) {
            streamBufferRef.current.push(featureArray);
            if (streamBufferRef.current.length > STREAM_BUFFER_LIMIT) {
                streamBufferRef.current.splice(0, streamBufferRef.current.length - STREAM_BUFFER_LIMIT);
            }
            void sendStreamingInference();
        }
    }, [sendStreamingInference, showLandmarks]);

    useEffect(() => {
        let isMounted = true;

        getHolisticInstance()
            .then(instance => {
                if (!isMounted) return;
                holisticRef.current = instance;
                instance.onResults(onResults);
                setPipelineError(null);
            })
            .catch(err => {
                console.error('Failed to initialize MediaPipe Holistic', err);
                if (isMounted) {
                    setPipelineError('Unable to initialize landmark tracking. Close other tabs and reload this page.');
                }
            });

        return () => {
            isMounted = false;
            if (holisticRef.current) {
                holisticRef.current.onResults(() => {});
            }
        };
    }, [onResults]);

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

    useEffect(() => {
        if (!isPanelModalOpen) return;
        const videoEl = modalVideoRef.current;
        const src = webcamRef.current?.video?.srcObject as MediaStream | null | undefined;
        if (videoEl && src) {
            videoEl.srcObject = src;
        }
    }, [isPanelModalOpen]);

    useEffect(() => {
        return () => {
            if (recordTimeoutRef.current) {
                clearTimeout(recordTimeoutRef.current);
            }
        };
    }, []);

    // Recording progress update
    useEffect(() => {
        if (status === 'recording') {
            const interval = setInterval(() => {
                const elapsed = Date.now() - recordingStartRef.current;
                const progress = Math.min((elapsed / recordDurationMs) * 100, 100);
                setRecordingProgress(progress);
            }, 50);
            return () => clearInterval(interval);
        } else {
            setRecordingProgress(0);
        }
    }, [status, recordDurationMs]);

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
        const currentFrameCount = allCapturedFrames.length;
        setPredictionWindowStart(currentFrameCount);
        setPredictionWindowEnd(currentFrameCount);
        recordingStartRef.current = Date.now();
        if (recordTimeoutRef.current) {
            clearTimeout(recordTimeoutRef.current);
        }
        recordTimeoutRef.current = setTimeout(() => {
            stopRecording();
        }, recordDurationMs);
    };

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

    const stopRecording = () => {
        setStatus('processing');
        setPredictionWindowEnd(allCapturedFrames.length);
        if (recordTimeoutRef.current) {
            clearTimeout(recordTimeoutRef.current);
            recordTimeoutRef.current = null;
        }
        setTimeout(() => {
            sendInference();
        }, 100);
    };

    const handleVideoMetadata = (event: React.SyntheticEvent<HTMLVideoElement>) => {
        videoDurationRef.current = event.currentTarget;
        const duration = event.currentTarget.duration;
        if (!Number.isFinite(duration) || duration <= 0) {
            setRecordDurationMs(DEFAULT_RECORDING_MS);
            return;
        }
        setRecordDurationMs(Math.round((duration + 3) * 1000));
    };

    const toggleRealTime = () => {
        if (pipelineError) {
            return;
        }
        setRealTimeEnabled((prev) => !prev);
        setStreamingPrediction('');
    };

    const nextSample = () => {
        setCurrentSampleIndex((prev) => (prev + 1) % samples.length);
        setStatus('idle');
        setPrediction('');
    };

    const prevSample = () => {
        setCurrentSampleIndex((prev) => (prev - 1 + samples.length) % samples.length);
        setStatus('idle');
        setPrediction('');
    };

    if (isLoading) {
        return (
            <div className={styles.loadingContainer}>
                <div className={styles.spinner} />
                <p>Loading samples...</p>
            </div>
        );
    }

    if (error) {
        return (
            <div className={styles.errorContainer}>
                <div className={styles.errorIcon}>⚠️</div>
                <h2>Connection Error</h2>
                <p>{error}</p>
                <button className={styles.backButton} onClick={onBack}>
                    ← Back to Home
                </button>
            </div>
        );
    }

    return (
        <div className={styles.container}>
            {/* Header */}
            <header className={styles.header}>
                <button className={styles.backButton} onClick={onBack}>
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M19 12H5M12 19l-7-7 7-7"/>
                    </svg>
                    <span>Back</span>
                </button>
                <h1 className={styles.title}>
                    <span className={styles.titleIcon}>🤟</span>
                    Sign Language Test
                </h1>
                <div className={styles.sampleCounter}>
                    Sample {currentSampleIndex + 1} of {samples.length}
                </div>
            </header>

            {/* Main Content */}
            <main className={styles.main}>
                {/* Left Panel - Sample Video */}
                <section className={styles.panel}>
                    <div className={styles.panelHeader}>
                        <div className={styles.panelIcon}>📹</div>
                        <div>
                            <h2>Reference Video</h2>
                            <p>Watch and learn the sign</p>
                        </div>
                    </div>
                    
                    <div className={styles.targetSentence}>
                        <span className={styles.label}>Target Sentence</span>
                        <p className={styles.sentence}>
                            {currentSample ? `"${currentSample.sentence}"` : 'No sample available'}
                        </p>
                    </div>

                    <div className={styles.videoContainer}>
                        {currentSample ? (
                            <video
                                key={currentSample.video_file}
                                src={`http://localhost:8000/videos/${currentSample.video_file}`}
                                controls
                                className={styles.video}
                                onLoadedMetadata={handleVideoMetadata}
                            />
                        ) : (
                            <div className={styles.videoPlaceholder}>
                                <p>No video available</p>
                            </div>
                        )}
                    </div>

                    <div className={styles.navigationButtons}>
                        <button className={styles.navButton} onClick={prevSample}>
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <path d="M15 18l-6-6 6-6"/>
                            </svg>
                            Previous
                        </button>
                        <button className={styles.navButton} onClick={nextSample}>
                            Next
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <path d="M9 18l6-6-6-6"/>
                            </svg>
                        </button>
                    </div>
                </section>

                {/* Right Panel - Camera */}
                <section className={styles.panel}>
                    <div className={`${styles.panelHeader} ${styles.panelHeaderWithAction}`}>
                        <div className={styles.panelHeaderInfo}>
                            <div className={styles.panelIcon}>🎥</div>
                            <div>
                                <h2>Your Camera</h2>
                                <p>Perform the sign yourself</p>
                            </div>
                        </div>
                        <div className={styles.panelActions}>
                            <button
                                className={styles.expandPanelButton}
                                onClick={() => setIsPanelModalOpen(true)}
                                title="Expand preview"
                                type="button"
                            >
                                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                    <polyline points="15 3 21 3 21 9" />
                                    <polyline points="9 21 3 21 3 15" />
                                    <line x1="21" y1="3" x2="14" y2="10" />
                                    <line x1="3" y1="21" x2="10" y2="14" />
                                </svg>
                            </button>
                            <button
                                className={`${styles.landmarkToggle} ${showLandmarks ? styles.landmarkToggleActive : ''}`}
                                onClick={() => setShowLandmarks(prev => !prev)}
                                title={showLandmarks ? 'Hide skeleton overlay' : 'Show skeleton overlay'}
                            >
                                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                    <circle cx="12" cy="5" r="3"/>
                                    <line x1="12" y1="8" x2="12" y2="16"/>
                                    <line x1="12" y1="12" x2="8" y2="10"/>
                                    <line x1="12" y1="12" x2="16" y2="10"/>
                                    <line x1="12" y1="16" x2="9" y2="21"/>
                                    <line x1="12" y1="16" x2="15" y2="21"/>
                                </svg>
                                <span>{showLandmarks ? 'Skeleton On' : 'Skeleton Off'}</span>
                            </button>
                        </div>
                    </div>

                    <div className={styles.cameraContainer}>
                        <Webcam
                            ref={webcamRef}
                            className={styles.webcam}
                            mirrored
                            videoConstraints={{
                                width: 640,
                                height: 480,
                                facingMode: "user"
                            }}
                        />
                        <canvas
                            ref={canvasRef}
                            className={`${styles.overlayCanvas} ${showLandmarks ? styles.overlayVisible : ''}`}
                        />
                        
                        {/* Countdown Overlay */}
                        {status === 'countdown' && (
                            <div className={styles.countdownOverlay}>
                                <div className={styles.countdownNumber} key={countdown}>
                                    {countdown}
                                </div>
                                <p>Get ready...</p>
                            </div>
                        )}
                        
                        {/* Recording Indicator */}
                        {status === 'recording' && (
                            <div className={styles.recordingOverlay}>
                                <div className={styles.recordingIndicator}>
                                    <div className={styles.recordingDot} />
                                    <span>Recording</span>
                                </div>
                                <div className={styles.progressBar}>
                                    <div 
                                        className={styles.progressFill} 
                                        style={{ width: `${recordingProgress}%` }}
                                    />
                                </div>
                            </div>
                        )}
                        
                        {/* Processing Overlay */}
                        {status === 'processing' && (
                            <div className={styles.processingOverlay}>
                                <div className={styles.processingSpinner} />
                                <p>Analyzing your sign...</p>
                            </div>
                        )}

                        {pipelineError && (
                            <div className={styles.pipelineErrorOverlay}>
                                <h3>Tracking unavailable</h3>
                                <p>{pipelineError}</p>
                                <button
                                    onClick={() => {
                                        if (typeof window !== 'undefined') {
                                            window.location.reload();
                                        }
                                    }}
                                >
                                    Reload Page
                                </button>
                            </div>
                        )}
                    </div>

                    {/* Waveform Visualizer */}
                    <WaveformVisualizer
                        frames={allCapturedFrames}
                        streamWindowSize={STREAM_WINDOW}
                        maxBufferSize={500}
                        isRecording={status === 'recording'}
                        isPredicting={status === 'processing'}
                        predictionWindowStart={predictionWindowStart}
                        predictionWindowEnd={predictionWindowEnd}
                        motionThreshold={0.001}
                    />

                    <div className={styles.realtimeControls}>
                        <div className={styles.realtimeHeader}>
                            <div>
                                <h3>Real-time Inference</h3>
                                <p>Stream quick predictions while practicing.</p>
                            </div>
                            <button
                                className={`${styles.toggleButton} ${realTimeEnabled ? styles.toggleButtonActive : ''}`}
                                onClick={toggleRealTime}
                                disabled={Boolean(pipelineError)}
                            >
                                {realTimeEnabled ? 'Disable' : 'Enable'}
                            </button>
                        </div>
                        <div className={styles.realtimeBody}>
                            <div className={styles.streamingStatus}>
                                Status: {streamingStatus === 'processing' ? 'Processing latest window…' : streamingStatus === 'error' ? 'Streaming error - check network' : 'Idle'}
                            </div>
                            {realTimeEnabled && (
                                <div className={styles.streamingFeed}>
                                    <div className={styles.streamingPrediction}>
                                        <span className={styles.streamingLabel}>Current</span>
                                        <p>{streamingPrediction || 'Waiting for movement…'}</p>
                                    </div>
                                    <div className={styles.streamingHistory}>
                                        <span className={styles.streamingLabel}>Recent</span>
                                        {streamingHistory.length === 0 ? (
                                            <p className={styles.historyPlaceholder}>No predictions yet</p>
                                        ) : (
                                            streamingHistory.slice().reverse().map((item, index) => (
                                                <span key={`${item}-${index}`}>{item}</span>
                                            ))
                                        )}
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>

                    {/* Controls */}
                    <div className={styles.controls}>
                        {(status === 'idle' || status === 'result') && (
                            <button 
                                className={styles.recordButton}
                                onClick={startRecordingSequence}
                                disabled={Boolean(pipelineError)}
                            >
                                <div className={styles.recordButtonInner}>
                                    {status === 'result' ? '↻' : '●'}
                                </div>
                                <span>{status === 'result' ? 'Try Again' : 'Start Recording'}</span>
                            </button>
                        )}
                        
                        {status === 'countdown' && (
                            <div className={styles.statusText}>
                                <span className={styles.statusDot} /> Preparing...
                            </div>
                        )}
                        
                        {status === 'recording' && (
                            <div className={styles.statusText}>
                                <span className={`${styles.statusDot} ${styles.recording}`} /> Recording in progress...
                            </div>
                        )}
                        
                        {status === 'processing' && (
                            <div className={styles.statusText}>
                                <span className={styles.statusDot} /> Processing...
                            </div>
                        )}
                    </div>

                    {/* Result */}
                    {status === 'result' && prediction && (
                        <div className={styles.resultCard}>
                            <div className={styles.resultHeader}>
                                <span className={styles.resultIcon}>✨</span>
                                <span>Translation Result</span>
                            </div>
                            <p className={styles.prediction}>{prediction}</p>
                            {currentSample && (
                                <div className={styles.comparison}>
                                    <div className={styles.comparisonItem}>
                                        <span className={styles.comparisonLabel}>Expected</span>
                                        <span className={styles.comparisonText}>{currentSample.sentence}</span>
                                    </div>
                                    <div className={styles.comparisonItem}>
                                        <span className={styles.comparisonLabel}>Predicted</span>
                                        <span className={styles.comparisonText}>{prediction}</span>
                                    </div>
                                </div>
                            )}
                        </div>
                    )}
                </section>
            </main>

            {isPanelModalOpen && (
                <div className={styles.modalOverlay}>
                    <div className={styles.modalContent}>
                        <div className={styles.modalHeader}>
                            <div className={styles.modalTitle}>Expanded Preview</div>
                            <button
                                className={styles.modalClose}
                                onClick={() => setIsPanelModalOpen(false)}
                                aria-label="Close expanded preview"
                            >
                                ✕
                            </button>
                        </div>
                        <div className={styles.modalBody}>
                            <div className={styles.modalVideoSection}>
                                <video
                                    ref={modalVideoRef}
                                    className={styles.modalVideo}
                                    autoPlay
                                    muted
                                    playsInline
                                />
                            </div>
                            <div className={styles.modalRight}>
                                <div className={styles.modalWaveformStack}>
                                    <WaveformVisualizer
                                        frames={allCapturedFrames}
                                        streamWindowSize={STREAM_WINDOW}
                                        maxBufferSize={500}
                                        isRecording={status === 'recording'}
                                        isPredicting={status === 'processing'}
                                        predictionWindowStart={predictionWindowStart}
                                        predictionWindowEnd={predictionWindowEnd}
                                        motionThreshold={0.001}
                                    />
                                </div>
                                <div className={styles.modalPrediction}>
                                    <div className={styles.modalPredictionHeader}>Prediction</div>
                                    <p className={styles.modalPredictionText}>
                                        {streamingPrediction || prediction || 'Waiting for inference...'}
                                    </p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {/* Tips */}
            <div className={styles.tips}>
                <div className={styles.tip}>
                    <span className={styles.tipIcon}>💡</span>
                    <span>Position yourself so your hands and face are clearly visible</span>
                </div>
                <div className={styles.tip}>
                    <span className={styles.tipIcon}>🎯</span>
                    <span>Good lighting improves detection accuracy</span>
                </div>
                <div className={styles.tip}>
                    <span className={styles.tipIcon}>⏱️</span>
                    <span>Recording lasts about {recordingSeconds} seconds for this sample</span>
                </div>
            </div>
        </div>
    );
};

export default InferenceTest;
