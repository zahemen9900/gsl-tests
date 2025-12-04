import React from 'react';
import styles from './ModeSelectionPage.module.css';

interface ModeSelectionPageProps {
  onSelectSign2Text: () => void;
  onBack: () => void;
}

const ModeSelectionPage: React.FC<ModeSelectionPageProps> = ({ onSelectSign2Text, onBack }) => {
  return (
    <div className={styles.container}>
      {/* Background Effects */}
      <div className={styles.backgroundGlow} />
      <div className={styles.gridOverlay} />
      
      {/* Animated Particles */}
      <div className={styles.particles}>
        {[...Array(15)].map((_, i) => (
          <div
            key={i}
            className={styles.particle}
            style={{
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
              animationDelay: `${Math.random() * 5}s`,
              animationDuration: `${3 + Math.random() * 4}s`,
            }}
          />
        ))}
      </div>

      {/* Back Button */}
      <button className={styles.backButton} onClick={onBack}>
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M19 12H5M12 19l-7-7 7-7" />
        </svg>
        <span>Back to Home</span>
      </button>

      {/* Main Content */}
      <div className={styles.content}>
        <div className={styles.header}>
          <div className={styles.badge}>
            <span className={styles.badgeDot} />
            Choose Your Mode
          </div>
          <h1 className={styles.title}>
            What would you like to
            <span className={styles.gradientText}> translate?</span>
          </h1>
          <p className={styles.subtitle}>
            Select a translation mode to get started. Sign2Text is ready now,
            with Text2Sign coming soon!
          </p>
        </div>

        <div className={styles.cardsContainer}>
          {/* Sign2Text Card */}
          <div className={styles.modeCard} onClick={onSelectSign2Text}>
            <div className={styles.cardGlow} />
            <div className={styles.cardContent}>
              <div className={styles.iconContainer}>
                <div className={styles.iconBg}>
                  <span className={styles.icon}>🤟</span>
                </div>
                <div className={styles.iconRing} />
              </div>
              
              <div className={styles.cardBadge}>
                <span className={styles.availableDot} />
                Available Now
              </div>
              
              <h2 className={styles.cardTitle}>Sign → Text</h2>
              <p className={styles.cardDescription}>
                Translate sign language gestures into written text in real-time using
                your webcam. Perfect for learning and practicing.
              </p>
              
              <div className={styles.featureList}>
                <div className={styles.feature}>
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <polyline points="20 6 9 17 4 12" />
                  </svg>
                  <span>Real-time webcam processing</span>
                </div>
                <div className={styles.feature}>
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <polyline points="20 6 9 17 4 12" />
                  </svg>
                  <span>Transformer-based AI model</span>
                </div>
                <div className={styles.feature}>
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <polyline points="20 6 9 17 4 12" />
                  </svg>
                  <span>Frame-by-frame visualization</span>
                </div>
              </div>
              
              <button className={styles.cardButton}>
                <span>Start Translating</span>
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M5 12h14M12 5l7 7-7 7" />
                </svg>
              </button>
            </div>
            
            {/* Animated wave decoration */}
            <div className={styles.waveDecoration}>
              <svg viewBox="0 0 200 60" preserveAspectRatio="none">
                <path
                  className={styles.wave1}
                  d="M0,30 Q25,10 50,30 T100,30 T150,30 T200,30 V60 H0 Z"
                />
                <path
                  className={styles.wave2}
                  d="M0,35 Q25,15 50,35 T100,35 T150,35 T200,35 V60 H0 Z"
                />
              </svg>
            </div>
          </div>

          {/* Text2Sign Card (Teaser) */}
          <div className={`${styles.modeCard} ${styles.teaserCard}`}>
            <div className={styles.cardGlow} />
            <div className={styles.cardContent}>
              <div className={styles.iconContainer}>
                <div className={`${styles.iconBg} ${styles.teaserIconBg}`}>
                  <span className={styles.icon}>✍️</span>
                </div>
                <div className={`${styles.iconRing} ${styles.teaserRing}`} />
              </div>
              
              <div className={`${styles.cardBadge} ${styles.teaserBadge}`}>
                <span className={styles.comingSoonDot} />
                Coming Soon
              </div>
              
              <h2 className={styles.cardTitle}>Text → Sign</h2>
              <p className={styles.cardDescription}>
                Convert written text into animated sign language. A 3D avatar will
                demonstrate the signs for you to learn and follow.
              </p>
              
              <div className={styles.featureList}>
                <div className={`${styles.feature} ${styles.teaserFeature}`}>
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <circle cx="12" cy="12" r="10" />
                    <polyline points="12 6 12 12 16 14" />
                  </svg>
                  <span>3D Avatar animation</span>
                </div>
                <div className={`${styles.feature} ${styles.teaserFeature}`}>
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <circle cx="12" cy="12" r="10" />
                    <polyline points="12 6 12 12 16 14" />
                  </svg>
                  <span>Sentence-to-sign synthesis</span>
                </div>
                <div className={`${styles.feature} ${styles.teaserFeature}`}>
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <circle cx="12" cy="12" r="10" />
                    <polyline points="12 6 12 12 16 14" />
                  </svg>
                  <span>Playback speed control</span>
                </div>
              </div>
              
              <button className={`${styles.cardButton} ${styles.teaserButton}`} disabled>
                <span>Notify Me</span>
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9" />
                  <path d="M13.73 21a2 2 0 0 1-3.46 0" />
                </svg>
              </button>
            </div>
            
            {/* Lock overlay */}
            <div className={styles.lockOverlay}>
              <div className={styles.lockIcon}>
                <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <rect x="3" y="11" width="18" height="11" rx="2" ry="2" />
                  <path d="M7 11V7a5 5 0 0 1 10 0v4" />
                </svg>
              </div>
            </div>
            
            {/* Animated wave decoration */}
            <div className={styles.waveDecoration}>
              <svg viewBox="0 0 200 60" preserveAspectRatio="none">
                <path
                  className={styles.wave1Teaser}
                  d="M0,30 Q25,10 50,30 T100,30 T150,30 T200,30 V60 H0 Z"
                />
                <path
                  className={styles.wave2Teaser}
                  d="M0,35 Q25,15 50,35 T100,35 T150,35 T200,35 V60 H0 Z"
                />
              </svg>
            </div>
          </div>
        </div>

        {/* Additional Info */}
        <div className={styles.infoSection}>
          <div className={styles.infoCard}>
            <div className={styles.infoIcon}>🎯</div>
            <h3>High Accuracy</h3>
            <p>74.7% top-1 accuracy with Transformer architecture</p>
          </div>
          <div className={styles.infoCard}>
            <div className={styles.infoIcon}>⚡</div>
            <h3>Real-Time</h3>
            <p>Under 100ms inference time per prediction</p>
          </div>
          <div className={styles.infoCard}>
            <div className={styles.infoIcon}>📊</div>
            <h3>Visual Feedback</h3>
            <p>See your captured frames in a live waveform</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ModeSelectionPage;
