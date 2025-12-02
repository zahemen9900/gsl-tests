import React from 'react';
import styles from './LandingPage.module.css';

interface LandingPageProps {
  onStart: () => void;
}

const LandingPage: React.FC<LandingPageProps> = ({ onStart }) => {
  return (
    <div className={styles.container}>
      {/* Background Elements */}
      <div className={styles.backgroundGlow} />
      <div className={styles.gridOverlay} />
      
      {/* Floating particles */}
      <div className={styles.particles}>
        {[...Array(20)].map((_, i) => (
          <div key={i} className={styles.particle} style={{
            left: `${Math.random() * 100}%`,
            top: `${Math.random() * 100}%`,
            animationDelay: `${Math.random() * 5}s`,
            animationDuration: `${3 + Math.random() * 4}s`
          }} />
        ))}
      </div>

      {/* Navigation */}
      <nav className={styles.nav}>
        <div className={styles.logo}>
          <span className={styles.logoIcon}>🤟</span>
          <span className={styles.logoText}>SignTalk</span>
        </div>
        <div className={styles.navLinks}>
          <a href="#features" className={styles.navLink}>Features</a>
          <a href="#how-it-works" className={styles.navLink}>How it Works</a>
          <a href="#about" className={styles.navLink}>About</a>
        </div>
      </nav>

      {/* Hero Section */}
      <section className={styles.hero}>
        <div className={styles.heroContent}>
          <div className={styles.badge}>
            <span className={styles.badgeDot} />
            AI-Powered Translation
          </div>
          <h1 className={styles.title}>
            Breaking Barriers with
            <span className={styles.gradientText}> Sign Language AI</span>
          </h1>
          <p className={styles.subtitle}>
            Experience real-time Greek Sign Language translation powered by 
            state-of-the-art deep learning. Simply sign, and watch your 
            gestures transform into text instantly.
          </p>
          <div className={styles.ctaGroup}>
            <button className={styles.primaryButton} onClick={onStart}>
              <span>Start Translating</span>
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M5 12h14M12 5l7 7-7 7"/>
              </svg>
            </button>
            <button className={styles.secondaryButton}>
              <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                <polygon points="5 3 19 12 5 21 5 3"/>
              </svg>
              <span>Watch Demo</span>
            </button>
          </div>
          
          {/* Stats */}
          <div className={styles.stats}>
            <div className={styles.stat}>
              <span className={styles.statNumber}>540</span>
              <span className={styles.statLabel}>Pose Features</span>
            </div>
            <div className={styles.statDivider} />
            <div className={styles.stat}>
              <span className={styles.statNumber}>74%</span>
              <span className={styles.statLabel}>Top-1 Accuracy</span>
            </div>
            <div className={styles.statDivider} />
            <div className={styles.stat}>
              <span className={styles.statNumber}>&lt;100ms</span>
              <span className={styles.statLabel}>Inference Time</span>
            </div>
          </div>
        </div>

        {/* Hero Visual */}
        <div className={styles.heroVisual}>
          <div className={styles.visualCard}>
            <div className={styles.visualHeader}>
              <div className={styles.windowControls}>
                <span /><span /><span />
              </div>
              <span className={styles.windowTitle}>Live Translation</span>
            </div>
            <div className={styles.visualContent}>
              <div className={styles.mockCamera}>
                <div className={styles.cameraPlaceholder}>
                  <div className={styles.handIcon}>👋</div>
                  <p>Camera Preview</p>
                </div>
              </div>
              <div className={styles.mockOutput}>
                <div className={styles.outputLabel}>Translation</div>
                <div className={styles.outputText}>"Καλημέρα, πώς είσαι;"</div>
                <div className={styles.confidence}>
                  <div className={styles.confidenceBar} />
                  <span>92% confidence</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className={styles.features}>
        <h2 className={styles.sectionTitle}>Powerful Features</h2>
        <p className={styles.sectionSubtitle}>
          Built with cutting-edge technology for accurate and fast translation
        </p>
        <div className={styles.featureGrid}>
          <div className={styles.featureCard}>
            <div className={styles.featureIcon}>🎯</div>
            <h3>Real-Time Detection</h3>
            <p>MediaPipe Holistic captures 540 pose features per frame for precise hand, face, and body tracking.</p>
          </div>
          <div className={styles.featureCard}>
            <div className={styles.featureIcon}>🧠</div>
            <h3>Transformer Architecture</h3>
            <p>State-of-the-art sequence-to-sequence model with attention mechanism for context-aware translation.</p>
          </div>
          <div className={styles.featureCard}>
            <div className={styles.featureIcon}>⚡</div>
            <h3>Streaming Inference</h3>
            <p>Motion-gated processing ensures translations happen only when meaningful gestures are detected.</p>
          </div>
          <div className={styles.featureCard}>
            <div className={styles.featureIcon}>📊</div>
            <h3>Quality Metrics</h3>
            <p>Built-in WER and BLEU scoring to continuously evaluate and improve translation accuracy.</p>
          </div>
        </div>
      </section>

      {/* How it Works */}
      <section id="how-it-works" className={styles.howItWorks}>
        <h2 className={styles.sectionTitle}>How It Works</h2>
        <p className={styles.sectionSubtitle}>
          Three simple steps to translate sign language
        </p>
        <div className={styles.steps}>
          <div className={styles.step}>
            <div className={styles.stepNumber}>1</div>
            <div className={styles.stepContent}>
              <h3>Watch the Sample</h3>
              <p>View a reference video showing the target phrase in Greek Sign Language.</p>
            </div>
          </div>
          <div className={styles.stepConnector} />
          <div className={styles.step}>
            <div className={styles.stepNumber}>2</div>
            <div className={styles.stepContent}>
              <h3>Record Your Sign</h3>
              <p>Use your webcam to perform the sign. Our AI captures every movement.</p>
            </div>
          </div>
          <div className={styles.stepConnector} />
          <div className={styles.step}>
            <div className={styles.stepNumber}>3</div>
            <div className={styles.stepContent}>
              <h3>Get Translation</h3>
              <p>Receive instant text translation with confidence scores.</p>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className={styles.ctaSection}>
        <div className={styles.ctaCard}>
          <h2>Ready to Experience Sign Language AI?</h2>
          <p>Start translating Greek Sign Language in real-time with our interactive demo.</p>
          <button className={styles.primaryButton} onClick={onStart}>
            <span>Launch Demo</span>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M5 12h14M12 5l7 7-7 7"/>
            </svg>
          </button>
        </div>
      </section>

      {/* Footer */}
      <footer className={styles.footer}>
        <div className={styles.footerContent}>
          <div className={styles.footerLogo}>
            <span className={styles.logoIcon}>🤟</span>
            <span className={styles.logoText}>SignTalk</span>
          </div>
          <p className={styles.footerText}>
            An AI-powered Greek Sign Language translation system built with PyTorch, FastAPI, and React.
          </p>
          <div className={styles.footerLinks}>
            <a href="#" className={styles.footerLink}>Documentation</a>
            <a href="#" className={styles.footerLink}>GitHub</a>
            <a href="#" className={styles.footerLink}>Contact</a>
          </div>
        </div>
        <div className={styles.footerBottom}>
          <p>© 2025 SignTalk Demo. Built for research purposes.</p>
        </div>
      </footer>
    </div>
  );
};

export default LandingPage;
