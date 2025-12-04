import React, { useState } from 'react';
import LandingPage from './components/LandingPage';
import ModeSelectionPage from './components/ModeSelectionPage';
import InferenceTest from './components/InferenceTest';

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

export default App;
