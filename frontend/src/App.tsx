import React, { useState } from 'react';
import LandingPage from './components/LandingPage';
import InferenceTest from './components/InferenceTest';

type Page = 'landing' | 'inference';

function App() {
  const [currentPage, setCurrentPage] = useState<Page>('landing');

  return (
    <div className="App">
      {currentPage === 'landing' && (
        <LandingPage onStart={() => setCurrentPage('inference')} />
      )}
      {currentPage === 'inference' && (
        <InferenceTest onBack={() => setCurrentPage('landing')} />
      )}
    </div>
  );
}

export default App;
